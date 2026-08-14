"""FastAPI backend for the Image Locator app.

Flow:
    MIT App Inventor -> POST /analyze (image)
      -> save image to uploads/
      -> extract_exif()      (utils/exif.py)
      -> analyze_image()     (utils/vision.py)
      -> determine_location()(utils/osint.py)
      -> build_response()    (utils/response.py)
      -> return JSON

App Inventor's Web component cannot send multipart/form-data, so /analyze
accepts the image in whichever way App Inventor can actually send it:
  1. multipart/form-data field named "image" (browsers, Postman, curl -F)
  2. the raw image bytes as the request body (Web.PostFile)
  3. a base64 string (Web.PostText, or JSON {"image_base64": "..."})
"""

from __future__ import annotations

import base64
import binascii
import io
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from utils.cache import parallel_map
from utils.exif import extract_exif
from utils.forensics import assess_forensics
from utils.osint import determine_location
from utils.response import build_response, error_response
from utils.video import (
    extract_key_frames,
    extract_video_gps,
    probe_video,
    video_support_available,
)
from utils.videofuse import merge_frame_visions
from utils.vision import analyze_image

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_BYTES = 25 * 1024 * 1024  # 25 MB (images)
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(60 * 1024 * 1024)))  # 60 MB
VIDEO_ENABLED = os.getenv("VIDEO_ENABLED", "1") not in ("0", "false", "False")

# Map Pillow's detected format to a file extension for saving.
_FORMAT_EXT = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
    "TIFF": ".tiff",
    "BMP": ".bmp",
    "GIF": ".gif",
    "HEIF": ".heic",
}

# Content-type / extension -> saved suffix for videos.
_VIDEO_EXT = {
    "video/mp4": ".mp4",
    "video/quicktime": ".mov",
    "video/webm": ".webm",
    "video/x-matroska": ".mkv",
    "video/x-msvideo": ".avi",
    "video/3gpp": ".3gp",
}

app = FastAPI(
    title="Image Locator Backend",
    description="Finds the likely location of an uploaded image using EXIF + AI vision OSINT.",
    version="1.1.0",
)

# App Inventor is not a browser so it ignores CORS, but enabling it lets you
# test from web pages / the Swagger UI without errors.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict[str, str]:
    return {"status": "ok", "message": "Image Locator backend is running."}


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "healthy",
        "vision_enabled": bool(os.getenv("OPENAI_API_KEY")),
        "video_enabled": VIDEO_ENABLED and video_support_available(),
    }


def _decode_base64(text: str) -> Optional[bytes]:
    """Decode a base64 string, tolerating data-URI prefixes and whitespace."""
    text = text.strip()
    if text.startswith("data:") and "," in text:
        text = text.split(",", 1)[1]
    text = text.replace("\n", "").replace("\r", "").replace(" ", "")
    try:
        return base64.b64decode(text, validate=False)
    except (binascii.Error, ValueError):
        return None


async def _extract_image_bytes(
    request: Request,
) -> tuple[Optional[bytes], Optional[str]]:
    """Pull image bytes out of whatever format the client sent.

    Returns (bytes, original_filename) or (None, None) if nothing usable.

    NOTE: we deliberately do NOT declare an UploadFile/File() parameter on the
    route. Doing so makes FastAPI parse EVERY request body as a form, which
    crashes on raw image bytes ("Too many fields"). Instead we read the body
    ourselves and only invoke multipart parsing when the content type says so.
    """
    content_type = (request.headers.get("content-type") or "").lower()

    # 1. multipart/form-data: parse the form and grab the first file part.
    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception:
            return None, None
        for value in form.values():
            if hasattr(value, "read"):  # it's an UploadFile
                return await value.read(), getattr(value, "filename", None)
        return None, None

    body = await request.body()
    if not body:
        return None, None

    # 2. JSON payload: {"image_base64": "..."} or {"image": "..."}
    if "application/json" in content_type:
        import json

        try:
            data = json.loads(body)
            b64 = data.get("image_base64") or data.get("image")
            if b64:
                return _decode_base64(b64), data.get("filename")
        except (json.JSONDecodeError, AttributeError):
            return None, None
        return None, None

    # 3. Body that is actually a base64 text string (Web.PostText).
    #    Heuristic: looks like text and decodes cleanly to a valid image.
    stripped = body.lstrip()[:64]
    looks_like_base64 = stripped.startswith(b"data:") or all(
        chr(c) in
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\r\n "
        for c in stripped
    )
    if looks_like_base64:
        decoded = _decode_base64(body.decode("utf-8", errors="ignore"))
        if decoded:
            return decoded, None

    # 4. Raw image bytes as the request body (Web.PostFile).
    return body, None


@app.post("/analyze")
async def analyze(request: Request) -> JSONResponse:
    """Receive an image (any supported format), run the OSINT pipeline."""
    content_type = request.headers.get("content-type") or "(none)"
    contents, original_name = await _extract_image_bytes(request)
    received_bytes = len(contents) if contents else 0
    head_hex = contents[:8].hex() if contents else ""
    print(
        f"[/analyze] content-type={content_type!r} bytes={received_bytes} "
        f"head={head_hex} filename={original_name!r}"
    )

    if not contents:
        err = error_response(
            "No image found in request. Send multipart field 'image', "
            "raw image bytes, or a base64 string."
        )
        err["received"] = {"content_type": content_type, "bytes": received_bytes}
        return JSONResponse(status_code=400, content=err)

    if len(contents) > MAX_BYTES:
        return JSONResponse(
            status_code=413,
            content=error_response("Image exceeds the 25 MB size limit.", filename=original_name),
        )

    # Validate it is a real image and detect its true format (not by extension).
    try:
        with Image.open(io.BytesIO(contents)) as probe:
            fmt = probe.format
            probe.verify()
    except Exception:
        err = error_response("Uploaded data is not a valid image.", filename=original_name)
        err["received"] = {
            "content_type": content_type,
            "bytes": received_bytes,
            "head_hex": head_hex,
        }
        return JSONResponse(status_code=400, content=err)

    suffix = _FORMAT_EXT.get(fmt or "", ".jpg")
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        saved_path.write_bytes(contents)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error_response(f"Failed to save upload: {exc}", filename=original_name),
        )

    try:
        metadata = extract_exif(str(saved_path))
        forensics = assess_forensics(str(saved_path), fmt, metadata)
        vision = analyze_image(str(saved_path))
        location = determine_location(metadata, vision, forensics)
        payload = build_response(
            filename=original_name or saved_name,
            metadata=metadata,
            vision=vision,
            location=location,
            forensics=forensics,
        )
        return JSONResponse(status_code=200, content=payload)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error_response(f"Analysis failed: {exc}", filename=original_name),
        )
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass


def _video_suffix(content_type: str, filename: Optional[str]) -> str:
    """Pick a file suffix for the saved video from its type or name."""
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in _VIDEO_EXT:
        return _VIDEO_EXT[ct]
    if filename and "." in filename:
        ext = "." + filename.rsplit(".", 1)[1].lower()
        if ext in _VIDEO_EXT.values():
            return ext
    return ".mp4"


def _video_forensics(
    video_info: dict, gps: Optional[dict], frames: list[dict]
) -> dict:
    """A forensics-style provenance block for a video (parallels image forensics)."""
    notes: list[str] = [
        f"Analysed {len(frames)} key frame(s) sampled from the video."
    ]
    if gps:
        notes.append("GPS coordinates are embedded in the video container.")
    else:
        notes.append("No GPS embedded in the video (common for shared/re-encoded clips).")
    if video_info.get("duration_s"):
        notes.append(f"Clip duration ~{video_info['duration_s']}s.")
    return {
        "format": "VIDEO",
        "has_gps": gps is not None,
        "gps_missing_expected": gps is None,
        "is_screenshot": False,
        "edited": False,
        "metadata_stripped": gps is None,
        "duration_s": video_info.get("duration_s"),
        "fps": video_info.get("fps"),
        "frame_count": video_info.get("frame_count"),
        "frames_analyzed": len(frames),
        "notes": notes,
    }


@app.post("/analyze-video")
async def analyze_video(request: Request) -> JSONResponse:
    """Locate a short video by extracting a few key frames and fusing them.

    Accepts the video the same ways `/analyze` accepts images (multipart field,
    raw body bytes, or base64). Extracts sharp, distinct frames, runs the normal
    vision+OSINT pipeline on each, then fuses the results so clues spread across
    the clip (a fleeting sign, a revealed landmark, agreeing frames) combine.
    """
    if not VIDEO_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response("Video analysis is disabled on this server."),
        )
    if not video_support_available():
        return JSONResponse(
            status_code=503,
            content=error_response(
                "Video support is unavailable (OpenCV not installed on the server)."
            ),
        )

    content_type = request.headers.get("content-type") or "(none)"
    contents, original_name = await _extract_image_bytes(request)
    received_bytes = len(contents) if contents else 0
    print(
        f"[/analyze-video] content-type={content_type!r} bytes={received_bytes} "
        f"filename={original_name!r}"
    )

    if not contents:
        err = error_response(
            "No video found in request. Send multipart field 'image', raw video "
            "bytes, or a base64 string."
        )
        err["received"] = {"content_type": content_type, "bytes": received_bytes}
        return JSONResponse(status_code=400, content=err)

    if len(contents) > MAX_VIDEO_BYTES:
        return JSONResponse(
            status_code=413,
            content=error_response(
                f"Video exceeds the {MAX_VIDEO_BYTES // (1024 * 1024)} MB size limit.",
                filename=original_name,
            ),
        )

    suffix = _video_suffix(content_type, original_name)
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name
    frames_dir = UPLOAD_DIR / f"{saved_path.stem}_frames"

    try:
        saved_path.write_bytes(contents)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error_response(f"Failed to save upload: {exc}", filename=original_name),
        )

    try:
        info = probe_video(str(saved_path))
        if not info.get("ok"):
            return JSONResponse(
                status_code=400,
                content=error_response(
                    info.get("error") or "Could not read the video.",
                    filename=original_name,
                ),
            )

        frames = extract_key_frames(str(saved_path), str(frames_dir))
        if not frames:
            return JSONResponse(
                status_code=422,
                content=error_response(
                    "Could not extract any usable frames from the video.",
                    filename=original_name,
                ),
            )

        video_gps = extract_video_gps(str(saved_path))

        # Analyse the selected frames in parallel (network-bound vision calls).
        frame_visions = parallel_map(
            lambda fr: analyze_image(fr["path"]), frames
        )

        merged_vision = merge_frame_visions(frame_visions)

        # Synthesize metadata: embedded video GPS is authoritative, like EXIF GPS.
        metadata = {
            "has_exif": video_gps is not None,
            "gps": video_gps,
            "camera": {"make": None, "model": None},
            "timestamp": None,
            "raw": {},
        }
        forensics = _video_forensics(info, video_gps, frames)

        location = determine_location(metadata, merged_vision, forensics)
        payload = build_response(
            filename=original_name or saved_name,
            metadata=metadata,
            vision=merged_vision,
            location=location,
            forensics=forensics,
        )

        # Video-specific reporting so the client can show what was used.
        payload["media_type"] = "video"
        payload["video"] = {
            "duration_s": info.get("duration_s"),
            "fps": info.get("fps"),
            "frame_count": info.get("frame_count"),
            "frames_analyzed": len(frames),
            "embedded_gps": video_gps,
        }
        payload["contributing_frames"] = [
            {
                "index": fr["index"],
                "timestamp_s": fr["timestamp_s"],
                "blur_score": fr["blur_score"],
                "candidates": [
                    c.get("name")
                    for c in (fv.get("candidates") or [])[:3]
                    if c.get("name")
                ],
                "clues_found": bool(
                    (fv.get("ocr_text") or [])
                    or (fv.get("signage") or [])
                    or (fv.get("landmarks") or [])
                ),
            }
            for fr, fv in zip(frames, frame_visions)
        ]
        return JSONResponse(status_code=200, content=payload)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error_response(f"Video analysis failed: {exc}", filename=original_name),
        )
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass
        shutil.rmtree(frames_dir, ignore_errors=True)


if __name__ == "__main__":
    import uvicorn

    # Render provides $PORT. reload defaults off; set RELOAD=1 locally if wanted.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )
