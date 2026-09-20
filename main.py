"""FastAPI backend for the Image Locator app.

Flow:
    MIT App Inventor -> POST /analyze (image)
      -> save image to uploads/
      -> extract_exif()      (utils/exif.py)
      -> assess_forensics()  (utils/forensics.py, incl. synthetic-media check)
      -> enhance_image()     (utils/preprocess.py, only if quality is poor)
      -> analyze_image()     (utils/vision.py)
      -> determine_location()(utils/osint.py, incl. street-level refinement)
         || identify_vehicles() (utils/vehicle.py, concurrently)
      -> build_response()    (utils/response.py)
      -> return JSON

Ordering is deliberate. EXIF and forensics read the ORIGINAL bytes, because
re-encoding destroys the provenance evidence they depend on; only the vision call
sees the enhanced copy. Vehicle identification is an independent vision call, so
it runs concurrently with the location pipeline and is folded in afterwards.

Endpoints:
    POST /analyze        one image
    POST /analyze-video  a short clip (key frames are fused)
    POST /case           several images analysed and cross-referenced
    GET  /case/{id}      a stored case
    GET  /ui             browser interface

App Inventor's Web component cannot send multipart/form-data, so the upload
endpoints accept the image in whichever way App Inventor can actually send it:
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
from typing import Any, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from utils import casestore, geocode, streetlevel, synthetic, vehicle
from utils.cache import parallel_map
from utils.clustering import cluster_multi_image_candidates
from utils.exif import extract_exif
from utils.forensics import assess_forensics
from utils.osint import DEFINED_RADIUS_KM, apply_vehicle_signal, determine_location
from utils.preprocess import enhance_image, enhancement_available
from utils.response import build_response, error_response
from utils.vehicle import identify_vehicles, should_identify
from utils.video import (
    extract_key_frames,
    extract_video_gps,
    probe_video,
    video_support_available,
)
from utils.videofuse import merge_frame_visions
from utils.vision import SCENE_ROUTING_ENABLED, analyze_image, classify_scene
from utils.visionclient import model_name, provider, vision_available

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
WEB_DIR = BASE_DIR / "web"

MAX_BYTES = 25 * 1024 * 1024  # 25 MB (images)
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", str(60 * 1024 * 1024)))  # 60 MB
VIDEO_ENABLED = os.getenv("VIDEO_ENABLED", "1") not in ("0", "false", "False")
# Multi-image cases are bounded so one request cannot exceed the platform's
# request timeout or exhaust the vision-API budget.
CASE_MAX_IMAGES = int(os.getenv("CASE_MAX_IMAGES", "6"))
# How many images of a case to analyse at once. Each one makes several vision
# calls, so this stays low deliberately.
CASE_PARALLEL_IMAGES = int(os.getenv("CASE_PARALLEL_IMAGES", "2"))

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


if WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(WEB_DIR), html=True), name="ui")


@app.get("/")
def root(request: Request):
    """JSON for API clients, the web interface for browsers.

    App Inventor and curl get the machine-readable status they already expect;
    anyone who opens the URL in a browser lands on the interface instead of a
    bare JSON blob.
    """
    accept = (request.headers.get("accept") or "").lower()
    if "text/html" in accept and WEB_DIR.is_dir():
        return RedirectResponse(url="/ui/")
    return JSONResponse(
        content={
            "status": "ok",
            "message": "Image Locator backend is running.",
            "endpoints": ["/analyze", "/analyze-video", "/case", "/health", "/ui"],
        }
    )


@app.get("/health")
def health() -> dict[str, object]:
    vision_ok, vision_note = vision_available()
    return {
        "status": "healthy",
        "vision_enabled": vision_ok,
        "vision_provider": provider(),
        "vision_model": model_name() if vision_ok else None,
        "vision_note": vision_note,
        "video_enabled": VIDEO_ENABLED and video_support_available(),
        "enhancement_enabled": enhancement_available(),
        "cases_enabled": casestore.CASE_ENABLED,
        "web_ui": WEB_DIR.is_dir(),
        # Kept flat (no nesting) so App Inventor can read each flag directly.
        "street_refine_enabled": streetlevel.STREET_REFINE_ENABLED,
        "vehicle_id_enabled": vehicle.VEHICLE_ID_ENABLED,
        "synthetic_detection_enabled": synthetic.SYNTHETIC_ENABLED,
        "scene_routing_enabled": SCENE_ROUTING_ENABLED,
        "photon_enabled": geocode.PHOTON_ENABLED,
        "defined_radius_km": DEFINED_RADIUS_KM,
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


async def _extract_media_list(
    request: Request,
) -> list[tuple[bytes, Optional[str]]]:
    """Pull one or more images out of a request, for the multi-image case flow.

    Handles a multipart form with several file parts and a JSON body carrying a
    list of base64 strings, then falls back to the single-image extraction so a
    case can be built one upload at a time from App Inventor.
    """
    content_type = (request.headers.get("content-type") or "").lower()
    media: list[tuple[bytes, Optional[str]]] = []

    if "multipart/form-data" in content_type:
        try:
            form = await request.form()
        except Exception:
            return []
        for value in form.values():
            if hasattr(value, "read"):
                data = await value.read()
                if data:
                    media.append((data, getattr(value, "filename", None)))
        return media

    if "application/json" in content_type:
        import json

        try:
            body = await request.body()
            payload = json.loads(body)
        except (json.JSONDecodeError, ValueError):
            return []
        entries = payload.get("images") if isinstance(payload, dict) else None
        if isinstance(entries, list):
            for entry in entries:
                if isinstance(entry, str):
                    decoded = _decode_base64(entry)
                    if decoded:
                        media.append((decoded, None))
                elif isinstance(entry, dict):
                    raw = entry.get("image_base64") or entry.get("image")
                    if raw:
                        decoded = _decode_base64(str(raw))
                        if decoded:
                            media.append((decoded, entry.get("filename")))
            if media:
                return media

    single, name = await _extract_image_bytes(request)
    if single:
        media.append((single, name))
    return media


def _validate_image(contents: bytes) -> Optional[str]:
    """Return the true image format, or None if the bytes are not an image."""
    try:
        with Image.open(io.BytesIO(contents)) as probe:
            fmt = probe.format
            probe.verify()
        return fmt
    except Exception:
        return None


def _analyze_image_bytes(
    contents: Optional[bytes],
    original_name: Optional[str],
    content_type: str = "(none)",
) -> tuple[int, dict[str, Any]]:
    """Run the full single-image OSINT pipeline. Returns (status_code, payload).

    Shared by `/analyze` and `/case` so both paths behave identically. Synchronous
    on purpose: callers hand it to a worker thread, keeping the event loop free to
    serve other requests during the (slow) vision calls.
    """
    received_bytes = len(contents) if contents else 0
    head_hex = contents[:8].hex() if contents else ""

    if not contents:
        err = error_response(
            "No image found in request. Send multipart field 'image', "
            "raw image bytes, or a base64 string."
        )
        err["received"] = {"content_type": content_type, "bytes": received_bytes}
        return 400, err

    if received_bytes > MAX_BYTES:
        return 413, error_response(
            "Image exceeds the 25 MB size limit.", filename=original_name
        )

    fmt = _validate_image(contents)
    if fmt is None:
        err = error_response(
            "Uploaded data is not a valid image.", filename=original_name
        )
        err["received"] = {
            "content_type": content_type,
            "bytes": received_bytes,
            "head_hex": head_hex,
        }
        return 400, err

    suffix = _FORMAT_EXT.get(fmt or "", ".jpg")
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        saved_path.write_bytes(contents)
    except Exception as exc:
        return 500, error_response(
            f"Failed to save upload: {exc}", filename=original_name
        )

    enhanced_path: Optional[str] = None
    try:
        # EXIF and forensics must see the untouched original: enhancement
        # re-encodes, which would erase the provenance evidence they read.
        metadata = extract_exif(str(saved_path))
        forensics = assess_forensics(str(saved_path), fmt, metadata)

        quality = enhance_image(str(saved_path))
        vision_path = quality["path"]
        if quality.get("enhanced"):
            enhanced_path = vision_path

        vision = analyze_image(vision_path)

        # Vehicle identification is an independent vision call, so run it
        # alongside the location pipeline instead of after it.
        jobs = [
            lambda: determine_location(
                metadata, vision, forensics, image_path=vision_path
            )
        ]
        wants_vehicle = should_identify(vision)
        if wants_vehicle:
            jobs.append(lambda: identify_vehicles(vision_path))

        outputs = parallel_map(lambda fn: fn(), jobs)
        location = outputs[0]
        vehicle = outputs[1] if wants_vehicle and len(outputs) > 1 else None
        apply_vehicle_signal(location, vehicle)

        payload = build_response(
            filename=original_name or saved_name,
            metadata=metadata,
            vision=vision,
            location=location,
            forensics=forensics,
            # Drop the on-disk path; it is a temporary file the client never sees.
            image_quality={k: v for k, v in quality.items() if k != "path"},
        )
        return 200, payload
    except Exception as exc:
        return 500, error_response(
            f"Analysis failed: {exc}", filename=original_name
        )
    finally:
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass
        if enhanced_path and enhanced_path != str(saved_path):
            try:
                os.unlink(enhanced_path)
            except OSError:
                pass


@app.post("/analyze")
async def analyze(request: Request) -> JSONResponse:
    """Receive an image (any supported format), run the OSINT pipeline."""
    content_type = request.headers.get("content-type") or "(none)"
    contents, original_name = await _extract_image_bytes(request)
    print(
        f"[/analyze] content-type={content_type!r} "
        f"bytes={len(contents) if contents else 0} filename={original_name!r}"
    )
    status, payload = await run_in_threadpool(
        _analyze_image_bytes, contents, original_name, content_type
    )
    return JSONResponse(status_code=status, content=payload)


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

        # Frames of one clip share a scene, so classify once and reuse it rather
        # than paying for a routing call per frame.
        sharpest = max(frames, key=lambda fr: fr.get("blur_score") or 0.0)
        shared_scene = classify_scene(sharpest["path"]) if SCENE_ROUTING_ENABLED else None

        def analyze_frame(frame: dict) -> dict:
            # Video frames are often darker and softer than photos, so the same
            # enhancement the image path uses is worth more here, not less.
            quality = enhance_image(frame["path"])
            return analyze_image(quality["path"], scene_type=shared_scene)

        # Analyse the selected frames in parallel (network-bound vision calls).
        frame_visions = parallel_map(analyze_frame, frames)

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

        # Street-level refinement needs a single real frame to look at again;
        # the sharpest one gives it the best chance of reading small text.
        location = determine_location(
            metadata, merged_vision, forensics, image_path=sharpest["path"]
        )
        vehicle = (
            identify_vehicles(sharpest["path"])
            if should_identify(merged_vision)
            else None
        )
        apply_vehicle_signal(location, vehicle)

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


def _case_summary(payload: dict[str, Any]) -> dict[str, Any]:
    """The per-image fields a case view needs, without the full analysis blob."""
    return {
        "filename": payload.get("filename"),
        "success": payload.get("success"),
        "error": payload.get("error"),
        "location_name": payload.get("location_name"),
        "country": payload.get("country"),
        "region": payload.get("region"),
        "latitude": payload.get("latitude"),
        "longitude": payload.get("longitude"),
        "confidence": payload.get("confidence"),
        "verified": payload.get("verified"),
        "authenticity": payload.get("authenticity"),
        "precision_m": payload.get("precision_m"),
        "defined_radius_km": payload.get("defined_radius_km", DEFINED_RADIUS_KM),
        "defined_radius": payload.get("defined_radius") or f"{DEFINED_RADIUS_KM:g} km",
        "meets_defined_radius": payload.get("meets_defined_radius"),
        "source": payload.get("source"),
        "map_url": payload.get("map_url"),
        "alternatives": payload.get("alternatives", []),
    }


@app.post("/case")
async def analyze_case(request: Request) -> JSONResponse:
    """Analyse several images together and cross-reference their locations.

    Send multiple multipart file parts, or JSON `{"images": ["<base64>", ...]}`.
    Pass `?case_id=` to add images to an existing case, so a phone client can
    build a case up one photo at a time instead of in a single large upload.
    """
    if not casestore.CASE_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response("Case management is disabled on this server."),
        )

    media = await _extract_media_list(request)
    if not media:
        return JSONResponse(
            status_code=400,
            content=error_response(
                "No images found. Send multiple multipart file parts or JSON "
                '{"images": ["<base64>", ...]}.'
            ),
        )

    case_id = request.query_params.get("case_id") or ""
    title = request.query_params.get("title")
    try:
        if case_id:
            if not casestore.case_exists(case_id):
                return JSONResponse(
                    status_code=404,
                    content=error_response(f"Case {case_id} not found."),
                )
        else:
            case_id = casestore.create_case(title)

        existing = casestore.item_count(case_id)
        room = max(0, casestore.CASE_MAX_ITEMS - existing)
        batch = media[: min(CASE_MAX_IMAGES, room)]
        skipped = len(media) - len(batch)
        if not batch:
            return JSONResponse(
                status_code=413,
                content=error_response(
                    f"Case {case_id} already holds its maximum of "
                    f"{casestore.CASE_MAX_ITEMS} images."
                ),
            )

        def run(entry: tuple[bytes, Optional[str]]) -> dict[str, Any]:
            data, name = entry
            _status, payload = _analyze_image_bytes(data, name)
            return payload

        payloads = await run_in_threadpool(
            parallel_map, run, batch, CASE_PARALLEL_IMAGES
        )

        for (data, name), payload in zip(batch, payloads):
            payload.setdefault("filename", name)
            casestore.add_item(case_id, name, payload)

        items = casestore.list_items(case_id)
        clusters = cluster_multi_image_candidates(items)

        response: dict[str, Any] = {
            "success": True,
            "case_id": case_id,
            "images_in_case": len(items),
            "images_added": len(batch),
            "results": [_case_summary(item["payload"]) for item in items],
            "clusters": clusters["clusters"],
            "consensus": clusters["consensus"],
            "heatmap": clusters["heatmap"],
            "images_located": clusters["images_located"],
            "images_unlocated": clusters["images_unlocated"],
            "summary": (
                clusters["consensus"]["note"]
                if clusters.get("consensus")
                else "No location is supported by enough images to call it a consensus."
            ),
        }
        if skipped > 0:
            response["warning"] = (
                f"{skipped} image(s) were not analysed: a single request handles "
                f"at most {CASE_MAX_IMAGES}, and a case holds "
                f"{casestore.CASE_MAX_ITEMS}."
            )
        return JSONResponse(status_code=200, content=response)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error_response(f"Case analysis failed: {exc}"),
        )


@app.get("/case/{case_id}")
def read_case(case_id: str) -> JSONResponse:
    """Retrieve a stored case with its combined clustering."""
    if not casestore.CASE_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response("Case management is disabled on this server."),
        )
    case = casestore.get_case(case_id)
    if case is None:
        return JSONResponse(
            status_code=404, content=error_response(f"Case {case_id} not found.")
        )

    clusters = cluster_multi_image_candidates(case["items"])
    return JSONResponse(
        status_code=200,
        content={
            "success": True,
            "case_id": case["case_id"],
            "title": case["title"],
            "created_at": case["created_at"],
            "images_in_case": len(case["items"]),
            "results": [_case_summary(item["payload"]) for item in case["items"]],
            "clusters": clusters["clusters"],
            "consensus": clusters["consensus"],
            "heatmap": clusters["heatmap"],
            "images_located": clusters["images_located"],
            "images_unlocated": clusters["images_unlocated"],
        },
    )


@app.get("/cases")
def list_cases() -> JSONResponse:
    """List recent cases (id, title, image count)."""
    if not casestore.CASE_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response("Case management is disabled on this server."),
        )
    return JSONResponse(
        status_code=200, content={"success": True, "cases": casestore.list_cases()}
    )


@app.delete("/case/{case_id}")
def remove_case(case_id: str) -> JSONResponse:
    if not casestore.CASE_ENABLED:
        return JSONResponse(
            status_code=503,
            content=error_response("Case management is disabled on this server."),
        )
    if not casestore.delete_case(case_id):
        return JSONResponse(
            status_code=404, content=error_response(f"Case {case_id} not found.")
        )
    return JSONResponse(status_code=200, content={"success": True, "case_id": case_id})


@app.on_event("startup")
def _startup() -> None:
    """Create the case database up front so the first request is not the one
    that discovers a filesystem problem."""
    if casestore.CASE_ENABLED:
        try:
            casestore.init_db()
        except Exception as exc:  # a read-only disk must not stop the API
            print(f"[startup] case store unavailable: {exc}")


if __name__ == "__main__":
    import uvicorn

    # Render provides $PORT. reload defaults off; set RELOAD=1 locally if wanted.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )
