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
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image

from utils.exif import extract_exif
from utils.osint import determine_location
from utils.response import build_response, error_response
from utils.vision import analyze_image

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

MAX_BYTES = 25 * 1024 * 1024  # 25 MB

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
        vision = analyze_image(str(saved_path))
        location = determine_location(metadata, vision)
        payload = build_response(
            filename=original_name or saved_name,
            metadata=metadata,
            vision=vision,
            location=location,
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


if __name__ == "__main__":
    import uvicorn

    # Render provides $PORT. reload defaults off; set RELOAD=1 locally if wanted.
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=os.getenv("RELOAD", "0") == "1",
    )
