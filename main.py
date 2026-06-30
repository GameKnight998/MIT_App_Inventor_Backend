"""FastAPI backend for the Image Locator app.

Flow:
    MIT App Inventor -> POST /analyze (multipart image)
      -> save image to uploads/
      -> extract_exif()      (utils/exif.py)
      -> analyze_image()     (utils/vision.py)
      -> determine_location()(utils/osint.py)
      -> build_response()    (utils/response.py)
      -> return JSON
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from utils.exif import extract_exif
from utils.osint import determine_location
from utils.response import build_response, error_response
from utils.vision import analyze_image

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic", ".webp", ".tiff", ".bmp"}
MAX_BYTES = 25 * 1024 * 1024  # 25 MB

app = FastAPI(
    title="Image Locator Backend",
    description="Finds the likely location of an uploaded image using EXIF + AI vision OSINT.",
    version="1.0.0",
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


@app.post("/analyze")
async def analyze(image: UploadFile = File(...)) -> JSONResponse:
    """Receive an image, run the OSINT pipeline, and return location JSON."""
    suffix = Path(image.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        return JSONResponse(
            status_code=400,
            content=error_response(
                f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
                filename=image.filename,
            ),
        )

    # Save the upload with a unique name to avoid collisions.
    saved_name = f"{uuid.uuid4().hex}{suffix}"
    saved_path = UPLOAD_DIR / saved_name

    try:
        contents = await image.read()
        if len(contents) > MAX_BYTES:
            return JSONResponse(
                status_code=413,
                content=error_response(
                    "Image exceeds the 25 MB size limit.", filename=image.filename
                ),
            )
        saved_path.write_bytes(contents)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error_response(f"Failed to save upload: {exc}", filename=image.filename),
        )

    try:
        metadata = extract_exif(str(saved_path))
        vision = analyze_image(str(saved_path))
        location = determine_location(metadata, vision)
        payload = build_response(
            filename=image.filename or saved_name,
            metadata=metadata,
            vision=vision,
            location=location,
        )
        return JSONResponse(status_code=200, content=payload)
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=error_response(f"Analysis failed: {exc}", filename=image.filename),
        )
    finally:
        # Clean up the temporary upload.
        try:
            saved_path.unlink(missing_ok=True)
        except Exception:
            pass


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
