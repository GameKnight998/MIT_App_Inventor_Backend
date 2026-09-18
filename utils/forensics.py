"""Image forensics: what the file itself says about its origin.

Two questions, both of which change how the result should be read:

  1. Is a MISSING GPS tag *expected* (a screenshot, or an export that had its
     metadata stripped) or *surprising* (an original camera photo that would
     normally carry GPS)? That context changes how we talk about a missing
     location and feeds the reasoning trace, without hard-blocking analysis.
  2. Is this a photograph at all? Synthetic-media assessment is delegated to
     `utils.synthetic` and folded in here, because a generated image has no real
     location and the pipeline must say so rather than confidently geolocate a
     picture of a place that does not exist.

Fail-soft: any error returns a minimal, safe assessment so the pipeline runs.
"""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image

from utils.synthetic import assess_synthetic, is_unreliable

# Software signatures that indicate the file was edited / re-exported.
_EDIT_SOFTWARE = (
    "photoshop",
    "lightroom",
    "gimp",
    "snapseed",
    "pixlr",
    "affinity",
    "capture one",
    "luminar",
    "picasa",
    "paint.net",
    "acdsee",
)
# File types a real camera produces (GPS would normally be present).
_PHOTO_FORMATS = ("JPEG", "HEIF", "HEIC", "TIFF")
# File types typical of screenshots / UI captures and web exports.
_SCREENSHOT_FORMATS = ("PNG", "WEBP")


def _dimensions(image_path: str) -> tuple[Optional[int], Optional[int]]:
    try:
        with Image.open(image_path) as img:
            return img.width, img.height
    except Exception:
        return None, None


def assess_forensics(
    image_path: str,
    image_format: Optional[str],
    metadata: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Infer the provenance of the file from its format and EXIF metadata."""
    metadata = metadata or {}
    fmt = (image_format or "").upper()
    camera = metadata.get("camera", {}) or {}
    make = camera.get("make")
    model = camera.get("model")
    software = (camera.get("software") or "").lower()
    has_exif = bool(metadata.get("has_exif"))
    has_gps = metadata.get("gps") is not None

    edited = any(sig in software for sig in _EDIT_SOFTWARE)
    is_original_camera = bool(make or model)
    width, height = _dimensions(image_path)

    # Screenshot: a UI-style format with no camera identity and no GPS.
    is_screenshot = (
        fmt in _SCREENSHOT_FORMATS and not is_original_camera and not has_gps
    )
    # Stripped/re-encoded: a photo-type file with no EXIF at all, or EXIF that
    # has had the camera identity and GPS removed (typical of social re-uploads).
    metadata_stripped = (fmt in _PHOTO_FORMATS and not has_exif) or (
        has_exif and not is_original_camera and not has_gps and not is_screenshot
    )

    notes: list[str] = []
    if is_original_camera:
        label = f"{make or ''} {model or ''}".strip()
        notes.append(f"Original camera metadata present ({label}).")
    if edited:
        notes.append(f"Edited/exported with {camera.get('software')}.")
    if is_screenshot:
        notes.append("Looks like a screenshot (UI image format, no camera data).")
    if metadata_stripped and not is_screenshot:
        notes.append("Metadata appears stripped or re-encoded.")

    # Is a missing GPS tag expected here, or is it surprising?
    if has_gps:
        gps_missing_expected = False
        notes.append("GPS coordinates are embedded in the file.")
    elif is_screenshot or metadata_stripped or edited or not has_exif:
        gps_missing_expected = True
        notes.append("No GPS, which is expected for this kind of file.")
    else:
        gps_missing_expected = False
        notes.append(
            "No GPS despite original camera metadata - unusual (possibly removed)."
        )

    synthetic = assess_synthetic(image_path, metadata, fmt, width, height)
    if synthetic.get("note") and synthetic.get("status") != "authentic":
        notes.append(synthetic["note"])
    if synthetic.get("provenance_credentials"):
        notes.append("File carries signed Content Credentials (C2PA).")

    return {
        "format": fmt or None,
        "width": width,
        "height": height,
        "has_exif": has_exif,
        "has_gps": has_gps,
        "is_original_camera": is_original_camera,
        "is_screenshot": is_screenshot,
        "edited": edited,
        "editing_software": camera.get("software") if edited else None,
        "metadata_stripped": metadata_stripped,
        "gps_missing_expected": gps_missing_expected,
        # Whether the image can be trusted to depict a real place at all.
        "synthetic": synthetic,
        "authenticity": synthetic.get("status", "unknown"),
        "trustworthy_as_photo": not is_unreliable(synthetic),
        "notes": notes,
    }
