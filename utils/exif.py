"""Read image metadata (GPS, camera model, timestamp) from EXIF tags."""

from __future__ import annotations

from typing import Any, Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


def _to_degrees(value: Any) -> Optional[float]:
    """Convert a (degrees, minutes, seconds) EXIF rational tuple to a float."""
    try:
        d, m, s = value
        return float(d) + float(m) / 60.0 + float(s) / 3600.0
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _parse_gps(gps_info: dict[str, Any]) -> Optional[dict[str, float]]:
    """Turn the raw GPSInfo block into decimal latitude/longitude."""
    lat = _to_degrees(gps_info.get("GPSLatitude")) if "GPSLatitude" in gps_info else None
    lon = _to_degrees(gps_info.get("GPSLongitude")) if "GPSLongitude" in gps_info else None

    if lat is None or lon is None:
        return None

    if str(gps_info.get("GPSLatitudeRef", "N")).upper() == "S":
        lat = -lat
    if str(gps_info.get("GPSLongitudeRef", "E")).upper() == "W":
        lon = -lon

    result: dict[str, float] = {"latitude": round(lat, 6), "longitude": round(lon, 6)}

    altitude = gps_info.get("GPSAltitude")
    if altitude is not None:
        try:
            alt = float(altitude)
            if str(gps_info.get("GPSAltitudeRef", 0)) in ("1", "b'\\x01'"):
                alt = -alt
            result["altitude_m"] = round(alt, 2)
        except (TypeError, ValueError):
            pass

    return result


def extract_exif(image_path: str) -> dict[str, Any]:
    """Extract GPS coordinates, camera model and timestamp from an image.

    Returns a dict that always has the same shape so downstream modules can
    rely on the keys existing even when a photo has no metadata.
    """
    metadata: dict[str, Any] = {
        "has_exif": False,
        "gps": None,
        "camera": {"make": None, "model": None},
        "timestamp": None,
        "raw": {},
    }

    try:
        with Image.open(image_path) as img:
            exif = img.getexif()
    except Exception as exc:  # corrupt file, unsupported format, etc.
        metadata["error"] = f"Could not read EXIF: {exc}"
        return metadata

    if not exif:
        return metadata

    metadata["has_exif"] = True
    decoded: dict[str, Any] = {}

    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        decoded[str(tag)] = value

    # GPS lives in its own IFD block.
    gps_ifd: dict[str, Any] = {}
    try:
        raw_gps = exif.get_ifd(0x8825)  # GPSInfo IFD pointer
        for key, value in raw_gps.items():
            gps_ifd[GPSTAGS.get(key, key)] = value
    except Exception:
        gps_ifd = {}

    if gps_ifd:
        metadata["gps"] = _parse_gps(gps_ifd)

    metadata["camera"] = {
        "make": _clean(decoded.get("Make")),
        "model": _clean(decoded.get("Model")),
        "lens": _clean(decoded.get("LensModel")),
        "software": _clean(decoded.get("Software")),
    }

    metadata["timestamp"] = (
        _clean(decoded.get("DateTimeOriginal"))
        or _clean(decoded.get("DateTime"))
        or _clean(decoded.get("DateTimeDigitized"))
    )

    # Keep a few human-readable extras without dumping binary blobs.
    metadata["raw"] = {
        k: _clean(v)
        for k, v in decoded.items()
        if isinstance(v, (str, int, float)) and k != "MakerNote"
    }

    return metadata


def _clean(value: Any) -> Optional[str]:
    """Normalise EXIF string values, stripping null padding."""
    if value is None:
        return None
    if isinstance(value, bytes):
        try:
            value = value.decode(errors="ignore")
        except Exception:
            return None
    if isinstance(value, str):
        value = value.replace("\x00", "").strip()
        return value or None
    return value
