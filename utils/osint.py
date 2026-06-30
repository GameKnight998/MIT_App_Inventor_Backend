"""Combine EXIF and vision evidence into a single likely location.

Decision logic (highest trust first):
  1. EXIF GPS coordinates  -> authoritative, very high confidence.
  2. Vision model lat/lon   -> use the model's own confidence.
  3. Vision place name only -> location without coordinates, lower confidence.
  4. Nothing usable         -> unknown.
"""

from __future__ import annotations

from typing import Any, Optional


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _has_coords(d: Optional[dict[str, Any]]) -> bool:
    return bool(d) and d.get("latitude") is not None and d.get("longitude") is not None


def determine_location(
    metadata: dict[str, Any], vision: dict[str, Any]
) -> dict[str, Any]:
    """Fuse metadata + vision clues into a best-estimate location."""
    evidence: list[str] = []
    result: dict[str, Any] = {
        "location_name": None,
        "country": None,
        "latitude": None,
        "longitude": None,
        "confidence": 0.0,
        "source": "unknown",
        "evidence": evidence,
    }

    gps = metadata.get("gps") if metadata else None
    vguess = vision.get("best_guess_location") if vision else None
    vconf = float(vision.get("confidence", 0.0)) if vision else 0.0

    # 1. Authoritative EXIF GPS.
    if _has_coords(gps):
        result.update(
            {
                "latitude": gps["latitude"],
                "longitude": gps["longitude"],
                "confidence": 0.99,
                "source": "exif_gps",
            }
        )
        evidence.append(
            f"EXIF GPS coordinates {gps['latitude']}, {gps['longitude']}."
        )
        # If vision also named the place, enrich the label.
        if vguess and vguess.get("name"):
            result["location_name"] = vguess.get("name")
            result["country"] = vguess.get("country")
            evidence.append(f"Vision place label: {vguess.get('name')}.")
        return result

    # 2. Vision provided usable coordinates.
    if _has_coords(vguess):
        result.update(
            {
                "location_name": vguess.get("name"),
                "country": vguess.get("country"),
                "latitude": vguess.get("latitude"),
                "longitude": vguess.get("longitude"),
                "confidence": _clamp(vconf),
                "source": "vision_coordinates",
            }
        )
        if vguess.get("name"):
            evidence.append(f"Vision identified location: {vguess.get('name')}.")

    # 3. Vision named a place but gave no coordinates.
    elif vguess and vguess.get("name"):
        result.update(
            {
                "location_name": vguess.get("name"),
                "country": vguess.get("country"),
                "confidence": _clamp(vconf * 0.7),
                "source": "vision_place_name",
            }
        )
        evidence.append(f"Vision suggested place: {vguess.get('name')}.")

    # Add supporting clues from the vision model regardless of branch.
    if vision:
        for landmark in vision.get("landmarks", []) or []:
            evidence.append(f"Landmark detected: {landmark}.")
        for text in (vision.get("signage", []) or [])[:5]:
            evidence.append(f"Signage: {text}.")
        for lang in vision.get("languages", []) or []:
            evidence.append(f"Language seen: {lang}.")
        if vision.get("environment"):
            evidence.append(f"Environment: {vision['environment']}.")

    if metadata and metadata.get("timestamp"):
        evidence.append(f"Photo timestamp: {metadata['timestamp']}.")
    if metadata and metadata.get("camera", {}).get("model"):
        cam = metadata["camera"]
        evidence.append(
            f"Captured with {cam.get('make') or ''} {cam.get('model')}".strip() + "."
        )

    if result["source"] == "unknown":
        evidence.append("No GPS metadata and no confident visual location match.")

    return result
