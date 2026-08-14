"""Format the final API response in a stable, App Inventor-friendly shape.

MIT App Inventor's Web component parses JSON most easily when the structure is
flat and predictable, so we expose the key fields at the top level while keeping
the detailed breakdown nested under `details`.

For display we never hand back a bare `null`: coordinate and source fields have
human-friendly string counterparts so the app can show them directly.
"""

from __future__ import annotations

from typing import Any, Optional

# Turn the internal machine source codes into readable labels (no underscores).
_SOURCE_LABELS = {
    "exif_gps": "GPS metadata",
    "vision_coordinates": "AI visual analysis",
    "vision_geocoded": "AI visual analysis + map lookup",
    "vision_place_name": "AI visual analysis",
    "exif_caption": "Photo caption",
    "unknown": "Could not determine",
    "error": "Error",
}


def _map_url(lat: Optional[float], lon: Optional[float]) -> Optional[str]:
    if lat is None or lon is None:
        return None
    return f"https://www.google.com/maps?q={lat},{lon}"


def _source_label(code: Optional[str]) -> str:
    return _SOURCE_LABELS.get(code or "unknown", "Could not determine")


def _coord_display(value: Optional[float]) -> str:
    """A never-null string for a single coordinate."""
    return str(value) if value is not None else "Unknown"


def _build_summary(
    name: Optional[str],
    country: Optional[str],
    lat: Optional[float],
    lon: Optional[float],
) -> str:
    """A single human-readable sentence describing the result."""
    place = name
    if name and country and country.lower() not in name.lower():
        place = f"{name}, {country}"

    has_coords = lat is not None and lon is not None

    if place and has_coords:
        return f"{place} (Latitude {lat} Longitude {lon})"
    if place:
        return f"{place} (exact coordinates unavailable)"
    if has_coords:
        return f"Latitude {lat} Longitude {lon}"
    return "Location could not be determined from this image."


def build_response(
    *,
    filename: str,
    metadata: dict[str, Any],
    vision: dict[str, Any],
    location: dict[str, Any],
    forensics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the JSON payload returned to the client."""
    lat = location.get("latitude")
    lon = location.get("longitude")
    name = location.get("location_name")
    country = location.get("country")
    region = location.get("region")
    address = location.get("address")
    source_code = location.get("source") or "unknown"

    return {
        "success": True,
        "filename": filename,
        # Flat, top-level fields that App Inventor can read directly.
        "location_name": name or "Unknown location",
        "country": country or "Unknown",
        "region": region or "Unknown",
        "address": address or (name or "Unknown location"),
        # Raw numeric coordinates (null when unknown) for maps / logic.
        "latitude": lat,
        "longitude": lon,
        # Never-null display strings for the "Latitude X Longitude Y" format.
        "latitude_display": _coord_display(lat),
        "longitude_display": _coord_display(lon),
        "coordinates": (
            f"Latitude {lat} Longitude {lon}"
            if lat is not None and lon is not None
            else "Latitude Unknown Longitude Unknown"
        ),
        "map_url": _map_url(lat, lon),
        "confidence": round(float(location.get("confidence", 0.0)), 3),
        # Friendly, underscore-free source label for display.
        "source": _source_label(source_code),
        "summary": _build_summary(name, country, lat, lon),
        # Plausibility check of the coordinates vs. the described scene.
        "verified": location.get("verification", {}).get("status", "skipped"),
        "warning": location.get("warning", ""),
        "alternatives": location.get("alternatives", []),
        # Locations that were tried and crossed out because the scene did not
        # match their real-world surroundings.
        "rejected": location.get("rejected", []),
        # Notable places near the result, from an independent OSINT source
        # (Wikipedia GeoSearch) for further context / corroboration.
        "nearby_places": location.get("nearby_places", []),
        # Candidates grouped into geographic clusters (#11) so the client can
        # show "one region we're fairly sure of" instead of scattered cities.
        "alternative_clusters": location.get("alternative_clusters", []),
        # What the file itself reveals about its origin (#10): screenshot,
        # edited, original camera, stripped metadata, and whether missing GPS
        # is expected.
        "forensics": location.get("forensics", forensics or {}),
        # Ordered, auditable narrative of how the answer was reached (#13).
        "reasoning_trace": location.get("reasoning_trace", []),
        # How tightly the scene's features localise the point, in metres
        # (null unless iterative narrowing ran).
        "precision_m": location.get("precision_m"),
        # Sun-position consistency check (null unless a timestamp was present).
        "solar": location.get("solar"),
        # Ground elevation at the result, in metres (null unless looked up).
        "elevation_m": location.get("elevation_m"),
        # Climate/elevation consistency check vs. the described scene (null
        # unless the scene had a checkable climate feature).
        "climate_check": location.get("climate_check"),
        "landmark_check": (location.get("verification") or {}).get("named"),
        # Full breakdown for debugging / richer clients.
        "details": {
            "source_code": source_code,
            "verification": location.get("verification"),
            "exif": metadata,
            "vision": vision,
            "evidence": location.get("evidence", []),
        },
    }


def error_response(message: str, *, filename: str | None = None) -> dict[str, Any]:
    """Consistent error envelope."""
    return {
        "success": False,
        "filename": filename,
        "error": message,
        "location_name": "Unknown location",
        "country": "Unknown",
        "region": "Unknown",
        "address": "Unknown location",
        "latitude": None,
        "longitude": None,
        "latitude_display": "Unknown",
        "longitude_display": "Unknown",
        "coordinates": "Latitude Unknown Longitude Unknown",
        "map_url": None,
        "confidence": 0.0,
        "source": "Error",
        "summary": message,
        "verified": "skipped",
        "warning": "",
        "alternatives": [],
        "rejected": [],
        "nearby_places": [],
        "alternative_clusters": [],
        "forensics": {},
        "reasoning_trace": [],
        "precision_m": None,
        "solar": None,
        "elevation_m": None,
        "climate_check": None,
        "landmark_check": None,
    }
