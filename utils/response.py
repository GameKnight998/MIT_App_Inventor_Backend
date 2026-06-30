"""Format the final API response in a stable, App Inventor-friendly shape.

MIT App Inventor's Web component parses JSON most easily when the structure is
flat and predictable, so we expose the key fields at the top level while keeping
the detailed breakdown nested under `details`.
"""

from __future__ import annotations

from typing import Any


def build_response(
    *,
    filename: str,
    metadata: dict[str, Any],
    vision: dict[str, Any],
    location: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the JSON payload returned to the client."""
    return {
        "success": True,
        "filename": filename,
        # Flat, top-level fields that App Inventor can read directly.
        "location_name": location.get("location_name"),
        "country": location.get("country"),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "confidence": round(float(location.get("confidence", 0.0)), 3),
        "source": location.get("source"),
        # Full breakdown for debugging / richer clients.
        "details": {
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
        "location_name": None,
        "country": None,
        "latitude": None,
        "longitude": None,
        "confidence": 0.0,
        "source": "error",
    }
