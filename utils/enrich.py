"""Enrich a resolved location by consulting additional OSINT data sources.

Right now this queries Wikipedia's GeoSearch API (free, no key) to find notable
places, landmarks and points of interest near the coordinates. This gives the
user "further information" about where the photo likely is, and lets the OSINT
engine cross-check the AI's landmark guesses against an independent source.

Fail-soft: any error returns empty results so the pipeline keeps working.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import requests

WIKI_API = os.getenv("WIKI_API_URL", "https://en.wikipedia.org/w/api.php")
USER_AGENT = os.getenv(
    "GEOCODER_USER_AGENT", "ImageLocatorBackend/1.0 (MIT App Inventor capstone)"
)
ENRICH_TIMEOUT = float(os.getenv("ENRICH_TIMEOUT", "8"))


def _enabled() -> bool:
    return os.getenv("ENRICH_ENABLED", "1") not in ("0", "false", "False")


def nearby_places(
    latitude: float, longitude: float, radius_m: Optional[int] = None, limit: Optional[int] = None
) -> list[dict[str, Any]]:
    """Return notable Wikipedia places near the coordinates, closest first."""
    if not _enabled() or latitude is None or longitude is None:
        return []

    # Wikipedia GeoSearch caps radius at 10 km.
    radius = min(int(radius_m or os.getenv("ENRICH_RADIUS_M", "10000")), 10000)
    count = int(limit or os.getenv("ENRICH_LIMIT", "8"))

    try:
        resp = requests.get(
            WIKI_API,
            params={
                "action": "query",
                "list": "geosearch",
                "gscoord": f"{latitude}|{longitude}",
                "gsradius": radius,
                "gslimit": count,
                "format": "json",
            },
            headers={"User-Agent": USER_AGENT},
            timeout=ENRICH_TIMEOUT,
        )
        resp.raise_for_status()
        hits = resp.json().get("query", {}).get("geosearch", [])
    except Exception:
        return []

    places: list[dict[str, Any]] = []
    for h in hits:
        title = h.get("title")
        if not title:
            continue
        places.append(
            {
                "title": title,
                "distance_m": h.get("dist"),
                "latitude": h.get("lat"),
                "longitude": h.get("lon"),
                "url": "https://en.wikipedia.org/wiki/"
                + title.replace(" ", "_"),
            }
        )
    places.sort(key=lambda p: p.get("distance_m") or 1e12)
    return places


def enrich_location(latitude: float, longitude: float) -> dict[str, Any]:
    """Gather supplementary OSINT context for a coordinate."""
    return {
        "source": "wikipedia_geosearch",
        "nearby_places": nearby_places(latitude, longitude),
    }
