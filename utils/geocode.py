"""Geocoding helpers backed by OpenStreetMap Nominatim (no API key required).

- forward_geocode: place name / description  -> coordinates + address
- reverse_geocode: coordinates              -> readable address + components

Nominatim's usage policy asks for a descriptive User-Agent and <= 1 request per
second, which is fine for this app's one-image-at-a-time workload. All calls fail
soft: on any error (network, rate limit, no result) they return None so the
pipeline keeps working without geocoding.
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org"
USER_AGENT = os.getenv(
    "GEOCODER_USER_AGENT", "ImageLocatorBackend/1.0 (MIT App Inventor capstone)"
)
TIMEOUT = float(os.getenv("GEOCODER_TIMEOUT", "8"))


def _enabled() -> bool:
    return os.getenv("GEOCODING_ENABLED", "1") not in ("0", "false", "False")


def _headers() -> dict[str, str]:
    return {"User-Agent": USER_AGENT, "Accept-Language": "en"}


def _short_name(address: dict[str, Any]) -> Optional[str]:
    """Pick the most specific human-friendly place label from OSM components."""
    for key in (
        "city",
        "town",
        "village",
        "municipality",
        "county",
        "state_district",
        "state",
        "region",
        "country",
    ):
        if address.get(key):
            return address[key]
    return None


def forward_geocode(query: str) -> Optional[dict[str, Any]]:
    """Resolve a place name/description to coordinates and a canonical name."""
    if not _enabled() or not query or not query.strip():
        return None

    try:
        resp = requests.get(
            f"{NOMINATIM_URL}/search",
            params={
                "q": query.strip(),
                "format": "jsonv2",
                "limit": 1,
                "addressdetails": 1,
            },
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception:
        return None

    if not results:
        return None

    top = results[0]
    try:
        lat = round(float(top["lat"]), 6)
        lon = round(float(top["lon"]), 6)
    except (KeyError, TypeError, ValueError):
        return None

    address = top.get("address", {}) or {}
    return {
        "latitude": lat,
        "longitude": lon,
        "display_name": top.get("display_name"),
        "name": top.get("name") or _short_name(address),
        "country": address.get("country"),
        "country_code": (address.get("country_code") or "").upper() or None,
        "region": address.get("state") or address.get("region"),
        "importance": top.get("importance"),
    }


def reverse_geocode(latitude: float, longitude: float) -> Optional[dict[str, Any]]:
    """Resolve coordinates to a readable address and its components."""
    if not _enabled() or latitude is None or longitude is None:
        return None

    try:
        resp = requests.get(
            f"{NOMINATIM_URL}/reverse",
            params={
                "lat": latitude,
                "lon": longitude,
                "format": "jsonv2",
                "addressdetails": 1,
                "zoom": 14,
            },
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None

    if not data or "error" in data:
        return None

    address = data.get("address", {}) or {}
    return {
        "display_name": data.get("display_name"),
        "name": _short_name(address),
        "country": address.get("country"),
        "country_code": (address.get("country_code") or "").upper() or None,
        "region": address.get("state") or address.get("region"),
        "city": address.get("city") or address.get("town") or address.get("village"),
    }


def polite_pause() -> None:
    """Respect Nominatim's ~1 req/sec guidance between successive calls."""
    time.sleep(1.0)
