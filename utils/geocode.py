"""Geocoding helpers backed by OpenStreetMap Nominatim (no API key required).

- forward_geocode: place name / description  -> coordinates + address
- reverse_geocode: coordinates              -> readable address + components

Nominatim's usage policy asks for a descriptive User-Agent and <= 1 request per
second, which is fine for this app's one-image-at-a-time workload. All calls fail
soft: on any error (network, rate limit, no result) they return None so the
pipeline keeps working without geocoding.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any, Optional

import requests

from utils.cache import cached

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


def _parse_result(item: dict[str, Any]) -> Optional[dict[str, Any]]:
    try:
        lat = round(float(item["lat"]), 6)
        lon = round(float(item["lon"]), 6)
    except (KeyError, TypeError, ValueError):
        return None
    address = item.get("address", {}) or {}
    return {
        "latitude": lat,
        "longitude": lon,
        "display_name": item.get("display_name"),
        "name": item.get("name") or _short_name(address),
        "country": address.get("country"),
        "country_code": (address.get("country_code") or "").upper() or None,
        "region": address.get("state") or address.get("region"),
        "importance": item.get("importance"),
        # Nominatim jsonv2: category/type distinguish a lake from a state centroid.
        "category": item.get("category") or item.get("class"),
        "osm_type": item.get("type"),
        "addresstype": item.get("addresstype"),
    }


@cached(cache_empty=False)
def forward_geocode_candidates(query: str, limit: int = 3) -> list[dict[str, Any]]:
    """Resolve a place name to up to `limit` ranked real-world matches.

    A single HTTP request returns several candidates, which lets the caller try
    the next real place when the first one fails verification (e.g. an ambiguous
    name like "Springfield" that exists in many states).
    """
    if not _enabled() or not query or not query.strip():
        return []

    try:
        resp = requests.get(
            f"{NOMINATIM_URL}/search",
            params={
                "q": query.strip(),
                "format": "jsonv2",
                "limit": max(1, min(int(limit), 10)),
                "addressdetails": 1,
            },
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception:
        return []

    parsed = [_parse_result(item) for item in results]
    return [p for p in parsed if p is not None]


def forward_geocode(query: str) -> Optional[dict[str, Any]]:
    """Resolve a place name/description to the single best match (or None)."""
    candidates = forward_geocode_candidates(query, limit=1)
    return candidates[0] if candidates else None


@cached(cache_empty=False)
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


@cached(cache_empty=False)
def search_nearby(
    query: str,
    latitude: float,
    longitude: float,
    radius_km: float = 25.0,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Search Nominatim restricted to a bounding box around a coordinate.

    Used to answer "is there a named lake/landmark near this guess?" without
    another Overpass round-trip. Fail-soft: returns [] on any error.
    """
    if not _enabled() or not query or latitude is None or longitude is None:
        return []

    lat = round(float(latitude), 3)
    lon = round(float(longitude), 3)
    radius_km = max(1.0, float(radius_km))
    dlat = radius_km / 111.0
    dlon = radius_km / (111.0 * max(0.2, math.cos(math.radians(lat))))
    # viewbox: min_lon, max_lat, max_lon, min_lat
    viewbox = f"{lon - dlon},{lat + dlat},{lon + dlon},{lat - dlat}"

    try:
        resp = requests.get(
            f"{NOMINATIM_URL}/search",
            params={
                "q": query.strip(),
                "format": "jsonv2",
                "limit": max(1, min(int(limit), 10)),
                "addressdetails": 1,
                "viewbox": viewbox,
                "bounded": 1,
            },
            headers=_headers(),
            timeout=TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json()
    except Exception:
        return []

    parsed = [_parse_result(item) for item in results]
    return [p for p in parsed if p is not None]


def polite_pause() -> None:
    """Respect Nominatim's ~1 req/sec guidance between successive calls."""
    time.sleep(1.0)
