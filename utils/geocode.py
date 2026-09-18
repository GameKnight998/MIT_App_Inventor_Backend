"""Geocoding helpers backed by OpenStreetMap (no API key required).

- forward_geocode: place name / description  -> coordinates + address
- reverse_geocode: coordinates              -> readable address + components
- photon_geocode:  location-biased search, better at streets and businesses
- geocode_street:  resolve an address/business near a known point

Two back ends, because they are good at different things. Nominatim is the
authority on administrative places and natural features, but it is strict about
query wording and cannot be biased toward a point. Photon (Komoot's
Elasticsearch index over the same OSM data) is typo-tolerant and accepts a
`lat`/`lon` bias, which is exactly what street-level refinement needs: "Baker
Street, near here" rather than a global exact-match lookup.

Nominatim's usage policy asks for a descriptive User-Agent and <= 1 request per
second, which is fine for this app's one-image-at-a-time workload. All calls fail
soft: on any error (network, rate limit, no result) they return None/[] so the
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
PHOTON_URL = os.getenv("PHOTON_URL", "https://photon.komoot.io/api")
USER_AGENT = os.getenv(
    "GEOCODER_USER_AGENT", "ImageLocatorBackend/1.0 (MIT App Inventor capstone)"
)
TIMEOUT = float(os.getenv("GEOCODER_TIMEOUT", "8"))
PHOTON_ENABLED = os.getenv("PHOTON_ENABLED", "1") not in ("0", "false", "False")

# OSM keys that represent something specific enough to be a street-level answer.
_STREET_LEVEL_TYPES = ("house", "street", "locality")
_STREET_LEVEL_KEYS = (
    "amenity",
    "shop",
    "tourism",
    "leisure",
    "office",
    "building",
    "highway",
    "railway",
    "public_transport",
)


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


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _parse_photon(feature: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Normalise a Photon GeoJSON feature into our standard geocode shape."""
    try:
        lon, lat = feature["geometry"]["coordinates"][:2]
        lat = round(float(lat), 6)
        lon = round(float(lon), 6)
    except (KeyError, IndexError, TypeError, ValueError):
        return None

    props = feature.get("properties", {}) or {}
    # Photon returns components, not a formatted line, so build a readable one.
    parts = [
        " ".join(p for p in (props.get("housenumber"), props.get("street")) if p),
        props.get("name") if props.get("name") != props.get("street") else None,
        props.get("district"),
        props.get("city") or props.get("town") or props.get("village"),
        props.get("state"),
        props.get("postcode"),
        props.get("country"),
    ]
    display = ", ".join(dict.fromkeys(p for p in parts if p))

    return {
        "latitude": lat,
        "longitude": lon,
        "display_name": display or props.get("name"),
        "name": props.get("name")
        or props.get("street")
        or props.get("city")
        or props.get("state"),
        "country": props.get("country"),
        "country_code": (props.get("countrycode") or "").upper() or None,
        "region": props.get("state"),
        "importance": None,
        "category": props.get("osm_key"),
        "osm_type": props.get("osm_value"),
        "addresstype": props.get("type"),
        "street": props.get("street"),
        "housenumber": props.get("housenumber"),
        "city": props.get("city") or props.get("town") or props.get("village"),
        "postcode": props.get("postcode"),
        "provider": "photon",
    }


@cached(cache_empty=False)
def photon_geocode(
    query: str,
    limit: int = 5,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
) -> list[dict[str, Any]]:
    """Search Photon, optionally biased toward a coordinate.

    The bias is what makes this useful for refinement: "Main Street" is
    meaningless globally but unambiguous within a few km of a known point.
    """
    if not _enabled() or not PHOTON_ENABLED or not query or not query.strip():
        return []

    params: dict[str, Any] = {
        "q": query.strip(),
        "limit": max(1, min(int(limit), 20)),
        "lang": "en",
    }
    if latitude is not None and longitude is not None:
        params["lat"] = round(float(latitude), 5)
        params["lon"] = round(float(longitude), 5)

    try:
        resp = requests.get(
            PHOTON_URL, params=params, headers=_headers(), timeout=TIMEOUT
        )
        resp.raise_for_status()
        features = resp.json().get("features") or []
    except Exception:
        return []

    parsed = [_parse_photon(f) for f in features]
    return [p for p in parsed if p is not None]


def is_street_level(hit: dict[str, Any]) -> bool:
    """True when a geocode hit is specific enough to be an exact spot.

    A state centroid and a cafe both come back as "a result"; only one of them
    is an answer to "where exactly was this taken".
    """
    if hit.get("housenumber"):
        return True
    if (hit.get("addresstype") or "").lower() in _STREET_LEVEL_TYPES:
        return True
    return (hit.get("category") or "").lower() in _STREET_LEVEL_KEYS


def geocode_street(
    query: str,
    latitude: float,
    longitude: float,
    radius_km: float = 25.0,
    limit: int = 5,
) -> Optional[dict[str, Any]]:
    """Resolve an address / business / street near a known point.

    Fallback chain: Photon biased to the point (best at partial addresses and
    business names) -> Nominatim bounded to a box around it -> None. Results
    outside `radius_km` are discarded, because a same-named street on another
    continent is a wrong answer, not a fallback.
    """
    if not query or not query.strip() or latitude is None or longitude is None:
        return None

    def usable(hits: list[dict[str, Any]]) -> Optional[dict[str, Any]]:
        near = [
            h
            for h in hits
            if _distance_km(latitude, longitude, h["latitude"], h["longitude"])
            <= radius_km
        ]
        if not near:
            return None
        # Prefer a genuinely street-level hit, then the closest one.
        near.sort(
            key=lambda h: (
                0 if is_street_level(h) else 1,
                _distance_km(latitude, longitude, h["latitude"], h["longitude"]),
            )
        )
        best = dict(near[0])
        best["distance_km"] = round(
            _distance_km(latitude, longitude, best["latitude"], best["longitude"]), 2
        )
        return best

    hit = usable(photon_geocode(query, limit, latitude, longitude))
    if hit is not None:
        return hit

    hit = usable(search_nearby(query, latitude, longitude, radius_km, limit))
    if hit is not None:
        hit.setdefault("provider", "nominatim")
        return hit
    return None


def polite_pause() -> None:
    """Respect Nominatim's ~1 req/sec guidance between successive calls."""
    time.sleep(1.0)
