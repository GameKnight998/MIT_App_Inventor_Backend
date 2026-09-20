"""Named-landmark and large water-body verification (Nominatim, no extra Overpass).

Generic map checks only ask "is there *a* lake within 3 km?". That lets a wrong
region survive if it happens to have a pond. This module asks two sharper
questions a human OSINT analyst would ask:

  1. If the guess is named "Lake Chelan", does the real Lake Chelan actually
     sit near these coordinates?
  2. If the photo shows a large lake (or a named landmark), is a *named* water
     body / that landmark nearby — not just any water tag?

All lookups go through cached Nominatim helpers so we do not add Overpass load.
Fail-soft: missing names, geocode misses, and network errors yield "unknown"
and never crash the pipeline.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

from utils.cache import parallel_map
from utils.geocode import forward_geocode_candidates, search_nearby

LANDMARK_ENABLED = os.getenv("LANDMARK_VERIFY_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
# How far a named lake/peak/city may sit from the guessed point and still count.
# Lakes are long; a shoreline photo can be tens of km from the OSM centroid.
PLACE_MATCH_KM = float(os.getenv("LANDMARK_PLACE_MATCH_KM", "50"))
WATER_SEARCH_KM = float(os.getenv("LANDMARK_WATER_SEARCH_KM", "25"))
LANDMARK_MATCH_KM = float(os.getenv("LANDMARK_MATCH_KM", "40"))

# Place types specific enough that a large name↔coord gap is a real contradiction.
_SPECIFIC_TYPES = {
    "lake",
    "reservoir",
    "pond",
    "lagoon",
    "bay",
    "sea",
    "ocean",
    "river",
    "peak",
    "volcano",
    "island",
    "islet",
    "glacier",
    "waterfall",
    "city",
    "town",
    "village",
    "hamlet",
    "suburb",
    "neighbourhood",
    "neighborhood",
    "museum",
    "attraction",
    "monument",
    "park",
}
_WATER_TYPES = {
    "lake",
    "reservoir",
    "lagoon",
    "bay",
    "sea",
    "ocean",
    "river",
    "fjord",
    "strait",
}
_SMALL_WATER_TYPES = {"pond", "stream", "ditch", "drain", "canal"}
_FEATURE_WORDS = (
    "lake",
    "loch",
    "reservoir",
    "mount",
    "mountain",
    "mt.",
    "mt ",
    "peak",
    "river",
    "falls",
    "bay",
    "gulf",
    "sea",
    "ocean",
    "glacier",
    "canyon",
    "island",
    "volcano",
    "dam",
    "park",
    "tower",
    "bridge",
    "cathedral",
    "temple",
    "mosque",
    "palace",
)
_BROAD_REGION = re.compile(
    r"\b(pacific northwest|midwest|new england|scandinavia|balkans|"
    r"eastern|western|northern|southern|central|upstate|greater)\b",
    re.I,
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _looks_like_named_feature(name: str) -> bool:
    lower = f" {name.lower()} "
    if _BROAD_REGION.search(name) and not any(
        w in lower for w in (" lake ", " river ", " mount ", " bay ")
    ):
        return False
    return any(w in lower for w in _FEATURE_WORDS)


def extract_named_features(vision: Optional[dict[str, Any]]) -> list[str]:
    """Unique landmark / named-feature strings from vision, first-seen order."""
    if not vision:
        return []
    seen: set[str] = set()
    names: list[str] = []

    def add(raw: Any) -> None:
        if not isinstance(raw, str):
            return
        text = raw.strip()
        if not text or len(text) < 3:
            return
        key = text.lower()
        if key in seen:
            return
        seen.add(key)
        names.append(text)

    for item in vision.get("landmarks") or []:
        add(item)
    for cand in vision.get("candidates") or []:
        if isinstance(cand, dict) and _looks_like_named_feature(str(cand.get("name") or "")):
            add(cand.get("name"))
    return names[:6]


def _scene_wants_water(vision: Optional[dict[str, Any]], expected: list[str]) -> bool:
    scene = (vision or {}).get("scene") or {}
    if scene.get("lake") or scene.get("river") or scene.get("ocean_or_sea"):
        return True
    return "water" in (expected or []) or "coast" in (expected or [])


def _is_water_hit(hit: dict[str, Any]) -> bool:
    kind = (hit.get("osm_type") or hit.get("addresstype") or "").lower()
    cat = (hit.get("category") or "").lower()
    if kind in _SMALL_WATER_TYPES:
        return False
    if kind in _WATER_TYPES:
        return True
    if cat in ("waterway", "natural") and kind in _WATER_TYPES | {"water"}:
        return True
    name = (hit.get("name") or hit.get("display_name") or "").lower()
    return any(w in name for w in ("lake", "reservoir", "bay", "sea", "river"))


def _geocode_one(query: str) -> Optional[dict[str, Any]]:
    hits = forward_geocode_candidates(query, limit=3)
    return hits[0] if hits else None


def prefetch_named_lookups(
    coords: list[tuple[float, float]],
    vision: Optional[dict[str, Any]] = None,
    expected: Optional[list[str]] = None,
) -> None:
    """Warm the cached Nominatim lookups this module will make, concurrently.

    `verify_named_and_water` is called once per candidate inside a sequential
    selection loop, and each call is network-bound. Firing the same lookups in
    parallel first means those calls hit the cache instead of the network, which
    removes most of the wall-clock cost without altering selection order.
    """
    if not LANDMARK_ENABLED:
        return

    jobs: list[Any] = []
    for name in extract_named_features(vision):
        jobs.append(lambda n=name: forward_geocode_candidates(n, limit=3))
    if _scene_wants_water(vision, expected or []):
        for lat, lon in coords:
            jobs.append(
                lambda la=lat, lo=lon: search_nearby(
                    "lake", la, lo, WATER_SEARCH_KM, limit=5
                )
            )
    if jobs:
        parallel_map(lambda fn: fn(), jobs)


def _empty_named() -> dict[str, Any]:
    return {
        "place_status": "unknown",
        "landmark_status": "unknown",
        "water_status": "unknown",
        "matched_landmarks": [],
        "matched_water": None,
        "place_distance_km": None,
        "note": "",
    }


def verify_named_and_water(
    latitude: float,
    longitude: float,
    vision: Optional[dict[str, Any]],
    expected: Optional[list[str]] = None,
    place_name: Optional[str] = None,
) -> dict[str, Any]:
    """Cross-check named landmarks and large water against a candidate point."""
    report = _empty_named()
    if not LANDMARK_ENABLED or latitude is None or longitude is None:
        report["note"] = "Named-feature verification disabled or no coordinates."
        return report

    notes: list[str] = []

    # 1. Does the candidate's own name actually live near these coordinates?
    if place_name and _looks_like_named_feature(place_name):
        geo = _geocode_one(place_name)
        if geo and geo.get("latitude") is not None:
            dist = _haversine_km(
                latitude, longitude, geo["latitude"], geo["longitude"]
            )
            report["place_distance_km"] = round(dist, 1)
            kind = (geo.get("osm_type") or geo.get("addresstype") or "").lower()
            specific = kind in _SPECIFIC_TYPES or _looks_like_named_feature(place_name)
            if specific and dist <= PLACE_MATCH_KM:
                report["place_status"] = "matched"
                report["place_name"] = place_name
                report["place_latitude"] = geo["latitude"]
                report["place_longitude"] = geo["longitude"]
                report["place_osm_type"] = geo.get("osm_type") or geo.get("addresstype")
                notes.append(
                    f"'{place_name}' geocodes {dist:.0f} km from this point."
                )
            elif specific and dist > PLACE_MATCH_KM:
                report["place_status"] = "missed"
                notes.append(
                    f"'{place_name}' is actually ~{dist:.0f} km from these "
                    f"coordinates (expected within {PLACE_MATCH_KM:.0f} km)."
                )

    # 2. Named landmarks from vision near the point.
    landmarks = extract_named_features(vision)
    matched: list[dict[str, Any]] = []
    for name in landmarks:
        geo = _geocode_one(name)
        if not geo or geo.get("latitude") is None:
            continue
        dist = _haversine_km(latitude, longitude, geo["latitude"], geo["longitude"])
        if dist <= LANDMARK_MATCH_KM:
            matched.append(
                {
                    "name": name,
                    "distance_km": round(dist, 1),
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "osm_type": geo.get("osm_type") or geo.get("addresstype"),
                    "category": geo.get("category"),
                }
            )
    if matched:
        report["landmark_status"] = "matched"
        report["matched_landmarks"] = matched
        notes.append(
            "Named landmark nearby: "
            + ", ".join(f"{m['name']} ({m['distance_km']} km)" for m in matched[:3])
            + "."
        )
    elif landmarks:
        report["landmark_status"] = "missed"
        notes.append(
            "Named landmark(s) "
            + ", ".join(landmarks[:3])
            + " not found near these coordinates."
        )

    # 3. Large named water body, when the scene claims a lake/coast/river.
    if _scene_wants_water(vision, expected or []):
        hits = search_nearby("lake", latitude, longitude, WATER_SEARCH_KM, limit=5)
        water = next((h for h in hits if _is_water_hit(h)), None)
        if water is None:
            hits = search_nearby("reservoir", latitude, longitude, WATER_SEARCH_KM, limit=3)
            water = next((h for h in hits if _is_water_hit(h)), None)
        if water is not None:
            dist = _haversine_km(
                latitude, longitude, water["latitude"], water["longitude"]
            )
            report["water_status"] = "matched"
            report["matched_water"] = {
                "name": water.get("name") or water.get("display_name"),
                "distance_km": round(dist, 1),
                "osm_type": water.get("osm_type"),
                "latitude": water["latitude"],
                "longitude": water["longitude"],
            }
            notes.append(
                f"Named water nearby: {report['matched_water']['name']} "
                f"({dist:.0f} km)."
            )
        else:
            report["water_status"] = "missed"
            notes.append(
                f"No named lake/reservoir found within {WATER_SEARCH_KM:.0f} km."
            )

    report["note"] = " ".join(notes)
    return report
