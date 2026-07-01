"""Cross-check a resolved location against the scene the image actually shows.

Example problem this solves: the AI/geocoder places a lakeside balcony photo at
coordinates that have no water nearby. We use OpenStreetMap's Overpass API (free,
no key) to look for the geographic features implied by the image (water, coast,
mountains, forest) within a radius of the guessed point. If the expected features
are missing (or far away), we lower confidence and attach a warning.

All calls fail soft: any error returns a "skipped" report so the pipeline keeps
working without verification.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

import requests

OVERPASS_URL = os.getenv("OVERPASS_URL", "https://overpass-api.de/api/interpreter")
VERIFY_RADIUS_M = int(os.getenv("VERIFY_RADIUS_M", "3000"))
VERIFY_TIMEOUT = float(os.getenv("VERIFY_TIMEOUT", "20"))
# Beyond this distance a "confirmed" feature is treated as only approximate.
PRECISION_M = int(os.getenv("VERIFY_PRECISION_M", "800"))

# Words in the vision analysis that imply a checkable geographic feature.
_CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "water": [
        "lake", "pond", "river", "reservoir", "lagoon", "waterfront",
        "marina", "harbor", "harbour", "canal", "waterway", "bay", "water",
        "riverside", "lakeside", "waterside",
    ],
    "coast": [
        "beach", "coast", "coastal", "seaside", "shore", "ocean", "sea",
        "gulf", "seafront", "oceanfront",
    ],
    "mountain": [
        "mountain", "mountainous", "peak", "alpine", "summit", "ridge",
        "volcano", "highland", "glacier",
    ],
    "forest": [
        "forest", "woodland", "woods", "jungle", "conifer", "wooded",
        "rainforest", "pine forest",
    ],
}


def _enabled() -> bool:
    return os.getenv("VERIFY_ENABLED", "1") not in ("0", "false", "False")


def detect_expected_features(vision: Optional[dict[str, Any]]) -> list[str]:
    """Infer which geographic features the image should be near."""
    if not vision:
        return []

    parts: list[str] = []
    for key in ("environment", "terrain", "vegetation", "climate", "reasoning"):
        value = vision.get(key)
        if isinstance(value, str):
            parts.append(value)
    for key in ("landmarks", "candidates"):
        for item in vision.get(key) or []:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("name") or ""))
                parts.append(str(item.get("why") or ""))

    text = " ".join(parts).lower()
    expected: list[str] = []
    for category, words in _CATEGORY_KEYWORDS.items():
        if any(w in text for w in words):
            expected.append(category)
    return expected


def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _classify(tags: dict[str, Any]) -> Optional[str]:
    natural = tags.get("natural")
    if natural in ("water", "bay", "wetland") or "waterway" in tags:
        return "water"
    if natural in ("coastline", "beach"):
        return "coast"
    if natural in ("peak", "volcano", "ridge", "glacier"):
        return "mountain"
    if natural == "wood" or tags.get("landuse") == "forest":
        return "forest"
    return None


def _skipped(note: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "note": note,
        "expected": [],
        "confirmed": [],
        "missing": [],
        "nearest_m": {},
        "match_score": 1.0,
    }


def verify_location(
    latitude: float, longitude: float, expected: list[str]
) -> dict[str, Any]:
    """Check whether the expected features exist near the coordinates."""
    if not _enabled():
        return _skipped("Verification disabled.")
    if latitude is None or longitude is None:
        return _skipped("No coordinates to verify.")
    if not expected:
        return _skipped("No checkable features described in the image.")

    query = (
        f"[out:json][timeout:{int(VERIFY_TIMEOUT)}];"
        f'(nwr["natural"~"^(water|bay|wetland|coastline|beach|peak|volcano|ridge|glacier|wood)$"]'
        f"(around:{VERIFY_RADIUS_M},{latitude},{longitude});"
        f'nwr["waterway"](around:{VERIFY_RADIUS_M},{latitude},{longitude});'
        f'nwr["landuse"="forest"](around:{VERIFY_RADIUS_M},{latitude},{longitude}););'
        f"out tags center 120;"
    )

    try:
        resp = requests.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": os.getenv("GEOCODER_USER_AGENT", "ImageLocatorBackend/1.0")},
            timeout=VERIFY_TIMEOUT + 5,
        )
        resp.raise_for_status()
        elements = resp.json().get("elements", [])
    except Exception as exc:
        return _skipped(f"Overpass lookup failed: {exc}")

    # Overpass' around filter guarantees each returned feature is within the
    # radius, so presence = "nearby". We only measure precise distance from
    # POINT features (nodes); polygon centroids badly overstate distance for
    # large features like lakes, so we never use them to judge closeness.
    present: set[str] = set()
    nearest_node: dict[str, float] = {}
    for el in elements:
        cat = _classify(el.get("tags", {}) or {})
        if not cat:
            continue
        present.add(cat)
        if el.get("type") == "node" and el.get("lat") is not None:
            dist = _haversine_m(latitude, longitude, float(el["lat"]), float(el["lon"]))
            if cat not in nearest_node or dist < nearest_node[cat]:
                nearest_node[cat] = round(dist, 1)

    # A "water" expectation is satisfied by a lake/river OR a coastline.
    def satisfied(cat: str) -> bool:
        if cat == "water":
            return "water" in present or "coast" in present
        return cat in present

    confirmed = [c for c in expected if satisfied(c)]
    missing = [c for c in expected if not satisfied(c)]
    match_score = len(confirmed) / len(expected) if expected else 1.0

    if not confirmed:
        status = "mismatch"
    elif missing:
        status = "partial"
    else:
        status = "verified"

    return {
        "status": status,
        "note": "ok",
        "expected": expected,
        "confirmed": confirmed,
        "missing": missing,
        "nearest_m": {k: v for k, v in nearest_node.items()},
        "match_score": round(match_score, 3),
        "radius_m": VERIFY_RADIUS_M,
    }
