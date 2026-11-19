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
import threading
import time
from typing import Any, Optional

import requests

from utils.cache import cache_get, cache_set

# Prefer mirrors that answer from typical residential networks.
# osm.ch is fast but has been observed returning empty around: results for
# well-mapped alpine peaks; put the fuller French mirror first.
_DEFAULT_OVERPASS = (
    "https://overpass.openstreetmap.fr/api/interpreter,"
    "https://overpass.kumi.systems/api/interpreter,"
    "https://overpass.private.coffee/api/interpreter,"
    "https://overpass-api.de/api/interpreter,"
    "https://overpass.osm.ch/api/interpreter,"
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
)
_PREFERRED_OVERPASS = "https://overpass.openstreetmap.fr/api/interpreter"


def _load_overpass_endpoints() -> list[str]:
    raw = [
        u.strip()
        for u in os.getenv("OVERPASS_URL", _DEFAULT_OVERPASS).split(",")
        if u.strip()
    ]
    if _PREFERRED_OVERPASS in raw:
        return [_PREFERRED_OVERPASS] + [u for u in raw if u != _PREFERRED_OVERPASS]
    return raw


OVERPASS_ENDPOINTS = _load_overpass_endpoints()
VERIFY_RADIUS_M = int(os.getenv("VERIFY_RADIUS_M", "3000"))
VERIFY_TIMEOUT = float(os.getenv("VERIFY_TIMEOUT", "15"))
# Fail fast on a mirror we can't even connect to (separate from slow reads).
OVERPASS_CONNECT_TIMEOUT = float(os.getenv("OVERPASS_CONNECT_TIMEOUT", "4"))
OVERPASS_TOTAL_BUDGET = float(os.getenv("OVERPASS_TOTAL_BUDGET", "25"))
OVERPASS_COOLDOWN = float(os.getenv("OVERPASS_COOLDOWN", "30"))
OVERPASS_BREAKER_FAILS = int(os.getenv("OVERPASS_BREAKER_FAILS", "3"))
PRECISION_M = int(os.getenv("VERIFY_PRECISION_M", "800"))

_circuit_open_until = 0.0
_consecutive_failures = 0
_overpass_lock = threading.Lock()

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


def is_mismatch(report: dict[str, Any]) -> bool:
    """True when NONE of the expected features exist near the coordinates.

    Used by the OSINT retry loop to decide whether to cross out a location and
    try the next candidate.
    """
    return report.get("status") == "mismatch"


def is_confirmed_match(report: dict[str, Any]) -> bool:
    """True when the location is corroborated (or verification was skipped)."""
    return report.get("status") in ("verified", "skipped")


# Structured scene booleans (from vision) -> checkable natural feature category.
# Only well-tagged, discriminative natural features are used as *expected*
# (required) features. Poorly tagged categories like desert are deliberately
# excluded so a correct location is never crossed out for a mapping gap.
_SCENE_TO_CATEGORY: dict[str, str] = {
    "lake": "water",
    "river": "water",
    "ocean_or_sea": "coast",
    "beach": "coast",
    "coastline": "coast",
    "mountains": "mountain",
    "forest": "forest",
}


def detect_expected_features(vision: Optional[dict[str, Any]]) -> list[str]:
    """Infer which geographic features the image should be near.

    Prefers the structured `scene` booleans emitted by the vision model; only
    falls back to scanning free text when no structured scene is available.
    """
    if not vision:
        return []

    scene = vision.get("scene")
    if isinstance(scene, dict) and any(bool(v) for v in scene.values()):
        expected: list[str] = []
        for key, category in _SCENE_TO_CATEGORY.items():
            if scene.get(key) and category not in expected:
                expected.append(category)
        return expected

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
    expected = []
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


# Categories that are reported as nearby *context* (they enrich the reasoning
# trace and street-network corroboration) but do not affect the pass/fail score,
# because features like roads/farmland/urban land exist almost everywhere.
_CONTEXT_CATEGORIES = ("railway", "airport", "park", "farmland", "urban")


def _classify(tags: dict[str, Any]) -> Optional[str]:
    natural = tags.get("natural")
    landuse = tags.get("landuse")
    if natural in ("water", "bay", "wetland") or "waterway" in tags:
        return "water"
    if natural in ("coastline", "beach"):
        return "coast"
    if natural in ("peak", "volcano", "ridge", "glacier"):
        return "mountain"
    if natural == "wood" or landuse == "forest":
        return "forest"
    # --- richer street-network / land-use context (#5) ---
    if tags.get("railway") in ("rail", "light_rail", "subway", "tram", "station"):
        return "railway"
    if tags.get("aeroway") == "aerodrome":
        return "airport"
    if tags.get("leisure") == "park":
        return "park"
    if landuse in ("farmland", "orchard", "vineyard", "meadow"):
        return "farmland"
    if landuse in ("residential", "commercial", "industrial", "retail"):
        return "urban"
    return None


def _report(status: str, note: str, expected: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "note": note,
        "expected": expected or [],
        "confirmed": [],
        "missing": [],
        "nearest_m": {},
        "match_score": 1.0,
    }


def _skipped(note: str) -> dict[str, Any]:
    return _report("skipped", note)


def _unavailable(note: str, expected: list[str]) -> dict[str, Any]:
    """Verification could not be performed (map service unreachable)."""
    return _report("unavailable", note, expected)


def _query_overpass(
    query: str, *, reject_empty: bool = False
) -> tuple[Optional[list[dict[str, Any]]], str]:
    """Try each Overpass mirror in turn. Returns (elements, note).

    Serialized so parallel prefetch cannot stampede a mirror and trip the
    breaker for the rest of the request. When `reject_empty` is set, an HTTP
    200 with zero elements is treated as a soft miss and the next mirror is
    tried — some mirrors silently return empty around: results for regions
    they have not fully imported.
    """
    global _circuit_open_until, _consecutive_failures

    with _overpass_lock:
        now = time.monotonic()
        if now < _circuit_open_until:
            return None, "Overpass recently unreachable (cooling down, skipped network)"

        start = now
        last_error = "no endpoints configured"
        headers = {
            "User-Agent": os.getenv(
                "GEOCODER_USER_AGENT",
                "ImageLocatorBackend/1.0 (MIT App Inventor capstone)",
            )
        }
        empty_ok: Optional[list[dict[str, Any]]] = None
        empty_note = ""
        for url in OVERPASS_ENDPOINTS:
            if time.monotonic() - start > OVERPASS_TOTAL_BUDGET:
                last_error = (
                    f"time budget {OVERPASS_TOTAL_BUDGET:.0f}s exceeded ({last_error})"
                )
                break
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers=headers,
                    timeout=(OVERPASS_CONNECT_TIMEOUT, VERIFY_TIMEOUT + 5),
                )
                resp.raise_for_status()
                elements = resp.json().get("elements", []) or []
                if reject_empty and not elements:
                    last_error = f"{url}: empty result"
                    empty_ok = elements
                    empty_note = f"ok via {url} (empty)"
                    continue
                _consecutive_failures = 0
                return elements, f"ok via {url}"
            except Exception as exc:
                last_error = f"{url}: {exc}"
                continue

        if empty_ok is not None:
            # Every mirror that answered returned nothing; accept empty.
            _consecutive_failures = 0
            return empty_ok, empty_note or last_error

        _consecutive_failures += 1
        if _consecutive_failures >= OVERPASS_BREAKER_FAILS:
            _circuit_open_until = time.monotonic() + OVERPASS_COOLDOWN
            _consecutive_failures = 0
        return None, f"all Overpass mirrors failed ({last_error})"


def verify_location(
    latitude: float, longitude: float, expected: list[str], radius: Optional[int] = None
) -> dict[str, Any]:
    """Check whether the expected features exist near the coordinates.

    `radius` (metres) overrides the default search radius; the iterative
    narrowing pass shrinks it to localise the scene more tightly.
    """
    if not _enabled():
        return _skipped("Verification disabled.")
    if latitude is None or longitude is None:
        return _skipped("No coordinates to verify.")
    if not expected:
        return _skipped("No checkable features described in the image.")

    r = int(radius) if radius else VERIFY_RADIUS_M
    # Cache real map answers by rounded coords/features/radius. Never cache an
    # "unavailable" (outage) result, so a hiccup isn't remembered.
    cache_key = (
        "verify",
        round(latitude, 3),
        round(longitude, 3),
        tuple(sorted(expected)),
        r,
    )
    cached_report = cache_get(cache_key)
    if cached_report is not None:
        return cached_report

    query = (
        f"[out:json][timeout:{int(VERIFY_TIMEOUT)}];"
        f'(nwr["natural"~"^(water|bay|wetland|coastline|beach|peak|volcano|ridge|glacier|wood)$"]'
        f"(around:{r},{latitude},{longitude});"
        f'nwr["waterway"](around:{r},{latitude},{longitude});'
        f'nwr["landuse"~"^(forest|farmland|orchard|vineyard|meadow|residential|commercial|industrial|retail)$"]'
        f"(around:{r},{latitude},{longitude});"
        f'nwr["railway"~"^(rail|light_rail|subway|tram|station)$"](around:{r},{latitude},{longitude});'
        f'nwr["aeroway"="aerodrome"](around:{r},{latitude},{longitude});'
        f'nwr["leisure"="park"](around:{r},{latitude},{longitude}););'
        f"out tags center 200;"
    )

    elements, note = _query_overpass(query, reject_empty=True)
    if elements is None:
        # Distinct from "skipped": we DID want to verify but couldn't reach the
        # map service, so the caller knows the result is simply unverified.
        return _unavailable(note, expected)

    # Overpass' around filter guarantees each returned feature is within the
    # radius, so presence = "nearby". We only measure precise distance from
    # POINT features (nodes); polygon centroids badly overstate distance for
    # large features like lakes, so we never use them to judge closeness.
    present: set[str] = set()
    nearest_node: dict[str, float] = {}
    # Nearest anchor point per category (node coords, else the way/relation
    # "center"), used by the narrowing pass to re-centre on the real feature.
    anchors: dict[str, dict[str, float]] = {}
    # around:radius should keep every hit nearby; anything farther is a bad
    # mirror payload (or a lat/lon leak across elements) and must be ignored.
    max_anchor_m = float(r) * 1.5
    for el in elements:
        cat = _classify(el.get("tags", {}) or {})
        if not cat:
            continue
        plat: Optional[float] = None
        plon: Optional[float] = None
        if el.get("type") == "node" and el.get("lat") is not None:
            plat, plon = float(el["lat"]), float(el["lon"])
        else:
            center = el.get("center") or {}
            if center.get("lat") is not None and center.get("lon") is not None:
                plat, plon = float(center["lat"]), float(center["lon"])
        if plat is None or plon is None:
            # Tag presence without a usable point still counts as "nearby"
            # because Overpass around: already filtered the element.
            present.add(cat)
            continue
        adist = _haversine_m(latitude, longitude, plat, plon)
        if adist > max_anchor_m:
            continue
        present.add(cat)
        if el.get("type") == "node":
            if cat not in nearest_node or adist < nearest_node[cat]:
                nearest_node[cat] = round(adist, 1)
        if cat not in anchors or adist < anchors[cat]["dist_m"]:
            anchors[cat] = {
                "lat": round(plat, 6),
                "lon": round(plon, 6),
                "dist_m": round(adist, 1),
            }

    # A "water" expectation is satisfied by a lake/river OR a coastline.
    def satisfied(cat: str) -> bool:
        if cat == "water":
            return "water" in present or "coast" in present
        return cat in present

    confirmed = [c for c in expected if satisfied(c)]
    missing = [c for c in expected if not satisfied(c)]
    match_score = len(confirmed) / len(expected) if expected else 1.0

    # Nearby land-use / street-network features that weren't required but add
    # corroborating context (roads, railways, farmland, urban land, etc.).
    context = [c for c in _CONTEXT_CATEGORIES if c in present]

    # Soft: empty OSM answer is weak (sparse map data), not a hard contradict.
    if not confirmed and not present:
        status = "partial"
    elif not confirmed:
        status = "mismatch"
    elif missing:
        status = "partial"
    else:
        status = "verified"

    report = {
        "status": status,
        "note": "ok",
        "expected": expected,
        "confirmed": confirmed,
        "missing": missing,
        "context": context,
        "nearest_m": {k: v for k, v in nearest_node.items()},
        "anchors": anchors,
        "match_score": round(match_score, 3),
        "radius_m": r,
    }
    cache_set(cache_key, report)
    return report
