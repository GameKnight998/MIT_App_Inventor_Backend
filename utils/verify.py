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
import time
from typing import Any, Optional

import requests

from utils.cache import cache_get, cache_set

# Public Overpass mirrors with worldwide data, tried in order so one mirror being
# busy or down doesn't disable verification. Left out on purpose:
# overpass.osm.ch holds Swiss data only, so it answers "nothing here" for the
# rest of the world and would cross out correct answers; overpass.openstreetmap.fr
# serves approved users only; maps.mail.ru often takes 20 s or more.
_DEFAULT_OVERPASS = (
    "https://overpass-api.de/api/interpreter,"
    "https://overpass.kumi.systems/api/interpreter,"
    "https://overpass.private.coffee/api/interpreter"
)
OVERPASS_ENDPOINTS = [
    u.strip() for u in os.getenv("OVERPASS_URL", _DEFAULT_OVERPASS).split(",") if u.strip()
]
VERIFY_RADIUS_M = int(os.getenv("VERIFY_RADIUS_M", "3000"))
VERIFY_TIMEOUT = float(os.getenv("VERIFY_TIMEOUT", "15"))
# Fail fast on a mirror we can't even connect to (separate from slow reads).
OVERPASS_CONNECT_TIMEOUT = float(os.getenv("OVERPASS_CONNECT_TIMEOUT", "6"))
# Hard wall-clock cap for trying mirrors on a SINGLE verify call, so a total
# outage can't stack up (mirrors x timeout) into a multi-minute request.
OVERPASS_TOTAL_BUDGET = float(os.getenv("OVERPASS_TOTAL_BUDGET", "25"))
# After a full outage, skip the network for this long so the remaining
# candidates in one request return "unavailable" instantly instead of each
# re-probing every dead mirror.
OVERPASS_COOLDOWN = float(os.getenv("OVERPASS_COOLDOWN", "120"))
# Mirrors rate-limit anonymous clients; a descriptive User-Agent is required.
OVERPASS_USER_AGENT = os.getenv(
    "OVERPASS_USER_AGENT", "Whereabouts-geolocation/1.0 (student research project)"
)
# Successful map answers are also kept on disk, so reruns (benchmarks, the same
# photo again) don't depend on the public mirrors at all. Empty disables it.
VERIFY_DISK_CACHE = os.getenv(
    "VERIFY_DISK_CACHE",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache", "overpass.json"),
)
# Terrain relief (highest minus lowest ground within the radius) that counts as
# mountains when the map service is down and elevation data is used instead.
MOUNTAIN_RELIEF_M = float(os.getenv("MOUNTAIN_RELIEF_M", "250"))
# Beyond this distance a "confirmed" feature is treated as only approximate.
PRECISION_M = int(os.getenv("VERIFY_PRECISION_M", "800"))

# Simple in-process circuit breaker: monotonic timestamp until which we treat
# Overpass as down and short-circuit without hitting the network.
_circuit_open_until = 0.0

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
        # A beach on a lake or river is not a sea coast: OpenStreetMap only
        # tags coastlines on the sea, so expecting one there would cross out
        # a correct inland answer (e.g. Lake Sammamish State Park).
        inland_beach = bool(
            scene.get("beach") and (scene.get("lake") or scene.get("river"))
            and not scene.get("ocean_or_sea") and not scene.get("coastline")
        )
        for key, category in _SCENE_TO_CATEGORY.items():
            if key == "beach" and inland_beach:
                continue
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


def _query_overpass(query: str) -> tuple[Optional[list[dict[str, Any]]], str]:
    """Try each Overpass mirror in turn. Returns (elements, note).

    Stops at the first mirror that answers. Busy mirrors (HTTP 429/504, reset
    connections, slow reads) are common, so after a full failed pass we wait
    briefly and try once more, all within OVERPASS_TOTAL_BUDGET. A short
    circuit-breaker cooldown then stops a real outage from stalling every
    candidate in the same request.
    """
    global _circuit_open_until

    now = time.monotonic()
    if now < _circuit_open_until:
        return None, "Overpass recently unreachable (cooling down, skipped network)"

    start = now
    last_error = "no endpoints configured"
    headers = {"User-Agent": OVERPASS_USER_AGENT}
    for attempt in range(2):
        if attempt:
            if time.monotonic() - start > OVERPASS_TOTAL_BUDGET - 8:
                break
            time.sleep(2.0)
        for url in OVERPASS_ENDPOINTS:
            remaining = OVERPASS_TOTAL_BUDGET - (time.monotonic() - start)
            if remaining <= 1:
                last_error = f"time budget {OVERPASS_TOTAL_BUDGET:.0f}s exceeded ({last_error})"
                break
            try:
                resp = requests.post(
                    url,
                    data={"data": query},
                    headers=headers,
                    timeout=(OVERPASS_CONNECT_TIMEOUT, min(VERIFY_TIMEOUT + 5, remaining)),
                )
                resp.raise_for_status()
                return resp.json().get("elements", []), f"ok via {url}"
            except Exception as exc:
                last_error = f"{url}: {exc}"
                continue

    # Every mirror failed (or we ran out of time): open the breaker so the rest
    # of this request doesn't re-probe the same dead endpoints.
    _circuit_open_until = time.monotonic() + OVERPASS_COOLDOWN
    return None, f"all Overpass mirrors failed ({last_error})"


# Overpass filters per category the photo can require. Only the categories the
# photo needs are asked for, plus a capped handful of context features, so
# a check near a city no longer pulls hundreds of buildings and land-use areas.
_CATEGORY_FILTERS: dict[str, list[str]] = {
    "water": ['nwr["natural"~"^(water|bay|wetland)$"]', 'nwr["waterway"]'],
    "coast": ['nwr["natural"~"^(coastline|beach|bay)$"]'],
    "mountain": ['nwr["natural"~"^(peak|volcano|ridge|glacier)$"]'],
    "forest": ['nwr["natural"="wood"]', 'nwr["landuse"="forest"]'],
}
_CONTEXT_FILTERS = [
    'nwr["railway"~"^(rail|light_rail|subway|tram|station)$"]',
    'nwr["aeroway"="aerodrome"]',
    'nwr["leisure"="park"]',
    'nwr["landuse"~"^(farmland|orchard|vineyard|meadow|residential|commercial|industrial|retail)$"]',
]


def _build_query(latitude: float, longitude: float, radius: int, expected: list[str]) -> str:
    around = f"(around:{radius},{latitude},{longitude});"
    needed = list(expected) + (["coast"] if "water" in expected else [])  # coastline counts as water
    parts = []
    for cat in dict.fromkeys(needed):
        parts += [f + around for f in _CATEGORY_FILTERS.get(cat, [])]
    ctx = "".join(f + around for f in _CONTEXT_FILTERS)
    return (
        f"[out:json][timeout:{int(VERIFY_TIMEOUT)}];"
        f"({''.join(parts)})->.need;.need out tags center 60;"
        f"({ctx})->.ctx;.ctx out tags center 15;"
    )


_disk: Optional[dict[str, Any]] = None


def _disk_key(key: tuple) -> str:
    return "|".join(str(k) for k in key)


def _disk_get(key: tuple) -> Optional[dict[str, Any]]:
    global _disk
    if not VERIFY_DISK_CACHE:
        return None
    if _disk is None:
        try:
            import json

            with open(VERIFY_DISK_CACHE, encoding="utf-8") as f:
                _disk = json.load(f)
        except Exception:
            _disk = {}
    return _disk.get(_disk_key(key))


def _disk_set(key: tuple, report: dict[str, Any]) -> None:
    global _disk
    if not VERIFY_DISK_CACHE:
        return
    try:
        import json

        if _disk is None:
            _disk_get(key)
        _disk[_disk_key(key)] = report
        os.makedirs(os.path.dirname(VERIFY_DISK_CACHE), exist_ok=True)
        tmp = VERIFY_DISK_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_disk, f)
        os.replace(tmp, VERIFY_DISK_CACHE)
    except Exception:
        pass  # the disk cache is an optimisation only


def _elevation_fallback(
    latitude: float, longitude: float, expected: list[str], radius: int, why: str
) -> Optional[dict[str, Any]]:
    """Check a mountain expectation from terrain data when Overpass is down.

    Samples ground elevation at the point and on a ring at the search radius;
    relief of MOUNTAIN_RELIEF_M or more counts as mountains. Other categories
    can't be judged this way and are left out of the score (not counted as
    missing). Returns None when nothing could be checked.
    """
    if "mountain" not in expected:
        return None
    from utils.geoenv import elevations

    dlat = radius / 111_000.0
    dlon = radius / (111_000.0 * max(0.2, math.cos(math.radians(latitude))))
    ring = [(latitude + dlat * math.sin(a), longitude + dlon * math.cos(a))
            for a in (i * math.pi / 4 for i in range(8))]
    vals = [v for v in elevations([(latitude, longitude)] + ring) if v is not None]
    if len(vals) < 5:
        return None
    relief = max(vals) - min(vals)
    mountains = relief >= MOUNTAIN_RELIEF_M
    report = _report(
        "verified" if mountains and expected == ["mountain"] else ("partial" if mountains else "mismatch"),
        f"Map service unreachable ({why}); terrain checked with elevation data instead: "
        f"{relief:.0f} m of relief within {radius / 1000:.1f} km.",
        expected,
    )
    report["confirmed"] = ["mountain"] if mountains else []
    report["missing"] = [] if mountains else ["mountain"]
    # Categories elevation can't judge are unchecked, not missing.
    report["unchecked"] = [c for c in expected if c != "mountain"]
    report["match_score"] = 1.0 if mountains else 0.0
    report["radius_m"] = radius
    report["source"] = "elevation"
    if mountains and report["unchecked"]:
        report["status"] = "partial"
    return report


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

    disk_hit = _disk_get(cache_key)
    if disk_hit is not None:
        cache_set(cache_key, disk_hit)
        return disk_hit

    query = _build_query(latitude, longitude, r, expected)

    elements, note = _query_overpass(query)
    if elements is None:
        # Distinct from "skipped": we DID want to verify but couldn't reach the
        # map service. Use elevation data for what it can judge (mountains);
        # otherwise the result is simply unverified.
        fallback = _elevation_fallback(latitude, longitude, expected, r, note)
        return fallback if fallback is not None else _unavailable(note, expected)

    # Overpass' around filter guarantees each returned feature is within the
    # radius, so presence = "nearby". We only measure precise distance from
    # POINT features (nodes); polygon centroids badly overstate distance for
    # large features like lakes, so we never use them to judge closeness.
    present: set[str] = set()
    nearest_node: dict[str, float] = {}
    # Nearest anchor point per category (node coords, else the way/relation
    # "center"), used by the narrowing pass to re-centre on the real feature.
    anchors: dict[str, dict[str, float]] = {}
    for el in elements:
        cat = _classify(el.get("tags", {}) or {})
        if not cat:
            continue
        present.add(cat)
        if el.get("type") == "node" and el.get("lat") is not None:
            plat, plon = float(el["lat"]), float(el["lon"])
            dist = _haversine_m(latitude, longitude, plat, plon)
            if cat not in nearest_node or dist < nearest_node[cat]:
                nearest_node[cat] = round(dist, 1)
        else:
            center = el.get("center") or {}
            plat = center.get("lat")
            plon = center.get("lon")
        if plat is None or plon is None:
            continue
        adist = _haversine_m(latitude, longitude, float(plat), float(plon))
        if cat not in anchors or adist < anchors[cat]["dist_m"]:
            anchors[cat] = {
                "lat": round(float(plat), 6),
                "lon": round(float(plon), 6),
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

    if not confirmed:
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
    _disk_set(cache_key, report)
    return report
