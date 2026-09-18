"""Turn a regional guess into an exact spot ("Find Street" equivalent).

Regional estimation answers "which part of the world"; this answers "which
corner". The two need different methods, which is why it is a separate pass
rather than a better first prompt:

  1. A regional guess is already in hand, so the vision model is re-run with that
     anchor supplied. Freed from re-deriving the country, it can spend its whole
     attention on text and micro-detail (see `vision.analyze_street_level`).
  2. Whatever it reads is then GROUNDED against the map. A model claiming a photo
     shows "Columbus Avenue" proves nothing; a Columbus Avenue existing 1.2 km
     from the regional guess is evidence. Any candidate that does not resolve
     inside the search radius is discarded rather than trusted.

Grounding is what keeps this honest. The refined point can only ever be a real
OSM feature near the region we already believed in, so a hallucinated street name
degrades to "no refinement" instead of a confidently wrong address.

Cost control: the pass is skipped entirely for scenes that cannot support it (an
empty alpine valley has no street names to read), so wilderness images pay
nothing for the capability.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

from utils.cache import parallel_map
from utils.geocode import geocode_street, is_street_level
from utils.vision import analyze_street_level, empty_street_analysis

STREET_REFINE_ENABLED = os.getenv("STREET_REFINE_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
# How far from the regional guess a refined point may sit. Wider than a city so
# a slightly-off region still refines, tight enough to reject another continent.
STREET_REFINE_RADIUS_KM = float(os.getenv("STREET_REFINE_RADIUS_KM", "25"))
# Below this refinement confidence we do not move the point.
STREET_REFINE_MIN_CONFIDENCE = float(os.getenv("STREET_REFINE_MIN_CONFIDENCE", "0.2"))
# Cap the geocode fan-out so one image cannot hammer the public geocoders.
STREET_REFINE_MAX_QUERIES = int(os.getenv("STREET_REFINE_MAX_QUERIES", "4"))

# Precision we claim per kind of match, in metres. A house number is a doorway;
# a bare street name is only the right few blocks.
_PRECISION_BY_METHOD = {
    "address": 60,
    "business": 120,
    "transit": 200,
    "intersection": 150,
    "street": 400,
    "place": 800,
    "model_estimate": 1500,
}
# Query priority: most uniquely-identifying clue first.
_METHOD_PRIORITY = (
    "address",
    "business",
    "intersection",
    "transit",
    "street",
    "place",
)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _empty_result(note: str, attempted: bool = False) -> dict[str, Any]:
    return {
        "attempted": attempted,
        "refined": False,
        "note": note,
        "latitude": None,
        "longitude": None,
        "address": None,
        "precision_m": None,
        "method": None,
        "matched_query": None,
        "distance_moved_km": None,
        "street_clues": {},
        "candidates": [],
        "evidence": [],
    }


def worth_refining(vision: Optional[dict[str, Any]]) -> bool:
    """Whether this scene could plausibly contain street-level identifiers.

    Skipping hopeless scenes is what keeps the second pass affordable: a forest
    has no shopfronts to read, so spending a vision call on it is pure cost.
    """
    if not vision:
        return False
    scene = vision.get("scene") or {}
    if scene.get("urban") or scene.get("suburban"):
        return True
    if vision.get("scene_type") in ("urban", "suburban", "indoor", "vehicle_interior"):
        return True
    # Any readable text or a named landmark means there is something to ground.
    for key in ("ocr_text", "signage", "landmarks", "vehicles_plates"):
        if vision.get(key):
            return True
    return False


def _clue_queries(
    street: dict[str, Any], city_hint: Optional[str]
) -> list[tuple[str, str]]:
    """Build (method, query) pairs from the street-level clues, best first."""
    streets = street.get("street_names") or []
    numbers = street.get("house_numbers") or []
    businesses = street.get("business_names") or []
    stops = street.get("transit_stops") or []
    suffix = f", {city_hint}" if city_hint else ""

    queries: list[tuple[str, str]] = []

    # A house number only means something attached to its street.
    if numbers and streets:
        queries.append(("address", f"{numbers[0]} {streets[0]}{suffix}"))

    for name in businesses[:2]:
        # Pair the business with a street when known: far less ambiguous for chains.
        locality = f", {streets[0]}" if streets else suffix
        queries.append(("business", f"{name}{locality}"))

    if street.get("intersection"):
        queries.append(("intersection", f"{street['intersection']}{suffix}"))

    for stop in stops[:1]:
        queries.append(("transit", f"{stop}{suffix}"))

    for name in streets[:2]:
        queries.append(("street", f"{name}{suffix}"))

    if street.get("refined_place"):
        queries.append(("place", f"{street['refined_place']}{suffix}"))

    # De-duplicate while keeping priority order.
    seen: set[str] = set()
    unique: list[tuple[str, str]] = []
    for method, query in queries:
        key = query.strip().lower()
        if key and key not in seen:
            seen.add(key)
            unique.append((method, query.strip()))
    unique.sort(key=lambda mq: _METHOD_PRIORITY.index(mq[0]))
    return unique[:STREET_REFINE_MAX_QUERIES]


def _precision_for(method: str, hit: dict[str, Any]) -> int:
    base = _PRECISION_BY_METHOD.get(method, 1000)
    if hit.get("housenumber"):
        base = min(base, _PRECISION_BY_METHOD["address"])
    elif not is_street_level(hit):
        # Resolved to something broad (a city centroid); do not claim precision.
        base = max(base, 1500)
    return int(base)


def refine_street_level(
    latitude: float,
    longitude: float,
    vision: Optional[dict[str, Any]],
    *,
    image_path: Optional[str],
    place_label: Optional[str] = None,
    city_hint: Optional[str] = None,
    radius_km: Optional[float] = None,
) -> dict[str, Any]:
    """Refine a regional guess to a street/address, or report why it could not.

    Never raises and never returns a point outside `radius_km` of the input, so
    a caller can apply the result unconditionally.
    """
    if not STREET_REFINE_ENABLED:
        return _empty_result("Street-level refinement disabled.")
    if latitude is None or longitude is None or not image_path:
        return _empty_result("No anchor coordinates for refinement.")
    if not os.path.exists(image_path):
        return _empty_result("Source image no longer available for refinement.")
    if not worth_refining(vision):
        return _empty_result(
            "Scene has no street-level identifiers to refine (no text, signage "
            "or built environment)."
        )

    radius = float(radius_km or STREET_REFINE_RADIUS_KM)

    try:
        street = analyze_street_level(
            image_path,
            latitude=latitude,
            longitude=longitude,
            radius_km=radius,
            place_label=place_label,
            scene_type=(vision or {}).get("scene_type"),
        )
    except Exception as exc:
        return _empty_result(f"Street-level vision pass failed: {exc}", attempted=True)

    result = _empty_result("", attempted=True)
    result["street_clues"] = {
        key: street.get(key)
        for key in (
            "street_names",
            "house_numbers",
            "business_names",
            "transit_stops",
            "postal_codes",
            "intersection",
            "architectural_details",
            "street_furniture",
            "reasoning",
            "confidence",
        )
    }

    if not street.get("available"):
        result["note"] = street.get("note") or "Street-level pass unavailable."
        return result

    evidence: list[str] = []
    for label, key in (
        ("Street name(s) read", "street_names"),
        ("Building number(s) read", "house_numbers"),
        ("Business name(s) read", "business_names"),
        ("Transit stop(s) read", "transit_stops"),
        ("Postal code(s) read", "postal_codes"),
    ):
        if street.get(key):
            evidence.append(f"{label}: " + ", ".join(street[key][:3]) + ".")
    if street.get("architectural_details"):
        evidence.append(
            "Architectural detail: " + ", ".join(street["architectural_details"][:3]) + "."
        )
    result["evidence"] = evidence

    queries = _clue_queries(street, city_hint)
    if not queries:
        result["note"] = (
            "No street names, building numbers or business names were legible, "
            "so the regional estimate stands."
        )
        return result

    # Independent lookups; run them together since each is network-bound.
    hits = parallel_map(
        lambda mq: (mq[0], mq[1], geocode_street(mq[1], latitude, longitude, radius)),
        queries,
    )

    scored: list[dict[str, Any]] = []
    for method, query, hit in hits:
        if not hit:
            continue
        scored.append(
            {
                "method": method,
                "query": query,
                "name": hit.get("display_name") or hit.get("name"),
                "latitude": hit["latitude"],
                "longitude": hit["longitude"],
                "distance_km": hit.get("distance_km"),
                "provider": hit.get("provider"),
                "street_level": is_street_level(hit),
                "_hit": hit,
            }
        )

    result["candidates"] = [
        {k: v for k, v in s.items() if not k.startswith("_")} for s in scored
    ]

    if scored:
        scored.sort(
            key=lambda s: (
                _METHOD_PRIORITY.index(s["method"]),
                0 if s["street_level"] else 1,
                s["distance_km"] if s["distance_km"] is not None else 1e9,
            )
        )
        best = scored[0]
        hit = best["_hit"]
        moved = _haversine_km(latitude, longitude, hit["latitude"], hit["longitude"])
        precision = _precision_for(best["method"], hit)
        if street.get("precision_estimate_m"):
            # Trust whichever source is more conservative.
            precision = int(max(precision, min(street["precision_estimate_m"], 5000)))

        result.update(
            {
                "refined": True,
                "latitude": hit["latitude"],
                "longitude": hit["longitude"],
                "address": hit.get("display_name") or hit.get("name"),
                "precision_m": precision,
                "method": f"{best['method']}_geocode",
                "matched_query": best["query"],
                "distance_moved_km": round(moved, 2),
                "note": (
                    f"Refined to {hit.get('display_name') or hit.get('name')} by "
                    f"matching '{best['query']}' on the map "
                    f"({moved:.1f} km from the regional estimate)."
                ),
            }
        )
        result["evidence"].append(result["note"])
        return result

    # Nothing grounded. The model's own coordinates are a weak last resort, and
    # only inside the radius so they cannot override the verified region.
    mlat = street.get("refined_latitude")
    mlon = street.get("refined_longitude")
    if (
        mlat is not None
        and mlon is not None
        and street.get("confidence", 0.0) >= STREET_REFINE_MIN_CONFIDENCE
        and _haversine_km(latitude, longitude, mlat, mlon) <= radius
    ):
        moved = _haversine_km(latitude, longitude, mlat, mlon)
        result.update(
            {
                "refined": True,
                "latitude": round(float(mlat), 6),
                "longitude": round(float(mlon), 6),
                "precision_m": int(
                    max(
                        _PRECISION_BY_METHOD["model_estimate"],
                        min(street.get("precision_estimate_m") or 1500, 5000),
                    )
                ),
                "method": "model_estimate",
                "distance_moved_km": round(moved, 2),
                "note": (
                    "Refined using the model's own estimate; the named clues "
                    "could not be confirmed on the map, so precision is coarse."
                ),
            }
        )
        result["evidence"].append(result["note"])
        return result

    result["note"] = (
        "Street-level clues were read but none could be matched on the map "
        "nearby, so the regional estimate stands."
    )
    return result
