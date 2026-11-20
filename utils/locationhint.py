"""User-supplied location hints ("seattle", "arizona desert").

When the user already knows roughly where a photo was taken, searching the
whole world wastes the vision model's attention and lets a confident wrong
continent win. A hint is resolved once to a centre and search radius, then:

  1. the vision prompt is told to keep its candidates inside that area,
  2. candidates outside it are demoted below the ones inside it, and
  3. the response reports whether the final pin actually sits inside it.

The hint is a strong prior, not a hard constraint: if every image clue points
elsewhere the answer stays in-area (the hint centre as a fallback) but the
response warns that the photo seems to contradict the hint.
"""

from __future__ import annotations

import math
import re
import unicodedata
from typing import Any, Optional

from utils.geocode import _nominatim_search, forward_geocode_candidates

MAX_HINT_CHARS = 120

# Words that describe the kind of place rather than name it. Stripped before
# geocoding ("arizona desert" -> "arizona") but kept for the vision prompt.
_DESCRIPTOR_WORDS = {
    "desert", "deserts", "mountain", "mountains", "forest", "forests", "coast",
    "coastal", "beach", "beaches", "area", "region", "downtown", "uptown",
    "outskirts", "suburbs", "countryside", "rural", "city", "town", "near",
    "around", "outside", "in", "the", "of", "somewhere", "probably", "maybe",
    "north", "south", "east", "west", "northern", "southern", "eastern",
    "western", "central", "lake", "lakes", "hills", "valley", "island",
    "islands", "park", "national",
}

# Search radius by resolved place type, km. Used when no bounding box exists.
_RADIUS_BY_TYPE = {
    "country": 800.0,
    "state": 300.0,
    "province": 300.0,
    "region": 300.0,
    "county": 60.0,
    # People name a town to mean its surroundings ("Banff" for Lake Louise).
    "city": 40.0,
    "town": 30.0,
    "village": 20.0,
    "hamlet": 15.0,
    "suburb": 10.0,
    "neighbourhood": 5.0,
    "peak": 15.0,
    "mountain_range": 80.0,
    "desert": 200.0,
    "lake": 25.0,
    "park": 40.0,
    "national_park": 60.0,
    "island": 40.0,
}
_DEFAULT_RADIUS_KM = 50.0
_MIN_RADIUS_KM = 3.0
_MAX_RADIUS_KM = 1500.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def clean_hint(raw: Any) -> Optional[str]:
    """Trimmed hint text, or None when the field is empty/unusable."""
    if raw is None:
        return None
    text = re.sub(r"\s+", " ", str(raw)).strip()
    if not text or text.lower() in ("none", "null", "unknown", "anywhere"):
        return None
    return text[:MAX_HINT_CHARS]


def _place_query(text: str) -> Optional[str]:
    words = re.findall(r"[\w'.-]+", text)
    kept = [w for w in words if w.lower() not in _DESCRIPTOR_WORDS]
    query = " ".join(kept).strip()
    return query if query and query.lower() != text.lower() else None


_ADMIN_TYPES = {"country", "state", "province", "region"}


def _place_type(hit: dict[str, Any]) -> str:
    # Nominatim's `type` is "administrative" for states; `addresstype` is the useful one.
    kind = str(hit.get("addresstype") or hit.get("osm_type") or "").lower()
    if kind in ("natural", "place", ""):
        kind = str(hit.get("osm_type") or kind).lower()
    return kind


def _radius_for(hit: dict[str, Any]) -> float:
    kind = _place_type(hit)
    radius = _RADIUS_BY_TYPE.get(kind, _DEFAULT_RADIUS_KM)
    population = hit.get("population")
    if isinstance(population, (int, float)) and kind in ("city", "town"):
        if population > 2_000_000:
            radius = max(radius, 60.0)
    # A bounding box can only widen the area: deserts and ranges are often a
    # single labelled node with a near-zero box.
    bbox = hit.get("bbox")
    if bbox:
        south, north, west, east = bbox
        radius = max(radius, _haversine_km(south, west, north, east) / 2.0)
    return min(max(radius, _MIN_RADIUS_KM), _MAX_RADIUS_KM)


def _norm(text: Any) -> str:
    folded = unicodedata.normalize("NFKD", str(text or ""))
    return re.sub(r"[^a-z0-9 ]", "", folded.encode("ascii", "ignore").decode().lower()).strip()


def _geocode_hint(query: str) -> Optional[dict[str, Any]]:
    """Best real place for a free-text area name.

    Nominatim understands whole phrases ("arizona desert", "Portland, Maine")
    and returns real extents. Open-Meteo only matches a head name and fuzzily
    ("Sahara" -> Saharanpur), so its hits must match the typed name exactly.
    """
    hits = _nominatim_search(
        {"q": query, "format": "jsonv2", "limit": 1, "addressdetails": 1}
    )
    if hits:
        return hits[0]
    head = _norm(query.split(",")[0])
    for hit in forward_geocode_candidates(query, limit=5):
        if _norm(hit.get("name")) == head:
            return hit
    return None


def resolve_location_hint(raw: Any) -> Optional[dict[str, Any]]:
    """Turn free text into {text, name, latitude, longitude, radius_km, ...}.

    Returns a hint with `resolved=False` (text only) when the place cannot be
    geocoded, so the vision prompt can still use the words.
    """
    text = clean_hint(raw)
    if text is None:
        return None

    hint: dict[str, Any] = {
        "text": text,
        "resolved": False,
        "name": None,
        "country": None,
        "latitude": None,
        "longitude": None,
        "radius_km": None,
    }

    stripped = _place_query(text)
    hit = None
    if stripped:
        # "arizona desert": if the named part is a whole state/country, that is
        # the user's area; the descriptor only shapes the vision prompt.
        named = _geocode_hint(stripped)
        if named and _place_type(named) in _ADMIN_TYPES:
            hit = named
    if hit is None:
        hit = _geocode_hint(text)
    if hit is None and stripped:
        hit = _geocode_hint(stripped)
    if hit is None:
        return hint

    hint.update(
        {
            "resolved": True,
            "name": hit.get("display_name") or hit.get("name"),
            "country": hit.get("country"),
            "latitude": hit["latitude"],
            "longitude": hit["longitude"],
            "radius_km": round(_radius_for(hit), 1),
            "place_type": _place_type(hit) or None,
        }
    )
    return hint


def prompt_text(hint: Optional[dict[str, Any]]) -> Optional[str]:
    """Instruction appended to the vision request for this hint."""
    if not hint:
        return None
    lines = [
        f"USER LOCATION HINT: the person who submitted this photo says it was "
        f"taken in or near \"{hint['text']}\"."
    ]
    if hint.get("resolved"):
        lines.append(
            f"That resolves to {hint['name']} (centre {hint['latitude']:.4f}, "
            f"{hint['longitude']:.4f}, roughly {hint['radius_km']:.0f} km radius)."
        )
    lines.append(
        "Treat this as a strong prior: do not re-derive the country or region. "
        "Spend your effort pinning the exact camera position INSIDE this area, "
        "and place every candidate inside it. Only if the image clearly "
        "contradicts the hint (e.g. visible text, plates or landmarks from "
        "elsewhere), say so in `reasoning` and add the out-of-area candidate "
        "last."
    )
    return " ".join(lines)


def _inside(hint: dict[str, Any], lat: Any, lon: Any, slack: float = 1.25) -> Optional[bool]:
    if not hint.get("resolved") or lat is None or lon is None:
        return None
    try:
        dist = _haversine_km(hint["latitude"], hint["longitude"], float(lat), float(lon))
    except (TypeError, ValueError):
        return None
    return dist <= float(hint["radius_km"]) * slack


def constrain_candidates(
    candidates: list[dict[str, Any]], hint: Optional[dict[str, Any]]
) -> tuple[list[dict[str, Any]], bool]:
    """Order candidates in-area first; returns (candidates, any_inside).

    Out-of-area candidates keep their relative order but drop below every
    in-area one with reduced confidence. When none are inside, the hint centre
    itself is added at the front so the answer still stays in the stated area.
    """
    if not hint or not hint.get("resolved"):
        return list(candidates), True

    inside: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    for cand in candidates:
        flag = _inside(hint, cand.get("latitude"), cand.get("longitude"))
        if flag is False:
            demoted = dict(cand)
            demoted["confidence"] = round(float(cand.get("confidence") or 0.0) * 0.3, 3)
            demoted["outside_hint"] = True
            outside.append(demoted)
        else:
            inside.append(cand)

    if inside:
        return inside + outside, True

    centre = {
        "name": hint["name"] or hint["text"],
        "region": None,
        "country": hint.get("country"),
        "latitude": hint["latitude"],
        "longitude": hint["longitude"],
        "confidence": 0.2,
        "why": "Centre of the user's location hint; no image-based candidate fell inside it.",
        "from_hint": True,
    }
    return [centre] + outside, False


def hint_report(
    hint: Optional[dict[str, Any]],
    latitude: Any,
    longitude: Any,
    any_candidate_inside: bool = True,
) -> Optional[dict[str, Any]]:
    """Client-facing summary of the hint and whether the pin respects it."""
    if not hint:
        return None
    report = {
        "text": hint["text"],
        "resolved": hint.get("resolved", False),
        "resolved_name": hint.get("name"),
        "latitude": hint.get("latitude"),
        "longitude": hint.get("longitude"),
        "radius_km": hint.get("radius_km"),
        "result_inside": None,
        "distance_km": None,
        "image_agrees": any_candidate_inside,
    }
    if hint.get("resolved") and latitude is not None and longitude is not None:
        dist = _haversine_km(hint["latitude"], hint["longitude"], float(latitude), float(longitude))
        report["distance_km"] = round(dist, 1)
        report["result_inside"] = dist <= float(hint["radius_km"]) * 1.25
    return report
