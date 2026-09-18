"""Ask an AI vision model for the location-relevant clues in an image.

The model acts like a professional geolocation (OSINT) analyst. It reads visible
text (OCR), identifies landmarks, flags, signage, license plates, languages,
architecture, vegetation, climate, terrain, driving side and sun/shadow hints,
then proposes a RANKED list of candidate locations with confidences.

Two refinements over a single generic prompt:

  * Scene routing (`SCENE_ROUTING_ENABLED`) spends one cheap low-detail call
    asking only "what kind of scene is this?", then appends instructions written
    for that scene type. A street corner and an alpine lake reward completely
    different attention, and a generic prompt splits it evenly.
  * `analyze_street_level` is a second, anchored pass used once a region is
    known. Telling the model where it already is frees it to spend its attention
    on micro-detail (street names, house numbers, shopfronts) instead of
    re-deriving the country.

The actual network call lives in `utils.visionclient`, so this module only owns
prompts and the normalisation of whatever the model returns. If no provider is
configured everything degrades to an empty analysis and the rest of the pipeline
still runs (e.g. EXIF-only locating).
"""

from __future__ import annotations

import os
from typing import Any, Optional

from utils.visionclient import call_vision_json

SCENE_ROUTING_ENABLED = os.getenv("SCENE_ROUTING_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)

# Boolean scene descriptors. Emitted as a structured object so downstream
# verification never has to grep free-text wording (which was brittle).
_SCENE_KEYS = (
    "mountains",
    "hills",
    "lake",
    "river",
    "ocean_or_sea",
    "beach",
    "coastline",
    "forest",
    "farmland",
    "desert",
    "snow",
    "urban",
    "suburban",
    "rural",
)

# Scene types the router can return, each mapped to what an analyst would
# actually prioritise there.
_SCENE_TYPES = (
    "urban",
    "suburban",
    "rural",
    "landscape",
    "waterfront",
    "indoor",
    "vehicle_interior",
    "aerial",
)

_SCENE_ADDENDA: dict[str, str] = {
    "urban": (
        "This is a built-up street scene, so micro-detail decides the answer. "
        "Prioritise: street-name plates and their exact typography/shape, house "
        "and building numbers, shopfront and chain names, phone-number formats, "
        "postal codes, bus-stop and metro branding, traffic-light mounting style, "
        "bollard and kerb design, utility-pole style, road-marking colour and "
        "pattern, and license-plate format. Name specific streets or "
        "intersections whenever any text is legible."
    ),
    "suburban": (
        "This is a residential area. Prioritise: house style and construction "
        "material, roof pitch and covering, fence and mailbox style, driveway and "
        "kerb form, garden plant species, power distribution (overhead vs buried), "
        "street-sign design and any legible street name."
    ),
    "rural": (
        "This is a rural scene. Prioritise: crop and pasture type, field shape and "
        "boundary style (hedge, wall, wire), barn and silo architecture, farm "
        "machinery brands, soil colour, fence construction, road surface and "
        "width, roadside vegetation, and utility-pole design."
    ),
    "landscape": (
        "This is a natural landscape with few human clues, so physical geography "
        "decides it. Prioritise: tree and shrub SPECIES (they are strongly "
        "region-bound), rock type and colour, mountain profile and glaciation, "
        "snow line, soil and sand colour, water colour and clarity, sun elevation "
        "and shadow direction. Be honest that a generic forest or hillside is "
        "ambiguous and reflect that in your confidence."
    ),
    "waterfront": (
        "This is a water's edge. Prioritise: whether the water is fresh or salt "
        "(tide marks, seaweed, wave form), shoreline material, opposite-shore "
        "profile and distance, dock/jetty construction style, boat and buoy types, "
        "navigation-marker colours, and any harbour or vessel-registration text."
    ),
    "indoor": (
        "This is an interior, so location comes from fittings and text. "
        "Prioritise: electrical outlet and light-switch type (a strong country "
        "signal), radiator/HVAC style, window and door hardware, ceiling and floor "
        "construction, product packaging and brands, any text on paperwork or "
        "screens, and the view through any window. If nothing localises it, say so."
    ),
    "vehicle_interior": (
        "This is shot from inside a vehicle. Prioritise: which side the steering "
        "wheel is on, dashboard language and units (km/h vs mph), infotainment "
        "text, visible road markings and sign shapes through the glass, and any "
        "legible exterior signage or license plates."
    ),
    "aerial": (
        "This is an aerial or elevated view. Prioritise: street grid geometry and "
        "block size, roof colours and materials, field and parcel shapes, road "
        "interchange design, coastline and river form, and the layout of any "
        "distinctive large structures (stadiums, ports, airports)."
    ),
}

_SYSTEM_PROMPT = (
    "You are a world-class geolocation (OSINT) image analyst, on par with expert "
    "GeoGuessr players and intelligence analysts. Study the image extremely "
    "carefully and extract EVERY clue that helps determine where it was taken.\n\n"
    "Work through these systematically:\n"
    "- OCR: transcribe ALL readable text exactly (signs, shopfronts, billboards, "
    "license plates, documents, screens). Note the script/alphabet.\n"
    "- Language(s) visible and likely spoken.\n"
    "- Landmarks, monuments, distinctive buildings, skylines.\n"
    "- Flags, emblems, brand/chain logos that are region-specific.\n"
    "- Architecture style, building materials, roof types.\n"
    "- Vehicles: makes common to regions, license plate format/colour, and which "
    "side of the road they drive on.\n"
    "- Road markings, signage design, utility poles, bollards, fire hydrants.\n"
    "- Vegetation, crops, soil colour, terrain, mountains, coastline.\n"
    "- Climate/weather and sun position/shadows (hemisphere & rough time of day).\n\n"
    "Then reason from these clues to a RANKED list of candidate locations, most "
    "likely first. Be as specific as the evidence allows (ideally city or "
    "neighbourhood; otherwise region or country). Provide coordinates only when "
    "you are reasonably confident; otherwise use null and rely on the place name.\n\n"
    "Respond with STRICT JSON only (no markdown) matching this schema:\n"
    "{\n"
    '  "ocr_text": [string],\n'
    '  "languages": [string],\n'
    '  "landmarks": [string],\n'
    '  "flags": [string],\n'
    '  "signage": [string],\n'
    '  "vehicles_plates": [string],\n'
    '  "architecture": string,\n'
    '  "architecture_style": string,\n'
    '  "architecture_regions": [string],\n'
    '  "vegetation": string,\n'
    '  "climate": string,\n'
    '  "terrain": string,\n'
    '  "scene": {\n'
    '     "mountains": bool, "hills": bool, "lake": bool, "river": bool,\n'
    '     "ocean_or_sea": bool, "beach": bool, "coastline": bool, "forest": bool,\n'
    '     "farmland": bool, "desert": bool, "snow": bool, "urban": bool,\n'
    '     "suburban": bool, "rural": bool\n'
    "  },\n"
    '  "text_analysis": {\n'
    '     "primary_script": string, "regional_spelling": [string],\n'
    '     "implied_countries": [string]\n'
    "  },\n"
    '  "sun": {\n'
    '     "sun_visible": bool, "shadows_visible": bool,\n'
    '     "shadow_direction": string, "shadow_length": "short"|"long"|"none",\n'
    '     "approx_solar_elevation": "low"|"medium"|"high"\n'
    "  },\n"
    '  "road_side": "left"|"right"|"unknown",\n'
    '  "time_of_day": string,\n'
    '  "hemisphere_hint": string,\n'
    '  "environment": string,\n'
    '  "candidates": [\n'
    '     {"name": string, "region": string, "country": string, '
    '"latitude": number|null, "longitude": number|null, '
    '"confidence": number, "why": string}\n'
    "  ],\n"
    '  "reasoning": string,\n'
    '  "confidence": number\n'
    "}\n"
    "Rules:\n"
    "- In `scene`, set each boolean STRICTLY by what is actually visible in the "
    "image. Do not guess a feature that is not shown, and do not infer it from the "
    "candidate region. This object drives map verification.\n"
    "- In `text_analysis`, reason from any visible text: the script/alphabet, "
    "regional spelling (e.g. 'colour' vs 'color', 'Apotek' vs 'Farmacia', "
    "'Strasse' vs 'Street'), and which countries those imply.\n"
    "- In `architecture_style`, name the building style if any (e.g. 'Nordic "
    "timber', 'Soviet-era apartment block', 'Mediterranean stucco', 'American "
    "strip mall', 'Dutch rowhouse') and list the regions it implies in "
    "`architecture_regions`.\n"
    "- In `sun`, report ONLY what is visible: whether the sun disc or clear "
    "shadows are shown, the compass direction the shadows point if you can tell "
    "(e.g. 'toward the camera', 'to the left/north-east'), whether shadows are "
    "short or long, and whether the sun looks low, medium or high. Leave fields "
    "null/false if not visible. Do NOT infer these from the guessed location.\n"
    "- Provide up to 3 candidates. Each confidence and the top-level confidence "
    "are 0.0-1.0. Do not invent text you cannot actually read."
)

_ROUTER_SYSTEM = (
    "You classify photographs into one scene type. Answer with JSON only: "
    '{"scene_type": one of '
    '"urban"|"suburban"|"rural"|"landscape"|"waterfront"|"indoor"|'
    '"vehicle_interior"|"aerial"}. '
    "Use 'urban' for built-up streets with shops or multi-storey buildings, "
    "'suburban' for detached housing, 'rural' for farmland and country roads, "
    "'landscape' for nature with no significant human structures, 'waterfront' "
    "when a lake/river/sea edge dominates, 'indoor' for interiors, "
    "'vehicle_interior' when shot from inside a vehicle, and 'aerial' for "
    "overhead or high elevated views."
)

_STREET_SYSTEM = (
    "You are an OSINT analyst performing STREET-LEVEL refinement. The general "
    "region of this photo is already established; your only job is to pin down "
    "the exact spot within it. Do not re-guess the country or region.\n\n"
    "Hunt for anything that names a specific place:\n"
    "- Street-name plates, road numbers, kilometre/mile markers, exit numbers.\n"
    "- House, building and unit numbers.\n"
    "- Business, shop, restaurant, hotel and institution names (these are the "
    "single most locatable clue: a named business usually has one address).\n"
    "- Bus stop, tram stop, station and platform names.\n"
    "- Phone numbers (area codes), postal codes, web addresses on signage.\n"
    "- Plaques, memorials, notice boards, graffiti tags with place names.\n"
    "- Micro-architecture: window arrangement, balcony style, roof material and "
    "pitch, door and trim colours, facade material, storey count.\n"
    "- Fixed street furniture: hydrant, bollard, lamp-post, manhole-cover and "
    "bin designs, pavement/kerb material and pattern.\n\n"
    "Respond with STRICT JSON only:\n"
    "{\n"
    '  "street_names": [string],\n'
    '  "house_numbers": [string],\n'
    '  "business_names": [string],\n'
    '  "transit_stops": [string],\n'
    '  "postal_codes": [string],\n'
    '  "phone_numbers": [string],\n'
    '  "intersection": string|null,\n'
    '  "architectural_details": [string],\n'
    '  "street_furniture": [string],\n'
    '  "refined_place": string|null,\n'
    '  "refined_latitude": number|null,\n'
    '  "refined_longitude": number|null,\n'
    '  "precision_estimate_m": number|null,\n'
    '  "reasoning": string,\n'
    '  "confidence": number\n'
    "}\n"
    "Rules:\n"
    "- Transcribe text EXACTLY as shown; never invent or auto-complete a name you "
    "cannot actually read. An empty list is a correct and useful answer.\n"
    "- `refined_place` should be the most searchable string you can build, e.g. "
    "'Cafe Vesuvio, Columbus Avenue' or '221B Baker Street'.\n"
    "- Give `refined_latitude`/`refined_longitude` only if a named landmark or "
    "address makes you genuinely confident; otherwise null.\n"
    "- `precision_estimate_m` is how tightly you believe the spot is pinned "
    "(e.g. 50 for a specific address, 500 for the right block, 5000 for only the "
    "right district).\n"
    "- `confidence` is 0.0-1.0 for the refinement itself, not the region."
)


def _empty_scene() -> dict[str, bool]:
    return {key: False for key in _SCENE_KEYS}


def _empty_analysis(note: str) -> dict[str, Any]:
    return {
        "available": False,
        "note": note,
        "ocr_text": [],
        "languages": [],
        "landmarks": [],
        "flags": [],
        "signage": [],
        "vehicles_plates": [],
        "architecture": None,
        "architecture_style": None,
        "architecture_regions": [],
        "vegetation": None,
        "climate": None,
        "terrain": None,
        "scene": _empty_scene(),
        "scene_type": None,
        "text_analysis": {
            "primary_script": None,
            "regional_spelling": [],
            "implied_countries": [],
        },
        "sun": {
            "sun_visible": False,
            "shadows_visible": False,
            "shadow_direction": None,
            "shadow_length": None,
            "approx_solar_elevation": None,
        },
        "road_side": "unknown",
        "time_of_day": None,
        "hemisphere_hint": None,
        "environment": None,
        "candidates": [],
        "best_guess_location": {
            "name": None,
            "country": None,
            "latitude": None,
            "longitude": None,
        },
        "reasoning": None,
        "confidence": 0.0,
    }


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in ("true", "yes", "1")
    return False


def _str_list(raw: Any, limit: int = 12) -> list[str]:
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        if item is None:
            continue
        text = str(item).strip()
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            out.append(text)
        if len(out) >= limit:
            break
    return out


def _normalize_scene(raw: Any) -> dict[str, bool]:
    scene = _empty_scene()
    if isinstance(raw, dict):
        for key in _SCENE_KEYS:
            if key in raw:
                scene[key] = _as_bool(raw[key])
    return scene


def _normalize_text_analysis(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "primary_script": None,
        "regional_spelling": [],
        "implied_countries": [],
    }
    if isinstance(raw, dict):
        script = raw.get("primary_script")
        if isinstance(script, str) and script.strip():
            out["primary_script"] = script.strip()
        for key in ("regional_spelling", "implied_countries"):
            val = raw.get(key)
            if isinstance(val, list):
                out[key] = [str(v) for v in val if v]
    return out


def _normalize_sun(raw: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "sun_visible": False,
        "shadows_visible": False,
        "shadow_direction": None,
        "shadow_length": None,
        "approx_solar_elevation": None,
    }
    if isinstance(raw, dict):
        out["sun_visible"] = _as_bool(raw.get("sun_visible"))
        out["shadows_visible"] = _as_bool(raw.get("shadows_visible"))
        for key in ("shadow_direction", "shadow_length", "approx_solar_elevation"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip() and val.strip().lower() != "none":
                out[key] = val.strip()
    return out


def _normalize_candidate(raw: dict[str, Any]) -> dict[str, Any]:
    lat = raw.get("latitude")
    lon = raw.get("longitude")
    try:
        lat = float(lat) if lat is not None else None
        lon = float(lon) if lon is not None else None
    except (TypeError, ValueError):
        lat, lon = None, None
    return {
        "name": raw.get("name"),
        "region": raw.get("region"),
        "country": raw.get("country"),
        "latitude": lat,
        "longitude": lon,
        "confidence": _clamp(raw.get("confidence", 0.0)),
        "why": raw.get("why"),
    }


def classify_scene(image_path: str) -> Optional[str]:
    """Cheap low-detail pass returning one of `_SCENE_TYPES` (or None).

    Uses `detail="low"` and a tiny token budget, so it costs a small fraction of
    the main analysis while letting that analysis be scene-specific.
    """
    parsed, _ = call_vision_json(
        image_path,
        _ROUTER_SYSTEM,
        "Classify this photograph's scene type.",
        temperature=0.0,
        detail="low",
        max_tokens=60,
    )
    if not parsed:
        return None
    value = str(parsed.get("scene_type") or "").strip().lower().replace(" ", "_")
    return value if value in _SCENE_TYPES else None


def analyze_image(
    image_path: str, scene_type: Optional[str] = None
) -> dict[str, Any]:
    """Run the vision model on the image and return structured clues + candidates.

    `scene_type` skips the routing call when the caller already knows the scene
    (e.g. every frame of one video shares a type).
    """
    if scene_type is None and SCENE_ROUTING_ENABLED:
        scene_type = classify_scene(image_path)

    system = _SYSTEM_PROMPT
    addendum = _SCENE_ADDENDA.get(scene_type or "")
    if addendum:
        system = f"{system}\n\nSCENE-SPECIFIC FOCUS ({scene_type}):\n{addendum}"

    parsed, note = call_vision_json(
        image_path,
        system,
        "Analyze this image for geolocation clues and rank candidate locations.",
    )
    if parsed is None:
        return _empty_analysis(note)

    analysis = _empty_analysis("ok")
    analysis["available"] = True
    analysis["note"] = "ok"
    analysis["scene_type"] = scene_type

    for key in (
        "ocr_text",
        "languages",
        "landmarks",
        "flags",
        "signage",
        "vehicles_plates",
        "architecture",
        "architecture_style",
        "architecture_regions",
        "vegetation",
        "climate",
        "terrain",
        "road_side",
        "time_of_day",
        "hemisphere_hint",
        "environment",
        "reasoning",
    ):
        if parsed.get(key) is not None:
            analysis[key] = parsed[key]

    analysis["scene"] = _normalize_scene(parsed.get("scene"))
    analysis["text_analysis"] = _normalize_text_analysis(parsed.get("text_analysis"))
    analysis["sun"] = _normalize_sun(parsed.get("sun"))

    candidates_raw = parsed.get("candidates")
    if isinstance(candidates_raw, list) and candidates_raw:
        candidates = [
            _normalize_candidate(c) for c in candidates_raw if isinstance(c, dict)
        ]
        candidates.sort(key=lambda c: c["confidence"], reverse=True)
        analysis["candidates"] = candidates

    # Backward-compatible single best guess = top candidate (or legacy field).
    if analysis["candidates"]:
        top = analysis["candidates"][0]
        analysis["best_guess_location"] = {
            "name": top.get("name"),
            "country": top.get("country"),
            "latitude": top.get("latitude"),
            "longitude": top.get("longitude"),
        }
    else:
        legacy = parsed.get("best_guess_location")
        if isinstance(legacy, dict):
            analysis["best_guess_location"] = {
                "name": legacy.get("name"),
                "country": legacy.get("country"),
                "latitude": legacy.get("latitude"),
                "longitude": legacy.get("longitude"),
            }
            analysis["candidates"] = [
                _normalize_candidate(
                    {**legacy, "confidence": parsed.get("confidence", 0.0)}
                )
            ]

    analysis["confidence"] = _clamp(parsed.get("confidence", 0.0))
    return analysis


def empty_street_analysis(note: str) -> dict[str, Any]:
    return {
        "available": False,
        "note": note,
        "street_names": [],
        "house_numbers": [],
        "business_names": [],
        "transit_stops": [],
        "postal_codes": [],
        "phone_numbers": [],
        "intersection": None,
        "architectural_details": [],
        "street_furniture": [],
        "refined_place": None,
        "refined_latitude": None,
        "refined_longitude": None,
        "precision_estimate_m": None,
        "reasoning": None,
        "confidence": 0.0,
    }


def analyze_street_level(
    image_path: str,
    *,
    latitude: float,
    longitude: float,
    radius_km: float,
    place_label: Optional[str] = None,
    scene_type: Optional[str] = None,
) -> dict[str, Any]:
    """Second, anchored pass that hunts for street-level identifiers.

    The anchor (coordinates plus a human place label) is given to the model so it
    can stop reasoning about which country it is in and spend its whole attention
    on text and micro-detail.
    """
    where = place_label or f"{round(latitude, 3)}, {round(longitude, 3)}"
    user = (
        f"This photo was taken within roughly {radius_km:.0f} km of "
        f"{latitude:.4f}, {longitude:.4f} ({where}). "
        "Identify the exact spot within that area: read every street name, "
        "building number, business name and sign, and describe the "
        "architectural micro-details. Return coordinates only if a named place "
        "or address genuinely pins it down."
    )
    system = _STREET_SYSTEM
    addendum = _SCENE_ADDENDA.get(scene_type or "")
    if addendum:
        system = f"{system}\n\nSCENE-SPECIFIC FOCUS ({scene_type}):\n{addendum}"

    parsed, note = call_vision_json(image_path, system, user, temperature=0.1)
    if parsed is None:
        return empty_street_analysis(note)

    out = empty_street_analysis("ok")
    out["available"] = True
    for key in (
        "street_names",
        "house_numbers",
        "business_names",
        "transit_stops",
        "postal_codes",
        "phone_numbers",
        "architectural_details",
        "street_furniture",
    ):
        out[key] = _str_list(parsed.get(key))

    for key in ("intersection", "refined_place", "reasoning"):
        val = parsed.get(key)
        if isinstance(val, str) and val.strip():
            out[key] = val.strip()

    for key, target in (
        ("refined_latitude", "refined_latitude"),
        ("refined_longitude", "refined_longitude"),
    ):
        try:
            val = parsed.get(key)
            out[target] = float(val) if val is not None else None
        except (TypeError, ValueError):
            out[target] = None

    try:
        prec = parsed.get("precision_estimate_m")
        out["precision_estimate_m"] = float(prec) if prec is not None else None
    except (TypeError, ValueError):
        out["precision_estimate_m"] = None

    out["confidence"] = _clamp(parsed.get("confidence", 0.0))
    return out
