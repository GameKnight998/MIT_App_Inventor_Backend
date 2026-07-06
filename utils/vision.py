"""Send an image to an AI vision model to extract location-relevant clues.

The model acts like a professional geolocation (OSINT) analyst. It reads visible
text (OCR), identifies landmarks, flags, signage, license plates, languages,
architecture, vegetation, climate, terrain, driving side and sun/shadow hints,
then proposes a RANKED list of candidate locations with confidences.

If no API key is configured the module degrades gracefully and returns an empty
analysis so the rest of the pipeline still runs (e.g. EXIF-only locating).
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")

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
    "- Provide up to 3 candidates. Each confidence and the top-level confidence "
    "are 0.0-1.0. Do not invent text you cannot actually read."
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
        "text_analysis": {
            "primary_script": None,
            "regional_spelling": [],
            "implied_countries": [],
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


def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


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


def analyze_image(image_path: str) -> dict[str, Any]:
    """Run the vision model on the image and return structured clues + candidates."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return _empty_analysis("OPENAI_API_KEY not set; vision analysis skipped.")

    try:
        from openai import OpenAI
    except ImportError:
        return _empty_analysis("openai package not installed; vision skipped.")

    try:
        client = OpenAI(api_key=api_key)
        b64 = _encode_image(image_path)

        response = client.chat.completions.create(
            model=VISION_MODEL,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Analyze this image for geolocation clues and rank candidate locations.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64}",
                                "detail": "high",
                            },
                        },
                    ],
                },
            ],
        )

        content = response.choices[0].message.content or "{}"
        parsed = json.loads(content)
    except Exception as exc:
        return _empty_analysis(f"Vision call failed: {exc}")

    analysis = _empty_analysis("ok")
    analysis["available"] = True
    analysis["note"] = "ok"

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
