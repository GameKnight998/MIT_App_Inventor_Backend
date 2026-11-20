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
import io
import json
import os
from typing import Any, Optional

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-6.1-terra")

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
    '  "named_landmarks": [string],\n'
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
    '  "reasoning": string,\n'
    '  "candidates": [\n'
    '     {"name": string, "region": string, "country": string, '
    '"latitude": number|null, "longitude": number|null, '
    '"confidence": number, "why": string}\n'
    "  ],\n"
    '  "confidence": number\n'
    "}\n"
    "Rules:\n"
    "- `landmarks` may describe notable structures in words (e.g. 'gothic clock "
    "tower'). `named_landmarks` must contain ONLY proper names of specific, "
    "identifiable places you can actually recognise or read (e.g. 'Space Needle', "
    "'Lake Chelan', 'Golden Gate Bridge'). Never put generic descriptions in "
    "`named_landmarks`; leave it empty if nothing can be named. These names are "
    "looked up on a map, so a wrong or generic entry counts against the answer.\n"
    "- When extra zoomed-in crops of the photo are attached, use them to read "
    "small text, signs and plates. Describe the scene from the full photo.\n"
    "- Fill the fields in the order shown. Write `reasoning` BEFORE `candidates`: "
    "weigh the clues step by step (what each one rules in or out, and where they "
    "agree or conflict), then derive the candidates from that reasoning.\n"
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


def _empty_scene() -> dict[str, bool]:
    return {key: False for key in _SCENE_KEYS}


def _empty_analysis(note: str) -> dict[str, Any]:
    return {
        "available": False,
        "note": note,
        "ocr_text": [],
        "languages": [],
        "landmarks": [],
        "named_landmarks": [],
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


# Independent answers requested per image. Agreement between them is used as a
# confidence signal (see utils/consensus.py). 1 disables sampling.
VISION_SAMPLES = max(1, int(os.getenv("VISION_SAMPLES", "3")))
VISION_TEMPERATURE = float(os.getenv("VISION_TEMPERATURE", "0.2"))
# Samples need some diversity, or agreement says nothing.
VISION_SAMPLE_TEMPERATURE = float(os.getenv("VISION_SAMPLE_TEMPERATURE", "0.8"))
# Zoomed-in quadrant crops help the model read small signs and plates. Only
# added for photos large enough that the API's downscaling loses that detail.
VISION_TILES = os.getenv("VISION_TILES", "1") not in ("0", "false", "False")
TILE_MIN_SIDE = int(os.getenv("VISION_TILE_MIN_SIDE", "1600"))
TILE_MAX_SIDE = int(os.getenv("VISION_TILE_MAX_SIDE", "1024"))

# Longest side sent to the model. High-detail vision input is downscaled to
# about this size by the API anyway, so larger uploads only add latency.
MAX_IMAGE_SIDE = int(os.getenv("VISION_MAX_IMAGE_SIDE", "2048"))


def _encode_image(image_path: str) -> str:
    """Return the image as base64 JPEG, whatever format was uploaded.

    The request labels the data as image/jpeg, so PNG, WebP, GIF, BMP and TIFF
    uploads are re-encoded to real JPEG (alpha flattened onto white, EXIF
    orientation applied) rather than sent under the wrong media type.
    """
    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as img:
            img = ImageOps.exif_transpose(img)
            if img.mode in ("RGBA", "LA", "P"):
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[-1])
                img = background
            elif img.mode != "RGB":
                img = img.convert("RGB")
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            return base64.b64encode(buf.getvalue()).decode("utf-8")
    except Exception:
        # Fall back to the original bytes rather than failing the analysis.
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


def _to_jpeg_b64(img: Any, max_side: int) -> str:
    img = img.copy()
    img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _encode_tiles(image_path: str) -> list[str]:
    """Four overlapping quadrant crops of a large photo, as base64 JPEG."""
    if not VISION_TILES:
        return []
    try:
        from PIL import Image, ImageOps

        with Image.open(image_path) as raw:
            img = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception:
        return []
    w, h = img.size
    if max(w, h) < TILE_MIN_SIDE:
        return []
    # 10% overlap so text on a quadrant boundary is whole in at least one crop.
    half_w, half_h = int(w * 0.55), int(h * 0.55)
    boxes = (
        (0, 0, half_w, half_h),
        (w - half_w, 0, w, half_h),
        (0, h - half_h, half_w, h),
        (w - half_w, h - half_h, w, h),
    )
    return [_to_jpeg_b64(img.crop(box), TILE_MAX_SIDE) for box in boxes]


def _user_prompt(image_path: str) -> tuple[str, list[str]]:
    """The user text and the base64 JPEG images (full photo, then any crops)."""
    tiles = _encode_tiles(image_path)
    text = "Analyze this image for geolocation clues and rank candidate locations."
    if tiles:
        text += (
            " Image 1 is the full photo. Images 2-5 are zoomed-in crops of its "
            "top-left, top-right, bottom-left and bottom-right quadrants, to help "
            "you read small text."
        )
    return text, [_encode_image(image_path), *tiles]


def analyze_image_samples(
    image_path: str, samples: Optional[int] = None, model: Optional[str] = None
) -> list[dict[str, Any]]:
    """Run the vision model and return one parsed analysis per sample.

    `model` overrides VISION_MODEL; its name picks the provider (see
    utils/providers.py), so OpenAI, Claude and Gemini models all work.
    """
    from utils.providers import missing_key, request_samples

    model = model or VISION_MODEL
    key = missing_key(model)
    if key:
        return [_empty_analysis(f"{key} not set; vision analysis skipped for {model}.")]

    n = max(1, samples or VISION_SAMPLES)
    temperature = VISION_SAMPLE_TEMPERATURE if n > 1 else VISION_TEMPERATURE
    try:
        text, images = _user_prompt(image_path)
        raw_answers = request_samples(model, _SYSTEM_PROMPT, text, images, n, temperature)
    except Exception as exc:
        return [_empty_analysis(f"Vision call failed ({model}): {exc}")]

    out: list[dict[str, Any]] = []
    for raw in raw_answers:
        try:
            parsed = json.loads(raw)
        except (TypeError, ValueError):
            continue
        if isinstance(parsed, dict):
            analysis = _parse_analysis(parsed)
            analysis["model"] = model
            out.append(analysis)
    return out or [_empty_analysis(f"{model} returned no parseable JSON.")]


def analyze_image(image_path: str, model: Optional[str] = None) -> dict[str, Any]:
    """Run the vision model (several samples) and return the consensus analysis."""
    from utils.consensus import merge_samples

    return merge_samples(analyze_image_samples(image_path, model=model))


def _parse_analysis(parsed: dict[str, Any]) -> dict[str, Any]:
    """Normalise one raw model answer into the analysis shape."""
    analysis = _empty_analysis("ok")
    analysis["available"] = True
    analysis["note"] = "ok"

    for key in (
        "ocr_text",
        "languages",
        "landmarks",
        "named_landmarks",
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
