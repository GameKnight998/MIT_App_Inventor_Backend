"""Identify vehicles and read license plates as geolocation evidence.

A car is one of the densest location clues in a photo. Its plate format and
colour narrow the country hard, the model and trim reveal which market it was
sold in, and the generation bounds when the photo could have been taken. A
professional analyst reads a car before they read the trees.

Two stages, the first optional:

  1. Detection (`ROBOFLOW_API_KEY`, optional). A hosted object detector finds
     vehicle boxes cheaply. When available we crop to the largest box before
     asking the vision model, which is a real accuracy win: the model stops
     describing the whole street and looks at the grille.
  2. Identification. A vision pass with a vehicle-specialist prompt returns
     make/model/generation/year plus a structured plate reading.

Detection is genuinely optional. Without a Roboflow key the whole-image path is
used, which still works because modern vision models identify cars well; the crop
just improves the odds on small or partial vehicles.

`implied_regions` is the part the OSINT engine consumes: the countries the plates
and market-specific models point to, used to corroborate or contradict the
region chosen from the scene.
"""

from __future__ import annotations

import base64
import os
from typing import Any, Optional

from utils.visionclient import call_vision_json

VEHICLE_ID_ENABLED = os.getenv("VEHICLE_ID_ENABLED", "1") not in ("0", "false", "False")
VEHICLE_DETECT_ENABLED = os.getenv("VEHICLE_DETECT_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
ROBOFLOW_MODEL = os.getenv("ROBOFLOW_MODEL", "vehicle-detection-3mmwj/1")
ROBOFLOW_URL = os.getenv("ROBOFLOW_URL", "https://detect.roboflow.com")
ROBOFLOW_TIMEOUT = float(os.getenv("ROBOFLOW_TIMEOUT", "10"))
DETECT_MIN_CONFIDENCE = float(os.getenv("VEHICLE_DETECT_MIN_CONFIDENCE", "0.7"))
# Padding around a detected box, as a fraction of its size. Some context helps
# the model judge scale and body style.
_CROP_PAD = 0.18

_VEHICLE_CLASSES = (
    "car",
    "truck",
    "bus",
    "van",
    "motorcycle",
    "motorbike",
    "vehicle",
    "suv",
    "pickup",
)

_SYSTEM_PROMPT = (
    "You are a vehicle identification specialist supporting a geolocation "
    "investigation. Identify every clearly visible vehicle and read every "
    "license plate.\n\n"
    "For identification, reason from: body lines and proportions, grille shape "
    "and pattern, headlight and tail-light signature, badges, wheel and alloy "
    "design, window and pillar shape, mirror style, and trim details. Partial or "
    "angled views are normal: infer what you can and lower your confidence "
    "accordingly.\n\n"
    "For plates, the FORMAT is more valuable than the characters: describe the "
    "plate's shape, colour scheme, character grouping, and any region code, EU "
    "band, province name or state slogan, then say which country or region that "
    "format implies.\n\n"
    "Also note market-specific signals: which side the steering wheel is on, "
    "whether the model is sold only in certain markets, mandatory equipment "
    "(e.g. daytime running lights, amber vs red rear indicators), taxi/police/"
    "emergency liveries, and commercial signage on the vehicle.\n\n"
    "Respond with STRICT JSON only:\n"
    "{\n"
    '  "vehicles_present": bool,\n'
    '  "vehicles": [\n'
    "    {\n"
    '      "make": string|null, "model": string|null,\n'
    '      "generation": string|null, "year_range": string|null,\n'
    '      "body_style": string|null, "color": string|null,\n'
    '      "steering_side": "left"|"right"|"unknown",\n'
    '      "market_hint": string|null,\n'
    '      "distinguishing_features": [string],\n'
    '      "confidence": number\n'
    "    }\n"
    "  ],\n"
    '  "plates": [\n'
    "    {\n"
    '      "text": string|null, "format_description": string|null,\n'
    '      "region_implied": string|null, "colour": string|null,\n'
    '      "confidence": number\n'
    "    }\n"
    "  ],\n"
    '  "regional_indicators": [string],\n'
    '  "implied_regions": [string],\n'
    '  "notes": string,\n'
    '  "confidence": number\n'
    "}\n"
    "Rules:\n"
    "- If no vehicle is visible, set vehicles_present false and return empty lists.\n"
    "- Never invent plate characters you cannot actually read; use null for "
    "`text` and still describe the format.\n"
    "- `implied_regions` lists country or region names (English) that the "
    "vehicles and plates together point to, most likely first.\n"
    "- All confidences are 0.0-1.0."
)


def _clamp(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _str_list(raw: Any, limit: int = 8) -> list[str]:
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


def empty_vehicle_report(note: str) -> dict[str, Any]:
    return {
        "available": False,
        "note": note,
        "vehicles_present": False,
        "vehicles": [],
        "plates": [],
        "regional_indicators": [],
        "implied_regions": [],
        "detector": None,
        "detections": [],
        "confidence": 0.0,
    }


def should_identify(vision: Optional[dict[str, Any]]) -> bool:
    """Whether the scene plausibly contains a vehicle worth a dedicated pass.

    Gating on the main analysis we already paid for means landscape and interior
    photos never spend a vision call looking for cars that are not there.
    """
    if not VEHICLE_ID_ENABLED or not vision:
        return False
    if vision.get("vehicles_plates"):
        return True
    if (vision.get("road_side") or "unknown") != "unknown":
        return True
    if vision.get("scene_type") in ("urban", "suburban", "vehicle_interior"):
        return True
    scene = vision.get("scene") or {}
    return bool(scene.get("urban") or scene.get("suburban"))


def _roboflow_detect(image_path: str) -> list[dict[str, Any]]:
    """Detect vehicle boxes via Roboflow's hosted inference API (or return [])."""
    if not VEHICLE_DETECT_ENABLED or not ROBOFLOW_API_KEY:
        return []
    try:
        import requests

        with open(image_path, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("utf-8")
        resp = requests.post(
            f"{ROBOFLOW_URL.rstrip('/')}/{ROBOFLOW_MODEL}",
            params={"api_key": ROBOFLOW_API_KEY},
            data=encoded,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            timeout=ROBOFLOW_TIMEOUT,
        )
        resp.raise_for_status()
        predictions = resp.json().get("predictions") or []
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for pred in predictions:
        label = str(pred.get("class") or "").lower()
        conf = _clamp(pred.get("confidence"))
        if conf < DETECT_MIN_CONFIDENCE:
            continue
        if label and not any(v in label for v in _VEHICLE_CLASSES):
            continue
        out.append(
            {
                "label": label or "vehicle",
                "confidence": round(conf, 3),
                "x": pred.get("x"),
                "y": pred.get("y"),
                "width": pred.get("width"),
                "height": pred.get("height"),
            }
        )
    out.sort(
        key=lambda d: (d.get("width") or 0) * (d.get("height") or 0), reverse=True
    )
    return out


def _crop_to_detection(
    image_path: str, detection: dict[str, Any]
) -> Optional[str]:
    """Write a padded crop around a detected vehicle; return its path or None."""
    try:
        import cv2  # type: ignore

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            return None
        img_h, img_w = image.shape[:2]
        cx = float(detection["x"])
        cy = float(detection["y"])
        bw = float(detection["width"])
        bh = float(detection["height"])
        pad_w = bw * _CROP_PAD
        pad_h = bh * _CROP_PAD
        x1 = max(0, int(cx - bw / 2 - pad_w))
        y1 = max(0, int(cy - bh / 2 - pad_h))
        x2 = min(img_w, int(cx + bw / 2 + pad_w))
        y2 = min(img_h, int(cy + bh / 2 + pad_h))
        if x2 - x1 < 40 or y2 - y1 < 40:
            return None
        crop = image[y1:y2, x1:x2]
        # Small crops lose plate legibility, so enlarge modestly.
        if max(crop.shape[:2]) < 640:
            scale = 640 / max(crop.shape[:2])
            crop = cv2.resize(
                crop,
                (int(crop.shape[1] * scale), int(crop.shape[0] * scale)),
                interpolation=cv2.INTER_LANCZOS4,
            )
        out = f"{os.path.splitext(image_path)[0]}_vehicle.jpg"
        if not cv2.imwrite(out, crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            return None
        return out
    except Exception:
        return None


def identify_vehicles(image_path: str) -> dict[str, Any]:
    """Identify vehicles and plates in an image. Never raises."""
    if not VEHICLE_ID_ENABLED:
        return empty_vehicle_report("Vehicle identification disabled.")

    detections = _roboflow_detect(image_path)
    target = image_path
    crop_path: Optional[str] = None
    if detections:
        crop_path = _crop_to_detection(image_path, detections[0])
        if crop_path:
            target = crop_path

    user = "Identify every vehicle and read every license plate in this image."
    if detections:
        user += (
            f" A detector found {len(detections)} vehicle(s); this image is "
            "cropped to the most prominent one."
        )

    try:
        parsed, note = call_vision_json(target, _SYSTEM_PROMPT, user, temperature=0.1)
    finally:
        if crop_path:
            try:
                os.unlink(crop_path)
            except OSError:
                pass

    if parsed is None:
        report = empty_vehicle_report(note)
        report["detections"] = detections
        report["detector"] = "roboflow" if detections else None
        return report

    report = empty_vehicle_report("ok")
    report["available"] = True
    report["note"] = "ok"
    report["detections"] = detections
    report["detector"] = "vision+roboflow" if detections else "vision"

    vehicles: list[dict[str, Any]] = []
    for raw in parsed.get("vehicles") or []:
        if not isinstance(raw, dict):
            continue
        if not (raw.get("make") or raw.get("model") or raw.get("body_style")):
            continue
        vehicles.append(
            {
                "make": raw.get("make"),
                "model": raw.get("model"),
                "generation": raw.get("generation"),
                "year_range": raw.get("year_range"),
                "body_style": raw.get("body_style"),
                "color": raw.get("color"),
                "steering_side": raw.get("steering_side") or "unknown",
                "market_hint": raw.get("market_hint"),
                "distinguishing_features": _str_list(
                    raw.get("distinguishing_features"), 6
                ),
                "confidence": _clamp(raw.get("confidence")),
            }
        )
    vehicles.sort(key=lambda v: v["confidence"], reverse=True)
    report["vehicles"] = vehicles[:5]

    plates: list[dict[str, Any]] = []
    for raw in parsed.get("plates") or []:
        if not isinstance(raw, dict):
            continue
        if not (raw.get("text") or raw.get("format_description")):
            continue
        plates.append(
            {
                "text": raw.get("text"),
                "format_description": raw.get("format_description"),
                "region_implied": raw.get("region_implied"),
                "colour": raw.get("colour"),
                "confidence": _clamp(raw.get("confidence")),
            }
        )
    report["plates"] = plates[:5]

    report["regional_indicators"] = _str_list(parsed.get("regional_indicators"))
    implied = _str_list(parsed.get("implied_regions"))
    # Plate-derived regions are the strongest signal here, so fold them in even
    # if the model forgot to repeat them in implied_regions.
    for plate in plates:
        if plate.get("region_implied"):
            region = str(plate["region_implied"]).strip()
            if region and region.lower() not in {r.lower() for r in implied}:
                implied.append(region)
    report["implied_regions"] = implied[:8]

    report["vehicles_present"] = bool(
        vehicles or plates or parsed.get("vehicles_present")
    )
    report["confidence"] = _clamp(parsed.get("confidence"))
    if isinstance(parsed.get("notes"), str) and parsed["notes"].strip():
        report["notes"] = parsed["notes"].strip()
    return report


def vehicle_evidence(report: Optional[dict[str, Any]]) -> list[str]:
    """Human-readable evidence lines describing what the vehicles reveal."""
    if not report or not report.get("available"):
        return []
    lines: list[str] = []
    for veh in report.get("vehicles", [])[:3]:
        label = " ".join(
            str(p) for p in (veh.get("make"), veh.get("model")) if p
        ).strip()
        if not label:
            label = veh.get("body_style") or "vehicle"
        detail = ", ".join(
            str(p)
            for p in (veh.get("generation"), veh.get("year_range"), veh.get("color"))
            if p
        )
        lines.append(
            f"Vehicle identified: {label}"
            + (f" ({detail})" if detail else "")
            + f" - confidence {veh.get('confidence')}."
        )
    for plate in report.get("plates", [])[:2]:
        bits = [p for p in (plate.get("text"), plate.get("format_description")) if p]
        if bits:
            line = "License plate: " + " - ".join(str(b) for b in bits)
            if plate.get("region_implied"):
                line += f" (implies {plate['region_implied']})"
            lines.append(line + ".")
    if report.get("implied_regions"):
        lines.append(
            "Vehicles imply: " + ", ".join(report["implied_regions"][:4]) + "."
        )
    return lines
