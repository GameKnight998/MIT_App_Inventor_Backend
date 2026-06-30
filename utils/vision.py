"""Send an image to an AI vision model to extract location-relevant clues.

The model is asked to act like an image analyst: read any visible text (OCR),
identify landmarks, signage, license plates, languages, vegetation, architecture
and anything else that hints at where a photo was taken.

If no API key is configured the module degrades gracefully and returns an empty
analysis so the rest of the pipeline keeps working (e.g. EXIF-only locating).
"""

from __future__ import annotations

import base64
import json
import os
from typing import Any

VISION_MODEL = os.getenv("VISION_MODEL", "gpt-4o")

_SYSTEM_PROMPT = (
    "You are a meticulous geolocation (OSINT) image analyst. Examine the image "
    "and extract every clue that could help determine where it was taken. "
    "Read all visible text exactly (OCR), including signs, shopfronts, license "
    "plates, billboards and documents. Identify landmarks, architecture style, "
    "natural features, vegetation, road markings, vehicles, and the apparent "
    "language(s). Do NOT guess wildly: only state what you can actually see. "
    "Respond with STRICT JSON only, no markdown, matching this schema:\n"
    "{\n"
    '  "ocr_text": [string],\n'
    '  "landmarks": [string],\n'
    '  "languages": [string],\n'
    '  "signage": [string],\n'
    '  "vehicles_plates": [string],\n'
    '  "environment": string,\n'
    '  "best_guess_location": {"name": string, "country": string, '
    '"latitude": number|null, "longitude": number|null},\n'
    '  "reasoning": string,\n'
    '  "confidence": number\n'
    "}\n"
    "confidence is 0.0-1.0 for how sure you are about best_guess_location."
)


def _empty_analysis(note: str) -> dict[str, Any]:
    return {
        "available": False,
        "note": note,
        "ocr_text": [],
        "landmarks": [],
        "languages": [],
        "signage": [],
        "vehicles_plates": [],
        "environment": None,
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


def analyze_image(image_path: str) -> dict[str, Any]:
    """Run the vision model on the image and return structured clues."""
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
                            "text": "Analyze this image for geolocation clues.",
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
        analysis = _empty_analysis(f"Vision call failed: {exc}")
        return analysis

    # Merge model output onto the canonical shape so missing keys are safe.
    analysis = _empty_analysis("ok")
    analysis["available"] = True
    analysis["note"] = "ok"
    for key in (
        "ocr_text",
        "landmarks",
        "languages",
        "signage",
        "vehicles_plates",
        "environment",
        "reasoning",
    ):
        if key in parsed and parsed[key] is not None:
            analysis[key] = parsed[key]

    guess = parsed.get("best_guess_location") or {}
    if isinstance(guess, dict):
        analysis["best_guess_location"] = {
            "name": guess.get("name"),
            "country": guess.get("country"),
            "latitude": guess.get("latitude"),
            "longitude": guess.get("longitude"),
        }

    try:
        analysis["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0.0))))
    except (TypeError, ValueError):
        analysis["confidence"] = 0.0

    return analysis
