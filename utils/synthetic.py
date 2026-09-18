"""Decide whether an image is a real photograph before trying to locate it.

This matters more here than it looks. Every other module in this pipeline assumes
the pixels depict a real place; if the image was generated, the "location" it
implies is fiction and a confident answer is actively harmful. So this runs
early, and a positive result caps confidence and warns rather than being a
footnote.

Detection is metadata-first, on purpose. The strongest evidence that an image was
generated is that the generator said so, and modern tools say so in three ways:

  * C2PA / Content Credentials - a signed provenance manifest embedded by
    OpenAI, Adobe, Leica and others, which names the generating tool;
  * IPTC `DigitalSourceType` - the standardised `trainedAlgorithmicMedia` value
    that Adobe, Google and others write for AI output;
  * tool-specific traces - Stable Diffusion writes its full prompt, sampler,
    steps and seed into PNG text chunks; ComfyUI embeds its whole workflow graph.

These are near-zero-cost, need no model, and give a definitive answer when
present. Absence is not innocence, so weak structural heuristics (a bare
1024x1024 PNG with no camera metadata is not something a camera produces) add a
"suspicious" tier without ever asserting a verdict alone.

For deployments that can afford it, a classifier backend can be enabled
(`SYNTHETIC_MODEL_ENABLED` with transformers installed, or `SYNTHETIC_API_URL`
for a hosted detector). Both are optional and lazily loaded: the default install
carries no ML weight, which is what lets this run on a 512 MB free tier.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

SYNTHETIC_ENABLED = os.getenv("SYNTHETIC_DETECTION_ENABLED", "1") not in (
    "0",
    "false",
    "False",
)
SYNTHETIC_AI_THRESHOLD = float(os.getenv("SYNTHETIC_AI_THRESHOLD", "0.8"))
SYNTHETIC_DEEPFAKE_THRESHOLD = float(os.getenv("SYNTHETIC_DEEPFAKE_THRESHOLD", "0.7"))
# Heavy classifier backends, both off by default.
SYNTHETIC_MODEL_ENABLED = os.getenv("SYNTHETIC_MODEL_ENABLED", "0") not in (
    "0",
    "false",
    "False",
)
SYNTHETIC_AI_MODEL = os.getenv("SYNTHETIC_AI_MODEL", "umm-maybe/AI-image-detector")
SYNTHETIC_DEEPFAKE_MODEL = os.getenv(
    "SYNTHETIC_DEEPFAKE_MODEL", "dima806/deepfake_vs_real_image_detection"
)
SYNTHETIC_API_URL = os.getenv("SYNTHETIC_API_URL", "")
SYNTHETIC_API_KEY = os.getenv("SYNTHETIC_API_KEY", "")
SYNTHETIC_API_TIMEOUT = float(os.getenv("SYNTHETIC_API_TIMEOUT", "15"))

# Metadata is at the head of a file, but PNG text chunks can trail the image
# data, so we scan both ends instead of loading a whole 25 MB upload.
_HEAD_BYTES = 1_048_576
_TAIL_BYTES = 262_144

# Named generators, matched case-insensitively against file bytes and EXIF.
_GENERATOR_SIGNATURES: tuple[tuple[str, str], ...] = (
    ("midjourney", "Midjourney"),
    ("stable diffusion", "Stable Diffusion"),
    ("stable-diffusion", "Stable Diffusion"),
    ("stablediffusionxl", "Stable Diffusion XL"),
    ("sdxl", "Stable Diffusion XL"),
    ("automatic1111", "Stable Diffusion (AUTOMATIC1111)"),
    ("comfyui", "ComfyUI"),
    ("invokeai", "InvokeAI"),
    ("novelai", "NovelAI"),
    ("dall-e", "DALL-E"),
    ("dall·e", "DALL-E"),
    ("dalle", "DALL-E"),
    ("adobe firefly", "Adobe Firefly"),
    ("firefly", "Adobe Firefly"),
    ("imagen", "Google Imagen"),
    ("leonardo.ai", "Leonardo.Ai"),
    ("ideogram", "Ideogram"),
    ("playground v2", "Playground"),
    ("flux.1", "FLUX"),
    ("black forest labs", "FLUX (Black Forest Labs)"),
    ("made with google ai", "Google AI"),
    ("gemini", "Google Gemini"),
    ("grok-image", "Grok"),
    ("recraft", "Recraft"),
)

# Stable-Diffusion-family parameter blocks written into PNG tEXt/iTXt chunks.
_PROMPT_CHUNK_MARKERS = (
    "negative prompt:",
    "sd-metadata",
    '"sampler_name"',
    '"class_type"',  # ComfyUI workflow graph
    "denoising_strength",
)
_PARAM_PATTERN = re.compile(
    r"steps:\s*\d+.*?(sampler|cfg scale|seed)\s*:", re.I | re.S
)

# The official IPTC values that declare synthetic origin.
_IPTC_AI_VALUES = (
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
)

_C2PA_MARKERS = ("c2pa", "jumbf", "contentcredentials", "claim_generator")

# Canvas sizes characteristic of image generators. A camera does not shoot these.
_GENERATOR_SIZES = {
    (512, 512),
    (768, 768),
    (1024, 1024),
    (2048, 2048),
    (1024, 1536),
    (1536, 1024),
    (1024, 1792),
    (1792, 1024),
    (1152, 896),
    (896, 1152),
    (1216, 832),
    (832, 1216),
    (1344, 768),
    (768, 1344),
    (1440, 1440),
    (1248, 832),
    (832, 1248),
}


def empty_report(note: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "ai_generated_probability": None,
        "deepfake_probability": None,
        "generator": None,
        "detector": None,
        "provenance_credentials": False,
        "signals": [],
        "note": note,
    }


def _read_ends(path: str) -> bytes:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as fh:
            head = fh.read(_HEAD_BYTES)
            if size > _HEAD_BYTES + _TAIL_BYTES:
                fh.seek(max(0, size - _TAIL_BYTES))
                return head + fh.read(_TAIL_BYTES)
            return head + fh.read()
    except Exception:
        return b""


def _metadata_text(metadata: Optional[dict[str, Any]]) -> str:
    """Flatten EXIF values into one lowercase string for signature matching."""
    if not metadata:
        return ""
    parts: list[str] = []
    camera = metadata.get("camera") or {}
    for key in ("make", "model", "software", "lens"):
        if camera.get(key):
            parts.append(str(camera[key]))
    raw = metadata.get("raw") or {}
    for key, value in raw.items():
        if isinstance(value, (str, int, float)):
            parts.append(f"{key}={value}")
    return " ".join(parts).lower()


def detect_synthetic_metadata(
    image_path: str,
    metadata: Optional[dict[str, Any]] = None,
    image_format: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, Any]:
    """Look for declarations and traces of synthetic origin. Never raises."""
    report = empty_report("")
    signals: list[str] = []
    score = 0.0
    generator: Optional[str] = None

    blob = _read_ends(image_path)
    text = blob.decode("latin-1", errors="ignore").lower()
    meta_text = _metadata_text(metadata)
    haystack = f"{text} {meta_text}"

    if any(marker in text for marker in _C2PA_MARKERS):
        report["provenance_credentials"] = True
        signals.append(
            "Content Credentials (C2PA) provenance manifest present in the file."
        )

    for value in _IPTC_AI_VALUES:
        if value in haystack:
            score = max(score, 0.97)
            signals.append(
                "Metadata declares AI origin (IPTC DigitalSourceType "
                f"'{value}')."
            )
            break

    for needle, label in _GENERATOR_SIGNATURES:
        if needle in haystack:
            generator = label
            score = max(score, 0.9)
            signals.append(f"Generator signature found in metadata: {label}.")
            break

    if any(marker in text for marker in _PROMPT_CHUNK_MARKERS) or _PARAM_PATTERN.search(
        text
    ):
        score = max(score, 0.95)
        signals.append(
            "Embedded generation parameters (prompt/sampler/seed) found in the "
            "image's text metadata."
        )

    # Structural heuristics: only meaningful together, never alone conclusive.
    fmt = (image_format or "").upper()
    camera = (metadata or {}).get("camera") or {}
    has_camera = bool(camera.get("make") or camera.get("model"))
    if width and height and not has_camera:
        if (int(width), int(height)) in _GENERATOR_SIZES:
            score = max(score, 0.5)
            signals.append(
                f"{width}x{height} is a standard image-generator canvas size and "
                "the file has no camera metadata."
            )
        elif width == height and fmt in ("PNG", "WEBP"):
            score = max(score, 0.32)
            signals.append(
                "Perfectly square image with no camera metadata (weak indicator)."
            )

    report["signals"] = signals
    report["generator"] = generator
    report["detector"] = "metadata"
    report["ai_generated_probability"] = round(score, 3) if score > 0 else None

    if score >= SYNTHETIC_AI_THRESHOLD:
        report["status"] = "synthetic"
        report["note"] = (
            "This image declares or clearly shows signs of AI generation"
            + (f" ({generator})" if generator else "")
            + "; any location it depicts may not be real."
        )
    elif score >= 0.45:
        report["status"] = "likely_synthetic"
        report["note"] = (
            "This image has characteristics of AI-generated content, though "
            "nothing conclusive."
        )
    elif report["provenance_credentials"]:
        report["status"] = "authentic"
        report["note"] = (
            "Signed Content Credentials are present and contain no AI-generation "
            "assertion."
        )
    elif has_camera:
        report["status"] = "authentic"
        report["note"] = "Original camera metadata present; no synthetic markers."
    else:
        report["note"] = (
            "No synthetic markers found, but metadata is too sparse to confirm "
            "the image is a genuine photograph."
        )
    return report


_PIPELINES: dict[str, Any] = {}


def _get_pipeline(model_name: str):
    """Lazily build and memoise a transformers image-classification pipeline."""
    if model_name in _PIPELINES:
        return _PIPELINES[model_name]
    try:
        from transformers import pipeline  # type: ignore

        _PIPELINES[model_name] = pipeline("image-classification", model=model_name)
    except Exception:
        _PIPELINES[model_name] = None
    return _PIPELINES[model_name]


def _positive_score(predictions: Any, positive_words: tuple[str, ...]) -> Optional[float]:
    """Pull the probability of the 'bad' class out of a pipeline result."""
    if not isinstance(predictions, list):
        return None
    for pred in predictions:
        if not isinstance(pred, dict):
            continue
        label = str(pred.get("label", "")).lower()
        if any(word in label for word in positive_words):
            try:
                return max(0.0, min(1.0, float(pred.get("score", 0.0))))
            except (TypeError, ValueError):
                return None
    return None


def detect_with_model(image_path: str) -> dict[str, Any]:
    """Optional local-classifier pass (requires transformers + torch)."""
    result: dict[str, Any] = {
        "ai_generated_probability": None,
        "deepfake_probability": None,
        "available": False,
    }
    if not SYNTHETIC_MODEL_ENABLED:
        return result
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        ai_pipe = _get_pipeline(SYNTHETIC_AI_MODEL)
        if ai_pipe is not None:
            result["ai_generated_probability"] = _positive_score(
                ai_pipe(image), ("artificial", "ai", "fake", "generated")
            )
        deepfake_pipe = _get_pipeline(SYNTHETIC_DEEPFAKE_MODEL)
        if deepfake_pipe is not None:
            result["deepfake_probability"] = _positive_score(
                deepfake_pipe(image), ("fake", "deepfake", "manipulated")
            )
        result["available"] = (
            result["ai_generated_probability"] is not None
            or result["deepfake_probability"] is not None
        )
    except Exception:
        return result
    return result


def detect_with_api(image_path: str) -> dict[str, Any]:
    """Optional hosted-detector pass (`SYNTHETIC_API_URL`)."""
    result: dict[str, Any] = {
        "ai_generated_probability": None,
        "deepfake_probability": None,
        "generator": None,
        "available": False,
    }
    if not SYNTHETIC_API_URL:
        return result
    try:
        import requests

        headers = {}
        if SYNTHETIC_API_KEY:
            headers["Authorization"] = f"Bearer {SYNTHETIC_API_KEY}"
        with open(image_path, "rb") as fh:
            resp = requests.post(
                SYNTHETIC_API_URL,
                files={"image": fh},
                headers=headers,
                timeout=SYNTHETIC_API_TIMEOUT,
            )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return result

    for key in ("ai_generated_probability", "deepfake_probability"):
        try:
            if data.get(key) is not None:
                result[key] = max(0.0, min(1.0, float(data[key])))
        except (TypeError, ValueError):
            pass
    if data.get("generator"):
        result["generator"] = str(data["generator"])
    result["available"] = (
        result["ai_generated_probability"] is not None
        or result["deepfake_probability"] is not None
    )
    return result


def assess_synthetic(
    image_path: str,
    metadata: Optional[dict[str, Any]] = None,
    image_format: Optional[str] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> dict[str, Any]:
    """Full synthetic-media assessment: metadata plus any enabled classifier."""
    if not SYNTHETIC_ENABLED:
        return empty_report("Synthetic-media detection disabled.")

    report = detect_synthetic_metadata(
        image_path, metadata, image_format, width, height
    )

    extra = detect_with_api(image_path)
    if not extra.get("available"):
        extra = detect_with_model(image_path)
    if not extra.get("available"):
        return report

    detectors = [report.get("detector") or "metadata", "classifier"]
    report["detector"] = "+".join(detectors)

    ai_prob = extra.get("ai_generated_probability")
    if ai_prob is not None:
        report["signals"].append(f"Classifier AI-generated probability {ai_prob:.2f}.")
        # Keep whichever evidence is stronger; a declaration outranks a guess.
        current = report.get("ai_generated_probability") or 0.0
        report["ai_generated_probability"] = round(max(current, ai_prob), 3)

    deepfake_prob = extra.get("deepfake_probability")
    if deepfake_prob is not None:
        report["deepfake_probability"] = round(deepfake_prob, 3)
        report["signals"].append(f"Classifier deepfake probability {deepfake_prob:.2f}.")

    if extra.get("generator") and not report.get("generator"):
        report["generator"] = extra["generator"]

    final_ai = report.get("ai_generated_probability") or 0.0
    final_deepfake = report.get("deepfake_probability") or 0.0
    if final_ai >= SYNTHETIC_AI_THRESHOLD:
        report["status"] = "synthetic"
        report["note"] = (
            "Assessed as AI-generated"
            + (f" ({report['generator']})" if report.get("generator") else "")
            + "; any location it depicts may not be real."
        )
    elif final_deepfake >= SYNTHETIC_DEEPFAKE_THRESHOLD:
        report["status"] = "manipulated"
        report["note"] = (
            "Assessed as manipulated (face swap or edited region); treat any "
            "location evidence with caution."
        )
    elif final_ai >= 0.45:
        report["status"] = "likely_synthetic"
        report["note"] = "Possible AI-generated content; nothing conclusive."
    return report


def is_unreliable(report: Optional[dict[str, Any]]) -> bool:
    """True when the image should not be trusted to depict a real place."""
    return bool(report) and report.get("status") in (
        "synthetic",
        "likely_synthetic",
        "manipulated",
    )
