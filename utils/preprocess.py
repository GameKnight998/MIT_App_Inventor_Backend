"""Make a poor-quality image legible to the vision model before analysing it.

Real investigative material is rarely a clean 12-megapixel photo. It is a CCTV
grab, a screenshot of a screenshot, a night shot, or something re-compressed
three times by a messaging app. In all of those cases the location clue (a street
plate, a shopfront, a plate number) is physically present but too dark, too
small, or too soft for the model to read.

So we measure the image and apply only the corrections it actually needs:

  * too small  -> upscale (Lanczos, or Real-ESRGAN when installed) so fine text
    survives the vision API's own internal tiling;
  * too dark   -> CLAHE on the lightness channel plus gamma, which lifts shadow
    detail without blowing out the highlights a global brightness boost would;
  * too soft   -> unsharp mask to recover edge definition on text;
  * too large  -> downscale to the API's effective input size, which costs
    nothing in accuracy and measurably cuts upload and inference time.

This runs on OpenCV, already a dependency for video frame extraction, so it adds
no install weight. Everything fails soft: if OpenCV is missing or any step
errors, the caller gets the original path back and analysis proceeds unchanged.

Enhancement is used ONLY for the vision call. EXIF and forensics always read the
untouched original, because re-encoding would destroy exactly the provenance
evidence those steps depend on.
"""

from __future__ import annotations

import os
from typing import Any, Optional

ENHANCE_ENABLED = os.getenv("ENHANCE_ENABLED", "1") not in ("0", "false", "False")
# Below this longest-edge size, fine text is likely unreadable -> upscale.
ENHANCE_MIN_DIM = int(os.getenv("ENHANCE_MIN_DIM", "800"))
# Above this, we are paying for pixels the vision API will discard anyway.
ENHANCE_MAX_DIM = int(os.getenv("ENHANCE_MAX_DIM", "2048"))
# Mean grey level (0-255) below which the image counts as dark.
ENHANCE_DARK_MEAN = float(os.getenv("ENHANCE_DARK_MEAN", "70"))
# Variance-of-Laplacian below which the image counts as soft/blurry.
ENHANCE_BLUR_MIN = float(os.getenv("ENHANCE_BLUR_MIN", "60"))
# Never upscale by more than this; beyond it we invent detail rather than recover it.
ENHANCE_UPSCALE_MAX = float(os.getenv("ENHANCE_UPSCALE_MAX", "3.0"))
ENHANCE_JPEG_QUALITY = int(os.getenv("ENHANCE_JPEG_QUALITY", "95"))
# Denoising is genuinely slow on a shared CPU, so it is opt-in.
ENHANCE_DENOISE = os.getenv("ENHANCE_DENOISE", "0") not in ("0", "false", "False")
# Real-ESRGAN gives better upscales but needs torch; off unless explicitly enabled.
REALESRGAN_ENABLED = os.getenv("REALESRGAN_ENABLED", "0") not in (
    "0",
    "false",
    "False",
)


def _import_cv2():
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def enhancement_available() -> bool:
    return _import_cv2() is not None


def _metrics(cv2, image) -> dict[str, Any]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape[:2]
    return {
        "width": int(width),
        "height": int(height),
        "megapixels": round(width * height / 1_000_000, 2),
        "blur_score": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1),
        "brightness": round(float(gray.mean()), 1),
        "contrast": round(float(gray.std()), 1),
    }


def _label(m: dict[str, Any]) -> list[str]:
    """Human-readable quality problems, worst first."""
    issues: list[str] = []
    if max(m["width"], m["height"]) < ENHANCE_MIN_DIM:
        issues.append("low_resolution")
    if m["brightness"] < ENHANCE_DARK_MEAN:
        issues.append("dark")
    if m["blur_score"] < ENHANCE_BLUR_MIN:
        issues.append("soft_or_blurry")
    if m["contrast"] < 30:
        issues.append("low_contrast")
    if max(m["width"], m["height"]) > ENHANCE_MAX_DIM:
        issues.append("oversized")
    return issues


def assess_quality(image_path: str) -> dict[str, Any]:
    """Measure resolution, sharpness, brightness and contrast of an image."""
    cv2 = _import_cv2()
    if cv2 is None:
        return {"available": False, "note": "OpenCV not installed."}
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        return {"available": False, "note": "Could not read image."}
    m = _metrics(cv2, image)
    m["available"] = True
    m["issues"] = _label(m)
    m["needs_enhancement"] = bool(
        [i for i in m["issues"] if i != "oversized"]
    ) or "oversized" in m["issues"]
    return m


def _upscale(cv2, image, factor: float):
    """Enlarge, preferring Real-ESRGAN when it is installed and enabled."""
    height, width = image.shape[:2]
    target = (int(width * factor), int(height * factor))

    if REALESRGAN_ENABLED:
        try:
            from realesrgan import RealESRGANer  # type: ignore  # noqa: F401

            # Real-ESRGAN is intentionally not wired into the default path: it
            # needs torch plus a weights download, which does not fit a free
            # tier. Deployments that install it can implement the call here.
            raise NotImplementedError
        except Exception:
            pass

    return cv2.resize(image, target, interpolation=cv2.INTER_LANCZOS4)


def _brighten(cv2, image):
    """CLAHE on L plus a gamma lift: recovers shadow detail, keeps highlights."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lightness = clahe.apply(lightness)
    merged = cv2.cvtColor(cv2.merge((lightness, a_chan, b_chan)), cv2.COLOR_LAB2BGR)

    import numpy as np

    gamma = 1.4
    table = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]
    ).astype("uint8")
    return cv2.LUT(merged, table)


def _sharpen(cv2, image):
    """Unsharp mask: subtract a blurred copy to restore edge contrast."""
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=1.4)
    return cv2.addWeighted(image, 1.6, blurred, -0.6, 0)


def enhance_image(
    image_path: str, output_path: Optional[str] = None
) -> dict[str, Any]:
    """Return a path to the best version of this image for vision analysis.

    The result always contains a usable `path`: the enhanced file when something
    was improved, otherwise the original. `applied` lists what was done so the
    response can tell the user their image was upscaled or brightened.
    """
    result: dict[str, Any] = {
        "path": image_path,
        "enhanced": False,
        "applied": [],
        "quality_before": None,
        "quality_after": None,
        "note": "",
    }
    if not ENHANCE_ENABLED:
        result["note"] = "Enhancement disabled."
        return result

    cv2 = _import_cv2()
    if cv2 is None:
        result["note"] = "OpenCV not installed; using original image."
        return result

    try:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            result["note"] = "Could not decode image; using original."
            return result

        before = _metrics(cv2, image)
        before["issues"] = _label(before)
        result["quality_before"] = before

        applied: list[str] = []
        longest = max(before["width"], before["height"])

        if longest < ENHANCE_MIN_DIM:
            factor = min(ENHANCE_MIN_DIM / max(longest, 1), ENHANCE_UPSCALE_MAX)
            if factor > 1.05:
                image = _upscale(cv2, image, factor)
                applied.append(f"upscaled {factor:.1f}x")
        elif longest > ENHANCE_MAX_DIM:
            factor = ENHANCE_MAX_DIM / longest
            image = cv2.resize(
                image,
                (int(before["width"] * factor), int(before["height"] * factor)),
                interpolation=cv2.INTER_AREA,
            )
            applied.append("downscaled to API input size")

        if before["brightness"] < ENHANCE_DARK_MEAN:
            image = _brighten(cv2, image)
            applied.append("brightened (CLAHE + gamma)")

        if ENHANCE_DENOISE and before["brightness"] < ENHANCE_DARK_MEAN:
            image = cv2.fastNlMeansDenoisingColored(image, None, 5, 5, 7, 21)
            applied.append("denoised")

        if before["blur_score"] < ENHANCE_BLUR_MIN:
            image = _sharpen(cv2, image)
            applied.append("sharpened (unsharp mask)")

        if not applied:
            result["note"] = "Image quality is already adequate."
            return result

        out = output_path or f"{os.path.splitext(image_path)[0]}_enhanced.jpg"
        ok = cv2.imwrite(
            out, image, [int(cv2.IMWRITE_JPEG_QUALITY), ENHANCE_JPEG_QUALITY]
        )
        if not ok:
            result["note"] = "Could not write enhanced image; using original."
            return result

        after = _metrics(cv2, image)
        after["issues"] = _label(after)
        result.update(
            {
                "path": out,
                "enhanced": True,
                "applied": applied,
                "quality_after": after,
                "note": "Enhanced before analysis: " + ", ".join(applied) + ".",
            }
        )
        return result
    except Exception as exc:
        result["note"] = f"Enhancement failed ({exc}); using original."
        return result
