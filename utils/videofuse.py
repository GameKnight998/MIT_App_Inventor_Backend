"""Fuse per-frame vision analyses into one combined "video" analysis.

The whole point of video is that clues are spread across time: a sign is legible
in frame 2, a mountain is revealed in frame 4, and three frames independently
point at the same region. This module captures exactly that value:

  * textual/visual clues (OCR, signage, landmarks, flags, languages, plates,
    scene features) are UNIONED across frames -- a clue seen in any frame counts;
  * descriptive scalars (vegetation, climate, terrain, ...) are taken from the
    most confident frame that reported them;
  * candidate regions are grouped by name and boosted when MULTIPLE frames agree
    (cross-frame agreement is the strongest signal a video adds).

The result has the exact shape `analyze_image` produces, so the existing OSINT
pipeline (verification, climate, solar, log-linear fusion) consumes it unchanged.
"""

from __future__ import annotations

from typing import Any, Optional

from utils.vision import _empty_analysis

# Confidence added per ADDITIONAL frame that agrees on the same region. Capped
# so agreement helps but can never fabricate certainty on its own.
AGREEMENT_BONUS = 0.08
AGREEMENT_BONUS_CAP = 0.24

_LIST_UNION_KEYS = (
    "ocr_text",
    "languages",
    "landmarks",
    "flags",
    "signage",
    "vehicles_plates",
    "architecture_regions",
)
_SCALAR_KEYS = (
    "architecture",
    "architecture_style",
    "vegetation",
    "climate",
    "terrain",
    "road_side",
    "time_of_day",
    "hemisphere_hint",
    "environment",
)


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _frame_conf(vision: dict[str, Any]) -> float:
    return _clamp(vision.get("confidence", 0.0) or 0.0)


def _union_list(frames: list[dict[str, Any]], key: str) -> list[Any]:
    """Case-insensitive de-duplicated union of a list field across frames."""
    seen: set[str] = set()
    out: list[Any] = []
    for fr in frames:
        for item in fr.get(key) or []:
            norm = str(item).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(item)
    return out


def _merge_scene(frames: list[dict[str, Any]]) -> dict[str, bool]:
    """A scene feature is present if ANY frame observed it (verification will
    still map-check the ones that matter, so a lone false positive is caught)."""
    scene: dict[str, bool] = {}
    for fr in frames:
        for key, val in (fr.get("scene") or {}).items():
            scene[key] = bool(scene.get(key)) or bool(val)
    return scene


def _merge_text_analysis(frames: list[dict[str, Any]]) -> dict[str, Any]:
    scripts: list[str] = []
    spelling: set[str] = set()
    countries: set[str] = set()
    for fr in frames:
        ta = fr.get("text_analysis") or {}
        if ta.get("primary_script"):
            scripts.append(ta["primary_script"])
        for s in ta.get("regional_spelling") or []:
            spelling.add(s)
        for c in ta.get("implied_countries") or []:
            countries.add(c)
    primary_script = max(set(scripts), key=scripts.count) if scripts else None
    return {
        "primary_script": primary_script,
        "regional_spelling": sorted(spelling),
        "implied_countries": sorted(countries),
    }


def _merge_sun(frames: list[dict[str, Any]]) -> dict[str, Any]:
    """Prefer the sun observation from the most confident frame that saw sun or
    shadows; otherwise keep the default empty block."""
    best: Optional[dict[str, Any]] = None
    best_conf = -1.0
    for fr in frames:
        sun = fr.get("sun") or {}
        if (sun.get("sun_visible") or sun.get("shadows_visible")) and _frame_conf(
            fr
        ) > best_conf:
            best = sun
            best_conf = _frame_conf(fr)
    return best or {
        "sun_visible": False,
        "shadows_visible": False,
        "shadow_direction": None,
        "shadow_length": None,
        "approx_solar_elevation": None,
    }


def _pick_scalar(frames: list[dict[str, Any]], key: str) -> Optional[Any]:
    """Value from the most confident frame that reported this field."""
    best_val = None
    best_conf = -1.0
    for fr in frames:
        val = fr.get(key)
        if val and val != "unknown" and _frame_conf(fr) > best_conf:
            best_val = val
            best_conf = _frame_conf(fr)
    return best_val


def _merge_candidates(frames: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group candidates by name; agreement across frames boosts confidence."""
    groups: dict[str, dict[str, Any]] = {}
    for fr in frames:
        for cand in fr.get("candidates") or []:
            name = (cand.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            conf = _clamp(cand.get("confidence", 0.0) or 0.0)
            g = groups.get(key)
            if g is None:
                groups[key] = {
                    "name": name,
                    "region": cand.get("region"),
                    "country": cand.get("country"),
                    "latitude": cand.get("latitude"),
                    "longitude": cand.get("longitude"),
                    "why": cand.get("why"),
                    "_max_conf": conf,
                    "_frames": 1,
                    "_lat_sum": cand.get("latitude") or 0.0,
                    "_lon_sum": cand.get("longitude") or 0.0,
                    "_coord_n": 1 if cand.get("latitude") is not None else 0,
                }
            else:
                g["_max_conf"] = max(g["_max_conf"], conf)
                g["_frames"] += 1
                if cand.get("latitude") is not None:
                    g["_lat_sum"] += cand["latitude"]
                    g["_lon_sum"] += cand.get("longitude") or 0.0
                    g["_coord_n"] += 1
                if cand.get("latitude") is not None and g.get("latitude") is None:
                    g["latitude"] = cand["latitude"]
                    g["longitude"] = cand.get("longitude")
                g["region"] = g["region"] or cand.get("region")
                g["country"] = g["country"] or cand.get("country")

    merged: list[dict[str, Any]] = []
    for g in groups.values():
        agree = min(AGREEMENT_BONUS * (g["_frames"] - 1), AGREEMENT_BONUS_CAP)
        conf = _clamp(g["_max_conf"] + agree)
        lat = g.get("latitude")
        lon = g.get("longitude")
        if g["_coord_n"] > 1:  # average agreeing coordinates
            lat = round(g["_lat_sum"] / g["_coord_n"], 6)
            lon = round(g["_lon_sum"] / g["_coord_n"], 6)
        merged.append(
            {
                "name": g["name"],
                "region": g["region"],
                "country": g["country"],
                "latitude": lat,
                "longitude": lon,
                "confidence": conf,
                "why": g["why"],
                "frames_agreeing": g["_frames"],
            }
        )
    merged.sort(key=lambda c: c["confidence"], reverse=True)
    return merged


def merge_frame_visions(frame_visions: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine several `analyze_image` outputs into one video-level analysis."""
    usable = [v for v in frame_visions if v and v.get("available")]
    if not usable:
        note = "No frame produced a usable vision analysis."
        if frame_visions:
            note = frame_visions[0].get("note") or note
        result = _empty_analysis(note)
        return result

    result = _empty_analysis("ok")
    result["available"] = True

    for key in _LIST_UNION_KEYS:
        result[key] = _union_list(usable, key)
    for key in _SCALAR_KEYS:
        val = _pick_scalar(usable, key)
        if val is not None:
            result[key] = val

    result["scene"] = _merge_scene(usable)
    result["text_analysis"] = _merge_text_analysis(usable)
    result["sun"] = _merge_sun(usable)

    candidates = _merge_candidates(usable)
    result["candidates"] = candidates
    if candidates:
        top = candidates[0]
        result["best_guess_location"] = {
            "name": top.get("name"),
            "country": top.get("country"),
            "latitude": top.get("latitude"),
            "longitude": top.get("longitude"),
        }
        result["confidence"] = top["confidence"]

    reasons = [v.get("reasoning") for v in usable if v.get("reasoning")]
    if reasons:
        result["reasoning"] = (
            "Combined from "
            f"{len(usable)} video frame(s). "
            + max(reasons, key=len)
        )
    return result
