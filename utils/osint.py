"""Fuse EXIF and vision evidence into a single likely location.

Trust order (highest first):
  1. EXIF GPS coordinates                 -> authoritative (reverse-geocoded).
  2. Vision candidate WITH coordinates    -> use directly.
  3. Vision candidate name -> forward-geocoded to coordinates.
  4. Vision candidate name only (geocode failed).
  5. EXIF caption (ImageDescription) -> optionally geocoded.
  6. Unknown.

Confidence is boosted when independent clue types (landmarks, OCR text, signage,
flags, languages) corroborate the guess, and when a name successfully resolves to
real coordinates via geocoding.
"""

from __future__ import annotations

from typing import Any, Optional

from utils.geocode import forward_geocode, reverse_geocode
from utils.verify import detect_expected_features, verify_location


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _has_coords(d: Optional[dict[str, Any]]) -> bool:
    return bool(d) and d.get("latitude") is not None and d.get("longitude") is not None


def _clue_bonus(vision: Optional[dict[str, Any]]) -> float:
    """Extra confidence from independent corroborating clue categories."""
    if not vision:
        return 0.0
    bonus = 0.0
    weights = {
        "landmarks": 0.10,
        "ocr_text": 0.06,
        "signage": 0.05,
        "flags": 0.05,
        "languages": 0.03,
        "vehicles_plates": 0.03,
    }
    for key, weight in weights.items():
        if vision.get(key):
            bonus += weight
    return min(bonus, 0.25)


def _caption(metadata: Optional[dict[str, Any]]) -> Optional[str]:
    raw = metadata.get("raw", {}) if metadata else {}
    for key in ("ImageDescription", "XPTitle", "XPSubject", "UserComment"):
        value = raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _candidates(vision: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not vision:
        return []
    cands = vision.get("candidates") or []
    if cands:
        return cands
    # Fall back to the legacy single best guess.
    guess = vision.get("best_guess_location")
    if guess and (guess.get("name") or _has_coords(guess)):
        return [{**guess, "confidence": vision.get("confidence", 0.0)}]
    return []


def determine_location(
    metadata: dict[str, Any], vision: dict[str, Any]
) -> dict[str, Any]:
    """Fuse metadata + vision clues into a best-estimate location."""
    evidence: list[str] = []
    result: dict[str, Any] = {
        "location_name": None,
        "country": None,
        "region": None,
        "address": None,
        "latitude": None,
        "longitude": None,
        "confidence": 0.0,
        "source": "unknown",
        "alternatives": [],
        "evidence": evidence,
    }

    gps = metadata.get("gps") if metadata else None
    bonus = _clue_bonus(vision)

    # 1. Authoritative EXIF GPS.
    if _has_coords(gps):
        result.update(
            {
                "latitude": gps["latitude"],
                "longitude": gps["longitude"],
                "confidence": 0.99,
                "source": "exif_gps",
            }
        )
        evidence.append(
            f"EXIF GPS coordinates {gps['latitude']}, {gps['longitude']}."
        )
        rev = reverse_geocode(gps["latitude"], gps["longitude"])
        if rev:
            result["location_name"] = rev.get("name")
            result["country"] = rev.get("country")
            result["region"] = rev.get("region")
            result["address"] = rev.get("display_name")
            evidence.append(f"Reverse-geocoded to {rev.get('display_name')}.")
        _finalize(result, metadata, vision)
        return result

    # 2-4. Work through ranked vision candidates.
    candidates = _candidates(vision)
    alternatives: list[dict[str, Any]] = []
    chosen: Optional[dict[str, Any]] = None
    chosen_source: Optional[str] = None

    for idx, cand in enumerate(candidates):
        name = cand.get("name")
        cconf = _clamp(float(cand.get("confidence", 0.0) or 0.0))

        if _has_coords(cand):
            resolved = {
                "name": name,
                "country": cand.get("country"),
                "region": cand.get("region"),
                "latitude": cand["latitude"],
                "longitude": cand["longitude"],
                "confidence": _clamp(cconf + bonus),
                "source": "vision_coordinates",
            }
        elif name:
            geo = forward_geocode(
                ", ".join(
                    p for p in (name, cand.get("region"), cand.get("country")) if p
                )
            )
            if geo:
                resolved = {
                    "name": name or geo.get("name"),
                    "country": geo.get("country") or cand.get("country"),
                    "region": geo.get("region") or cand.get("region"),
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "address": geo.get("display_name"),
                    "confidence": _clamp(cconf * 0.9 + bonus),
                    "source": "vision_geocoded",
                }
            else:
                resolved = {
                    "name": name,
                    "country": cand.get("country"),
                    "region": cand.get("region"),
                    "latitude": None,
                    "longitude": None,
                    "confidence": _clamp(cconf * 0.7 + bonus * 0.5),
                    "source": "vision_place_name",
                }
        else:
            continue

        if chosen is None:
            chosen = resolved
            chosen_source = resolved["source"]
        else:
            alternatives.append(
                {
                    "name": resolved.get("name"),
                    "country": resolved.get("country"),
                    "latitude": resolved.get("latitude"),
                    "longitude": resolved.get("longitude"),
                    "confidence": round(resolved.get("confidence", 0.0), 3),
                }
            )

    if chosen is not None:
        result.update(
            {
                "location_name": chosen.get("name"),
                "country": chosen.get("country"),
                "region": chosen.get("region"),
                "address": chosen.get("address"),
                "latitude": chosen.get("latitude"),
                "longitude": chosen.get("longitude"),
                "confidence": chosen.get("confidence", 0.0),
                "source": chosen_source or "vision_place_name",
            }
        )
        result["alternatives"] = alternatives
        if chosen.get("name"):
            evidence.append(f"Best match: {chosen.get('name')}.")
        # Enrich a coordinate-only match with a readable address.
        if _has_coords(chosen) and not chosen.get("address"):
            rev = reverse_geocode(chosen["latitude"], chosen["longitude"])
            if rev:
                result["address"] = rev.get("display_name")
                result["country"] = result["country"] or rev.get("country")
                result["region"] = result["region"] or rev.get("region")
        _finalize(result, metadata, vision)
        return result

    # 5. EXIF caption fallback (try to geocode it too).
    caption = _caption(metadata)
    if caption:
        geo = forward_geocode(caption)
        if geo:
            result.update(
                {
                    "location_name": geo.get("name") or caption,
                    "country": geo.get("country"),
                    "region": geo.get("region"),
                    "address": geo.get("display_name"),
                    "latitude": geo["latitude"],
                    "longitude": geo["longitude"],
                    "confidence": 0.5,
                    "source": "exif_caption",
                }
            )
        else:
            result.update(
                {
                    "location_name": caption,
                    "confidence": 0.4,
                    "source": "exif_caption",
                }
            )
        evidence.append(f"EXIF caption: {caption}.")

    _finalize(result, metadata, vision)

    if result["source"] == "unknown":
        evidence.append("No GPS metadata and no confident visual location match.")

    return result


def _finalize(
    result: dict[str, Any],
    metadata: Optional[dict[str, Any]],
    vision: Optional[dict[str, Any]],
) -> None:
    """Append clue evidence and cross-check the location against the scene."""
    _append_clue_evidence(result, metadata, vision)
    _verify_and_adjust(result, vision)


def _verify_and_adjust(
    result: dict[str, Any], vision: Optional[dict[str, Any]]
) -> None:
    """Confirm expected geographic features exist near the resolved point.

    Lowers confidence and adds a warning when the scene (e.g. a lake) does not
    match what is actually at the coordinates. EXIF GPS is authoritative, so it
    is checked for information only and never penalised heavily.
    """
    lat = result.get("latitude")
    lon = result.get("longitude")
    if lat is None or lon is None:
        return

    expected = detect_expected_features(vision)
    if not expected:
        return

    report = verify_location(lat, lon, expected)
    result["verification"] = report
    evidence: list[str] = result["evidence"]
    status = report.get("status")
    nearest = report.get("nearest_m", {})

    if status == "skipped":
        return

    for cat in report.get("confirmed", []):
        dist = nearest.get(cat) or nearest.get("coast" if cat == "water" else cat)
        if dist is not None:
            evidence.append(f"Verified: {cat} found ~{int(dist)} m away.")
        else:
            evidence.append(f"Verified: {cat} present nearby.")
    for cat in report.get("missing", []):
        evidence.append(
            f"Warning: image suggests {cat}, but none found within "
            f"{report.get('radius_m')} m of this location."
        )

    is_gps = result.get("source") == "exif_gps"
    factor = {
        "verified": 1.0,
        "partial": 0.7,
        "mismatch": 0.4,
    }.get(status, 1.0)

    if status == "mismatch":
        result["warning"] = (
            "The described scene does not match this location's surroundings; "
            "the result may be unreliable."
        )
    elif status == "partial":
        result["warning"] = (
            "Some described features could not be confirmed near this location."
        )

    if not is_gps:
        result["confidence"] = _clamp(result.get("confidence", 0.0) * factor)


def _append_clue_evidence(
    result: dict[str, Any],
    metadata: Optional[dict[str, Any]],
    vision: Optional[dict[str, Any]],
) -> None:
    """Add supporting clues to the evidence list (deduplicated, capped)."""
    evidence: list[str] = result["evidence"]

    if vision:
        for landmark in (vision.get("landmarks") or [])[:5]:
            evidence.append(f"Landmark detected: {landmark}.")
        for flag in (vision.get("flags") or [])[:3]:
            evidence.append(f"Flag/emblem: {flag}.")
        for text in (vision.get("signage") or [])[:5]:
            evidence.append(f"Signage: {text}.")
        for text in (vision.get("ocr_text") or [])[:5]:
            evidence.append(f"Text read: {text}.")
        for lang in (vision.get("languages") or [])[:3]:
            evidence.append(f"Language seen: {lang}.")
        if vision.get("road_side") and vision["road_side"] != "unknown":
            evidence.append(f"Traffic drives on the {vision['road_side']}.")
        if vision.get("architecture"):
            evidence.append(f"Architecture: {vision['architecture']}.")
        if vision.get("vegetation"):
            evidence.append(f"Vegetation: {vision['vegetation']}.")
        if vision.get("climate"):
            evidence.append(f"Climate: {vision['climate']}.")
        if vision.get("terrain"):
            evidence.append(f"Terrain: {vision['terrain']}.")
        if vision.get("hemisphere_hint"):
            evidence.append(f"Hemisphere hint: {vision['hemisphere_hint']}.")
        if vision.get("environment"):
            evidence.append(f"Environment: {vision['environment']}.")
        if vision.get("reasoning"):
            evidence.append(f"Reasoning: {vision['reasoning']}.")

    if metadata:
        if metadata.get("timestamp"):
            evidence.append(f"Photo timestamp: {metadata['timestamp']}.")
        cam = metadata.get("camera", {}) or {}
        if cam.get("model"):
            label = f"{cam.get('make') or ''} {cam.get('model')}".strip()
            evidence.append(f"Captured with {label}.")
