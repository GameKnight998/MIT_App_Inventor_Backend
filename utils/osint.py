"""Fuse EXIF and vision evidence into a single likely location.

Trust order (highest first):
  1. EXIF GPS coordinates                 -> authoritative (reverse-geocoded).
  2. Vision candidates, resolved to coordinates and VERIFIED against the map.
  3. Vision candidate name only (geocode failed).
  4. EXIF caption (ImageDescription) -> optionally geocoded.
  5. Unknown.

Iterative verification (step 2): instead of trusting the top guess, every
candidate location is checked against OpenStreetMap. If the scene's features
(e.g. a lake) are NOT near a candidate, that location is "crossed out" and the
next candidate -- or the next real-world match for an ambiguous name -- is tried,
repeating until a location whose surroundings match the photo is found (or the
attempt budget is exhausted). Confidence is boosted by corroborating clues and
reduced when the final choice still doesn't fully match.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from utils.enrich import enrich_location
from utils.geocode import (
    forward_geocode,
    forward_geocode_candidates,
    reverse_geocode,
)
from utils.verify import detect_expected_features, verify_location

MAX_VERIFY_ATTEMPTS = int(os.getenv("MAX_VERIFY_ATTEMPTS", "3"))
GEO_CANDIDATES_PER_NAME = int(os.getenv("GEO_CANDIDATES_PER_NAME", "3"))

_STATUS_FACTOR = {
    "verified": 1.0,
    "skipped": 1.0,
    "unavailable": 1.0,  # map service unreachable -> can't verify, don't penalise
    "partial": 0.7,
    "mismatch": 0.4,
}
# Statuses that end the retry loop (no point retrying when we can't check).
_ACCEPT_STATUSES = ("verified", "skipped", "unavailable")


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


def _ambiguity_factor(vision: Optional[dict[str, Any]]) -> tuple[float, float]:
    """Temper confidence when several candidate regions are similarly likely.

    A generic scene (e.g. pine forest by a road) often yields multiple regions
    with near-equal confidence; the model then picks one somewhat arbitrarily.
    When the top two candidate confidences are close, the specific region is
    uncertain, so we scale confidence down. Returns (factor, gap).

    gap 0.00 -> factor 0.70   (a near tie)
    gap 0.20+ -> factor 1.00   (a clear favourite)
    """
    cands = _candidates(vision)
    confs = sorted(
        (_clamp(float(c.get("confidence", 0.0) or 0.0)) for c in cands), reverse=True
    )
    if len(confs) < 2:
        return 1.0, 1.0
    gap = confs[0] - confs[1]
    return _clamp(0.7 + gap * 1.5), gap


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
    guess = vision.get("best_guess_location")
    if guess and (guess.get("name") or _has_coords(guess)):
        return [{**guess, "confidence": vision.get("confidence", 0.0)}]
    return []


def _alt(hyp: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": hyp.get("name"),
        "country": hyp.get("country"),
        "latitude": hyp.get("latitude"),
        "longitude": hyp.get("longitude"),
        "confidence": round(hyp.get("base", 0.0), 3),
    }


def _hypotheses_for_candidate(
    cand: dict[str, Any], bonus: float
) -> list[dict[str, Any]]:
    """All coordinate options to try for one vision candidate, best first."""
    cconf = _clamp(float(cand.get("confidence", 0.0) or 0.0))
    name = cand.get("name")
    options: list[dict[str, Any]] = []

    if _has_coords(cand):
        options.append(
            {
                "name": name,
                "country": cand.get("country"),
                "region": cand.get("region"),
                "latitude": cand["latitude"],
                "longitude": cand["longitude"],
                "address": None,
                "base": _clamp(cconf + bonus),
                "source": "vision_coordinates",
            }
        )

    if name:
        query = ", ".join(
            p for p in (name, cand.get("region"), cand.get("country")) if p
        )
        for g in forward_geocode_candidates(query, GEO_CANDIDATES_PER_NAME):
            options.append(
                {
                    "name": name or g.get("name"),
                    "country": g.get("country") or cand.get("country"),
                    "region": g.get("region") or cand.get("region"),
                    "latitude": g["latitude"],
                    "longitude": g["longitude"],
                    "address": g.get("display_name"),
                    "base": _clamp(cconf * 0.9 + bonus),
                    "source": "vision_geocoded",
                }
            )
    return options


def _select_location(
    candidates: list[dict[str, Any]], bonus: float, expected: list[str]
) -> dict[str, Any]:
    """Try candidate locations, crossing out those the map contradicts.

    Returns a dict with the chosen hypothesis (or None), its verification
    report, the list of rejected locations, untried alternatives, a name-only
    fallback, and the number of verification attempts made.
    """
    attempts = 0
    seen: set[tuple[float, float]] = set()
    rejected: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    name_only: Optional[dict[str, Any]] = None
    best_partial: Optional[tuple[dict[str, Any], dict[str, Any]]] = None
    best_mismatch: Optional[tuple[dict[str, Any], dict[str, Any]]] = None
    chosen: Optional[dict[str, Any]] = None
    chosen_report: Optional[dict[str, Any]] = None

    for cand in candidates:
        cconf = _clamp(float(cand.get("confidence", 0.0) or 0.0))
        name = cand.get("name")
        if name and name_only is None:
            name_only = {
                "name": name,
                "country": cand.get("country"),
                "region": cand.get("region"),
                "confidence": _clamp(cconf * 0.7 + bonus * 0.5),
                "source": "vision_place_name",
            }

        for hyp in _hypotheses_for_candidate(cand, bonus):
            key = (round(hyp["latitude"], 3), round(hyp["longitude"], 3))
            if key in seen:
                continue
            seen.add(key)

            if attempts >= MAX_VERIFY_ATTEMPTS:
                alternatives.append(_alt(hyp))
                continue

            report = verify_location(hyp["latitude"], hyp["longitude"], expected)
            attempts += 1
            hyp["verification"] = report
            status = report.get("status")

            if status in _ACCEPT_STATUSES:
                chosen, chosen_report = hyp, report
                break
            if status == "partial":
                if best_partial is None or report.get("match_score", 0) > best_partial[
                    1
                ].get("match_score", 0):
                    best_partial = (hyp, report)
                alternatives.append(_alt(hyp))
            else:  # mismatch -> cross it out
                rejected.append(
                    {
                        "name": hyp.get("name"),
                        "latitude": hyp.get("latitude"),
                        "longitude": hyp.get("longitude"),
                        "reason": (
                            f"expected {report.get('missing')} not found within "
                            f"{report.get('radius_m')} m"
                        ),
                    }
                )
                if best_mismatch is None or report.get("match_score", 0) > best_mismatch[
                    1
                ].get("match_score", 0):
                    best_mismatch = (hyp, report)

        if chosen is not None:
            break

    if chosen is None:
        if best_partial is not None:
            chosen, chosen_report = best_partial
        elif best_mismatch is not None:
            chosen, chosen_report = best_mismatch

    return {
        "chosen": chosen,
        "report": chosen_report,
        "rejected": rejected,
        "alternatives": alternatives,
        "name_only": name_only,
        "attempts": attempts,
    }


def determine_location(
    metadata: dict[str, Any], vision: dict[str, Any]
) -> dict[str, Any]:
    """Fuse metadata + vision clues into a best-estimate, map-verified location."""
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
        "rejected": [],
        "nearby_places": [],
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
        evidence.append(f"EXIF GPS coordinates {gps['latitude']}, {gps['longitude']}.")
        rev = reverse_geocode(gps["latitude"], gps["longitude"])
        if rev:
            result["location_name"] = rev.get("name")
            result["country"] = rev.get("country")
            result["region"] = rev.get("region")
            result["address"] = rev.get("display_name")
            evidence.append(f"Reverse-geocoded to {rev.get('display_name')}.")
        _append_clue_evidence(result, metadata, vision)
        _verify_and_adjust(result, vision)  # info only; GPS is never penalised
        _enrich(result, vision)
        return result

    # 2. Vision candidates with iterative, map-checked verification.
    expected = detect_expected_features(vision)
    selection = _select_location(_candidates(vision), bonus, expected)
    chosen = selection["chosen"]
    report = selection["report"]

    if chosen is not None:
        status = (report or {}).get("status", "skipped")
        factor = _STATUS_FACTOR.get(status, 1.0)
        amb_factor, gap = _ambiguity_factor(vision)
        result.update(
            {
                "location_name": chosen.get("name"),
                "country": chosen.get("country"),
                "region": chosen.get("region"),
                "address": chosen.get("address"),
                "latitude": chosen.get("latitude"),
                "longitude": chosen.get("longitude"),
                "confidence": _clamp(chosen.get("base", 0.0) * factor * amb_factor),
                "source": chosen.get("source", "vision_geocoded"),
            }
        )
        result["alternatives"] = _merge_alternatives(
            vision, chosen.get("name"), selection["alternatives"]
        )
        result["rejected"] = selection["rejected"][:5]
        if report:
            result["verification"] = report

        if _has_coords(chosen) and not chosen.get("address"):
            rev = reverse_geocode(chosen["latitude"], chosen["longitude"])
            if rev:
                result["address"] = rev.get("display_name")
                result["country"] = result["country"] or rev.get("country")
                result["region"] = result["region"] or rev.get("region")

        _emit_retry_evidence(result, selection)
        _emit_verification_evidence(result, report)
        _set_warning(result, status)
        if amb_factor < 1.0:
            evidence.append(
                "Several regions look similar in this photo, so the specific "
                    "region is uncertain (see alternatives)."
                )
            if not result.get("warning"):
                result["warning"] = (
                    "This scene lacks distinctive features; multiple regions are "
                    "plausible and the exact one is uncertain."
                )
        _append_clue_evidence(result, metadata, vision)
        _enrich(result, vision)
        return result

    # 3. Name-only fallback (geocoding produced no coordinates to verify).
    name_only = selection["name_only"]
    if name_only is not None:
        result.update(
            {
                "location_name": name_only.get("name"),
                "country": name_only.get("country"),
                "region": name_only.get("region"),
                "confidence": name_only.get("confidence", 0.0),
                "source": "vision_place_name",
            }
        )
        evidence.append(f"Best guess (unverified): {name_only.get('name')}.")
        _append_clue_evidence(result, metadata, vision)
        return result

    # 4. EXIF caption fallback (try to geocode it too).
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
        _append_clue_evidence(result, metadata, vision)
        _verify_and_adjust(result, vision)
        _enrich(result, vision)
        return result

    _append_clue_evidence(result, metadata, vision)
    evidence.append("No GPS metadata and no confident visual location match.")
    return result


def _enrich(result: dict[str, Any], vision: Optional[dict[str, Any]]) -> None:
    """Consult another OSINT source (Wikipedia) for nearby notable places.

    Adds `nearby_places` for user context and cross-checks the AI's landmark
    guesses against an independent source (small confidence boost on a match).
    """
    if not _has_coords(result):
        return

    data = enrich_location(result["latitude"], result["longitude"])
    places = data.get("nearby_places", [])
    if not places:
        return

    result["nearby_places"] = places[:8]
    evidence: list[str] = result["evidence"]
    evidence.append(
        "Nearby notable places (Wikipedia): "
        + ", ".join(p["title"] for p in places[:5])
        + "."
    )

    if vision and result.get("source") != "exif_gps":
        landmarks = [str(l).lower() for l in (vision.get("landmarks") or []) if l]
        titles = [p["title"].lower() for p in places]
        for lm in landmarks:
            if any(lm in t or t in lm for t in titles):
                evidence.append(
                    f"Corroborated landmark '{lm}' with a nearby Wikipedia place."
                )
                result["confidence"] = _clamp(result.get("confidence", 0.0) + 0.05)
                break


def _merge_alternatives(
    vision: Optional[dict[str, Any]],
    chosen_name: Optional[str],
    extra: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Always surface the other candidate regions, plus any tried alternatives."""
    alts: list[dict[str, Any]] = []
    seen_names: set[str] = set()

    def add(entry: dict[str, Any]) -> None:
        name = entry.get("name")
        if not name or name == chosen_name or name in seen_names:
            return
        seen_names.add(name)
        alts.append(entry)

    for cand in _candidates(vision):
        add(
            {
                "name": cand.get("name"),
                "country": cand.get("country"),
                "latitude": cand.get("latitude"),
                "longitude": cand.get("longitude"),
                "confidence": round(_clamp(float(cand.get("confidence", 0.0) or 0.0)), 3),
            }
        )
    for entry in extra:
        add(entry)
    return alts[:3]


def _emit_retry_evidence(result: dict[str, Any], selection: dict[str, Any]) -> None:
    """Record which locations were crossed out and how many were tried."""
    evidence: list[str] = result["evidence"]
    rejected = selection["rejected"]
    attempts = selection["attempts"]

    if rejected:
        evidence.append(
            f"Crossed out {len(rejected)} location(s) whose surroundings did not "
            f"match the photo."
        )
        for r in rejected[:3]:
            evidence.append(
                f"Rejected {r.get('name')} ({r.get('latitude')}, "
                f"{r.get('longitude')}): {r.get('reason')}."
            )
    if result.get("location_name"):
        suffix = (
            f" after checking {attempts} candidate location(s)"
            if attempts > 1
            else ""
        )
        evidence.append(f"Selected: {result['location_name']}{suffix}.")


def _emit_verification_evidence(
    result: dict[str, Any], report: Optional[dict[str, Any]]
) -> None:
    if not report or report.get("status") == "skipped":
        return
    evidence: list[str] = result["evidence"]
    if report.get("status") == "unavailable":
        evidence.append("Map verification unavailable; location is unverified.")
        return
    nearest = report.get("nearest_m", {})
    for cat in report.get("confirmed", []):
        dist = nearest.get(cat)
        if dist is not None:
            evidence.append(f"Verified: {cat} found ~{int(dist)} m away.")
        else:
            evidence.append(f"Verified: {cat} present nearby.")
    for cat in report.get("missing", []):
        evidence.append(
            f"Note: expected {cat} not found within {report.get('radius_m')} m."
        )


def _set_warning(result: dict[str, Any], status: Optional[str]) -> None:
    if status == "mismatch":
        result["warning"] = (
            "None of the described features could be confirmed near any candidate "
            "location; the result may be unreliable."
        )
    elif status == "partial":
        result["warning"] = (
            "Some described features could not be confirmed near this location."
        )
    elif status == "unavailable":
        result["warning"] = (
            "This location could not be cross-checked against map data "
            "(verification service unavailable), so it is unverified."
        )


def _verify_and_adjust(
    result: dict[str, Any], vision: Optional[dict[str, Any]]
) -> None:
    """Single verification pass for the GPS and caption paths.

    EXIF GPS is authoritative, so it is checked for information only and never
    penalised. The caption path (weaker evidence) can be reduced.
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
    status = report.get("status")
    if status == "skipped":
        return

    _emit_verification_evidence(result, report)
    is_gps = result.get("source") == "exif_gps"
    if not is_gps:
        _set_warning(result, status)
        result["confidence"] = _clamp(
            result.get("confidence", 0.0) * _STATUS_FACTOR.get(status, 1.0)
        )


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
