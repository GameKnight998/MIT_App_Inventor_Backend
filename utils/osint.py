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

import math
import os
import re
from typing import Any, Optional

from utils.cache import parallel_map
from utils.enrich import enrich_location
from utils.geocode import (
    forward_geocode,
    forward_geocode_candidates,
    reverse_geocode,
)
from utils.geoenv import annual_climate, climate_consistency, elevation_m
from utils.landmarks import (
    extract_named_features,
    prefetch_named_lookups,
    verify_named_and_water,
)
from utils.solar import sun_consistency
from utils.streetlevel import refine_street_level
from utils.verify import VERIFY_RADIUS_M, detect_expected_features, verify_location

MAX_VERIFY_ATTEMPTS = int(os.getenv("MAX_VERIFY_ATTEMPTS", "5"))
GEO_CANDIDATES_PER_NAME = int(os.getenv("GEO_CANDIDATES_PER_NAME", "3"))
# Candidates whose coordinates fall within this distance of each other are
# treated as the same geographic region for cluster reporting (#11).
CLUSTER_RADIUS_KM = float(os.getenv("CLUSTER_RADIUS_KM", "150"))
# Iterative narrowing: shrink the verify radius to localise the scene tightly.
NARROW_ENABLED = os.getenv("NARROW_ENABLED", "1") not in ("0", "false", "False")
MAX_NARROW_STEPS = int(os.getenv("MAX_NARROW_STEPS", "3"))
NARROW_MIN_RADIUS_M = int(os.getenv("NARROW_MIN_RADIUS_M", "500"))
# Product accuracy target: a pin is only treated as a successful placement if it
# sits within this many kilometres of the real viewpoint. Eval, the API, and the
# landmark-snap pass all share this number so the client, the tests and the
# engine agree on what "close enough" means.
DEFINED_RADIUS_KM = float(os.getenv("DEFINED_RADIUS_KM", "2"))
DEFINED_RADIUS_M = int(round(DEFINED_RADIUS_KM * 1000))
# Verification statuses that make street-level refinement worth its extra vision
# call: we only zoom in on a region the map did not contradict.
_REFINE_STATUSES = ("verified", "partial", "skipped", "unavailable")

_STATUS_FACTOR = {
    "verified": 1.0,
    "skipped": 1.0,
    "unavailable": 1.0,  # map service unreachable -> can't verify, don't penalise
    "partial": 0.7,
    "mismatch": 0.4,
}
# Statuses that end the retry loop with an immediate accept. "unavailable" is
# deliberately NOT here: when we can't reach the map, we must not accept
# whichever candidate happened to hit the outage. Instead we keep checking and,
# if nothing verifies, fall back to the AI's most-confident un-disproven pick.
_ACCEPT_STATUSES = ("verified", "skipped")


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _sigmoid(x: float) -> float:
    try:
        return 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        return 0.0 if x < 0 else 1.0


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _w(name: str, default: float) -> float:
    """Read one fusion weight from the environment, falling back to its default.

    Making every weight env-overridable is what lets `eval/tune_weights.py`
    calibrate them against labelled data and ship the result as configuration
    rather than a code change.
    """
    try:
        return float(os.getenv(f"FUSION_{name}", default))
    except (TypeError, ValueError):
        return default


# Log-odds contributions per independent evidence signal. These de-anchor the
# final confidence from the AI's self-reported number: instead of trusting the
# model and multiplying penalties, each signal casts an additive vote and we
# squash the sum back into [0, 1].
_FUSION_PRIOR = _w("PRIOR", 0.2)
_VERIFY_LOGIT = {
    "verified": _w("VERIFIED", 1.8),
    "partial": _w("PARTIAL", 0.3),
    "mismatch": _w("MISMATCH", -1.6),
    "unavailable": _w("UNAVAILABLE", 0.0),
    "skipped": _w("SKIPPED", 0.0),
}
_SOLAR_LOGIT = {
    "consistent": _w("SOLAR_OK", 0.6),
    "weak_mismatch": _w("SOLAR_WEAK", -0.4),
    "inconsistent": _w("SOLAR_BAD", -2.0),
}
_CLIMATE_LOGIT = {
    "consistent": _w("CLIMATE_OK", 0.5),
    "weak_mismatch": _w("CLIMATE_WEAK", -0.4),
}
_NAMED_LOGIT = {
    "matched": _w("NAMED_OK", 1.2),
    "missed": _w("NAMED_BAD", -0.9),
    "unknown": 0.0,
}
_WATER_NAMED_LOGIT = {
    "matched": _w("WATER_OK", 0.7),
    "missed": _w("WATER_BAD", -0.8),
    "unknown": 0.0,
}
# Script and regional spelling are among the strongest country-level signals in
# real geolocation work (Devanagari, Icelandic thorn, Cyrillic variants), and they
# are independent of the map checks, so they get their own vote rather than being
# lost in the generic clue bonus.
_TEXT_LOGIT = {"match": _w("TEXT_OK", 0.8), "conflict": _w("TEXT_BAD", -1.1)}
_AI_WEIGHT = _w("AI", 1.3)
_CLUE_WEIGHT = _w("CLUE", 3.0)
_AMBIGUITY_WEIGHT = _w("AMBIGUITY", -0.8)
_LANDMARK_CORROBORATED = _w("LANDMARK_CORROBORATED", 1.0)
_PRECISION_BONUS = _w("PRECISION", 0.3)
# A grounded street-level match is strong, independent corroboration: a named
# business or address existing where the scene said it would is hard to get by
# chance. An ungrounded model estimate earns much less.
_STREET_LOGIT = {"geocoded": _w("STREET_OK", 0.9), "model": _w("STREET_MODEL", 0.1)}
# Vehicles/plates pointing at a different country than the scene is a real
# contradiction; agreement is mild corroboration.
_VEHICLE_LOGIT = {
    "match": _w("VEHICLE_OK", 0.5),
    "conflict": _w("VEHICLE_BAD", -1.0),
}
# A generated or manipulated image has no genuine location to find.
_SYNTHETIC_LOGIT = {
    "synthetic": _w("SYNTHETIC", -2.5),
    "likely_synthetic": _w("SYNTHETIC_WEAK", -1.2),
    "manipulated": _w("MANIPULATED", -1.0),
}


def _fuse_confidence(sig: dict[str, Any]) -> float:
    """Combine independent evidence signals into a single confidence in [0, 1].

    Each signal shifts the log-odds up or down; the AI's own confidence is just
    one voter (bounded weight) rather than the anchor everything scales off.
    """
    log_odds = _logit(_FUSION_PRIOR)

    ai = sig.get("ai_conf")
    if ai is not None:
        log_odds += _AI_WEIGHT * (2 * _clamp(float(ai)) - 1)

    log_odds += _VERIFY_LOGIT.get(sig.get("verify_status"), 0.0)
    log_odds += _SOLAR_LOGIT.get(sig.get("solar_status"), 0.0)
    log_odds += _CLIMATE_LOGIT.get(sig.get("climate_status"), 0.0)
    log_odds += _NAMED_LOGIT.get(sig.get("place_status"), 0.0)
    log_odds += _NAMED_LOGIT.get(sig.get("landmark_named"), 0.0)
    log_odds += _WATER_NAMED_LOGIT.get(sig.get("water_named"), 0.0)
    log_odds += _STREET_LOGIT.get(sig.get("street_status"), 0.0)
    log_odds += _VEHICLE_LOGIT.get(sig.get("vehicle_status"), 0.0)
    log_odds += _TEXT_LOGIT.get(sig.get("text_status"), 0.0)
    log_odds += _SYNTHETIC_LOGIT.get(sig.get("synthetic_status"), 0.0)
    log_odds += _CLUE_WEIGHT * min(float(sig.get("clue_bonus", 0.0) or 0.0), 0.25)

    if sig.get("landmark_corroborated"):
        log_odds += _LANDMARK_CORROBORATED

    gap = sig.get("ambiguity_gap")
    if gap is not None:
        # Near-tie between top candidates -> penalty up to _AMBIGUITY_WEIGHT.
        log_odds += _AMBIGUITY_WEIGHT * (1 - min(float(gap) / 0.2, 1.0))

    precision = sig.get("precision_m")
    if precision is not None and precision <= NARROW_MIN_RADIUS_M * 2:
        log_odds += _PRECISION_BONUS

    return round(_clamp(_sigmoid(log_odds)), 3)


def _has_coords(d: Optional[dict[str, Any]]) -> bool:
    return bool(d) and d.get("latitude") is not None and d.get("longitude") is not None


# Short forms and endonyms that vision models emit freely. Without them a plate
# read as "UK" or a script analysis implying "Deutschland" would look like it
# contradicts a place the engine named "United Kingdom" or "Germany".
_REGION_ALIASES = {
    "uk": "united kingdom",
    "gb": "united kingdom",
    "gbr": "united kingdom",
    "britain": "united kingdom",
    "great britain": "united kingdom",
    "england": "united kingdom",
    "scotland": "united kingdom",
    "wales": "united kingdom",
    "northern ireland": "united kingdom",
    "us": "united states",
    "usa": "united states",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "united states of america": "united states",
    "america": "united states",
    "uae": "united arab emirates",
    "holland": "netherlands",
    "the netherlands": "netherlands",
    "deutschland": "germany",
    "espana": "spain",
    "españa": "spain",
    "italia": "italy",
    "nippon": "japan",
    "nihon": "japan",
    "korea": "south korea",
    "republic of korea": "south korea",
    "prc": "china",
    "roc": "taiwan",
    "czechia": "czech republic",
    "türkiye": "turkey",
    "turkiye": "turkey",
}


def _mentions(haystack: str, needle: str) -> bool:
    """Substring test, but short needles must stand as whole words."""
    if not needle:
        return False
    if len(needle) > 4:
        return needle in haystack
    return re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack) is not None


def _place_text(result: dict[str, Any]) -> str:
    """Everything the engine decided about *where*, as one lowercase haystack."""
    return " ".join(
        str(result.get(key) or "")
        for key in ("country", "region", "location_name", "address")
    ).lower()


def _implied_region_status(
    implied: Optional[list[Any]], result: dict[str, Any]
) -> Optional[str]:
    """Compare regions implied by an independent clue against the chosen place.

    Used for both plate/vehicle regions and script/spelling analysis: each is an
    independent read on *which country*, so agreement is corroboration and
    disagreement is a contradiction worth reporting. Returns None when there is
    nothing to compare, so a missing clue is never mistaken for a conflict.
    """
    place = _place_text(result)
    if not place.strip():
        return None

    names: list[str] = []
    for raw in implied or []:
        name = str(raw).strip().lower()
        if not name:
            continue
        names.append(name)
        alias = _REGION_ALIASES.get(name)
        if alias:
            names.append(alias)
    if not names:
        return None

    # Aliases run the other way too: a place named "USA" should accept an implied
    # "United States". Short tokens need word boundaries -- "us" is a substring of
    # "Russia", which would otherwise read as agreement with the United States.
    haystack = place
    for token, alias in _REGION_ALIASES.items():
        if _mentions(place, token):
            haystack += " " + alias
    return "match" if any(_mentions(haystack, n) for n in names) else "conflict"


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
                "cand_conf": cconf,
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
                    "cand_conf": cconf,
                    "source": "vision_geocoded",
                }
            )
    return options


def _verify_rank(report: Optional[dict[str, Any]]) -> float:
    """Rank a verification report, folding in named-landmark / water results."""
    if not report:
        return 0.0
    score = float(report.get("match_score") or 0.0)
    named = report.get("named") or {}
    if named.get("place_status") == "matched":
        score += 0.5
    if named.get("landmark_status") == "matched":
        score += 0.4
    if named.get("water_status") == "matched":
        score += 0.3
    if named.get("place_status") == "missed":
        score -= 0.8
    if named.get("water_status") == "missed":
        score -= 0.3
    return score


def _augment_verification(
    latitude: float,
    longitude: float,
    report: dict[str, Any],
    vision: Optional[dict[str, Any]],
    expected: list[str],
    place_name: Optional[str] = None,
) -> dict[str, Any]:
    """Copy an Overpass report and attach named-landmark / large-water checks.

    Copies so we never mutate a cached Overpass result. A named-place miss
    (e.g. 'Lake Chelan' geocodes hundreds of km away) is treated as a mismatch
    even if some generic water tag existed nearby.
    """
    out = dict(report)
    named = verify_named_and_water(
        latitude, longitude, vision, expected, place_name=place_name
    )
    out["named"] = named
    status = out.get("status")

    if named.get("place_status") == "missed":
        out["status"] = "mismatch"
        if named.get("note"):
            out["note"] = named["note"]
        return out

    if named.get("water_status") == "missed" and status == "verified":
        out["status"] = "partial"
        missing = list(out.get("missing") or [])
        if "named_water" not in missing:
            missing.append("named_water")
        out["missing"] = missing
        out["match_score"] = round(min(float(out.get("match_score") or 1.0), 0.7), 3)

    if named.get("landmark_status") == "matched" or named.get("place_status") == "matched":
        out["match_score"] = round(
            min(1.0, float(out.get("match_score") or 0.0) + 0.25), 3
        )
    return out


def _plan_hypotheses(
    candidates: list[dict[str, Any]], bonus: float
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], Optional[dict[str, Any]]]:
    """Expand candidates into a de-duplicated, ordered list of coordinates to try.

    Materialising the whole plan up front (rather than generating it inside the
    verification loop) is what makes parallel pre-fetching possible: we cannot
    warm caches for lookups we have not decided to make yet. Order is unchanged,
    so selection behaviour is identical.
    """
    plan: list[tuple[dict[str, Any], dict[str, Any]]] = []
    seen: set[tuple[float, float]] = set()
    name_only: Optional[dict[str, Any]] = None

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
            plan.append((cand, hyp))
    return plan, name_only


def _prefetch_verification(
    plan: list[tuple[dict[str, Any], dict[str, Any]]],
    expected: list[str],
    vision: Optional[dict[str, Any]],
) -> None:
    """Warm the map/named-feature caches for every hypothesis we may check.

    The selection loop below must stay sequential: it exits as soon as a location
    verifies, and that early exit is what keeps the AI's ranking meaningful. But
    the lookups it makes are independent and network-bound, so issuing them
    concurrently first turns a chain of round-trips into one. Only successful
    Overpass results are cached, so an outage is still retried rather than
    remembered.
    """
    if not expected:
        return
    coords = [
        (hyp["latitude"], hyp["longitude"]) for _, hyp in plan[:MAX_VERIFY_ATTEMPTS]
    ]
    if not coords:
        return
    prefetch_named_lookups(coords, vision, expected)
    parallel_map(lambda c: verify_location(c[0], c[1], expected), coords)


def _select_location(
    candidates: list[dict[str, Any]],
    bonus: float,
    expected: list[str],
    vision: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Try candidate locations, crossing out those the map contradicts.

    Returns a dict with the chosen hypothesis (or None), its verification
    report, the list of rejected locations, untried alternatives, a name-only
    fallback, and the number of verification attempts made.
    """
    attempts = 0
    rejected: list[dict[str, Any]] = []
    alternatives: list[dict[str, Any]] = []
    best_partial: Optional[tuple[dict[str, Any], dict[str, Any]]] = None
    best_unavailable: Optional[tuple[dict[str, Any], dict[str, Any]]] = None
    best_mismatch: Optional[tuple[dict[str, Any], dict[str, Any]]] = None
    chosen: Optional[dict[str, Any]] = None
    chosen_report: Optional[dict[str, Any]] = None

    plan, name_only = _plan_hypotheses(candidates, bonus)
    _prefetch_verification(plan, expected, vision)

    for _cand, hyp in plan:
        if attempts >= MAX_VERIFY_ATTEMPTS:
            alternatives.append(_alt(hyp))
            continue

        report = _augment_verification(
            hyp["latitude"],
            hyp["longitude"],
            verify_location(hyp["latitude"], hyp["longitude"], expected),
            vision,
            expected,
            place_name=hyp.get("name"),
        )
        attempts += 1
        hyp["verification"] = report
        status = report.get("status")

        if status in _ACCEPT_STATUSES:
            chosen, chosen_report = hyp, report
            break

        if status == "partial":
            if best_partial is None or _verify_rank(report) > _verify_rank(
                best_partial[1]
            ):
                best_partial = (hyp, report)
            alternatives.append(_alt(hyp))
        elif status == "unavailable":
            # Couldn't check this one. Keep the highest-confidence such
            # candidate (candidates arrive best-first) as a fallback, preferring
            # one whose named landmark/place still matched.
            if best_unavailable is None:
                best_unavailable = (hyp, report)
            elif _verify_rank(report) > _verify_rank(best_unavailable[1]):
                best_unavailable = (hyp, report)
            elif _verify_rank(report) == _verify_rank(
                best_unavailable[1]
            ) and hyp.get("base", 0.0) > best_unavailable[0].get("base", 0.0):
                best_unavailable = (hyp, report)
            alternatives.append(_alt(hyp))
        else:  # mismatch -> cross it out
            named_note = (report.get("named") or {}).get("note")
            reason = named_note or (
                f"expected {report.get('missing')} not found within "
                f"{report.get('radius_m')} m"
            )
            rejected.append(
                {
                    "name": hyp.get("name"),
                    "latitude": hyp.get("latitude"),
                    "longitude": hyp.get("longitude"),
                    "reason": reason,
                }
            )
            if best_mismatch is None or _verify_rank(report) > _verify_rank(
                best_mismatch[1]
            ):
                best_mismatch = (hyp, report)

    if chosen is None:
        # Preference when nothing verified cleanly: a real partial match (has
        # corroborating evidence) > the AI's top un-checkable pick > a location
        # the map actively contradicted.
        if best_partial is not None:
            chosen, chosen_report = best_partial
        elif best_unavailable is not None:
            chosen, chosen_report = best_unavailable
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
    metadata: dict[str, Any],
    vision: dict[str, Any],
    forensics: Optional[dict[str, Any]] = None,
    *,
    image_path: Optional[str] = None,
) -> dict[str, Any]:
    """Fuse all evidence into a location, then attach forensics, clustered
    alternatives and an auditable reasoning trace.

    `image_path` enables the street-level refinement pass, which needs to look at
    the image again once a region is known. Omitting it simply skips refinement,
    which is what the offline eval harness does.

    The core fusion logic lives in `_determine_core`; this wrapper enriches the
    result with the cross-cutting fields (forensics, clusters, trace) so those
    are computed once regardless of which evidence path produced the answer.
    """
    result = _determine_core(metadata, vision, image_path=image_path, forensics=forensics)
    result["forensics"] = forensics or {}
    result["alternative_clusters"] = _cluster_candidates(vision)
    _warn_if_synthetic(result, forensics)
    if forensics and forensics.get("notes"):
        # Surface the most useful provenance note in the evidence list.
        for note in forensics["notes"]:
            if "expected" in note or "unusual" in note:
                result["evidence"].append(f"Forensics: {note}")
                break
    _attach_defined_radius(result)
    result["reasoning_trace"] = _build_reasoning_trace(
        result, metadata, vision, forensics
    )
    return result


def _determine_core(
    metadata: dict[str, Any],
    vision: dict[str, Any],
    *,
    image_path: Optional[str] = None,
    forensics: Optional[dict[str, Any]] = None,
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
        _apply_solar(result, metadata, vision)  # info only for GPS
        _enrich(result, vision)
        return result

    # 2. Vision candidates with iterative, map-checked verification.
    expected = detect_expected_features(vision)
    _prefetch_geocode(vision)  # warm the geocode cache for all names in parallel
    selection = _select_location(_candidates(vision), bonus, expected, vision)
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
        # Collect the independent signals; confidence is recomputed by log-linear
        # fusion at the end of this branch (see _fuse_confidence).
        result["_signals"] = {
            "ai_conf": chosen.get("cand_conf"),
            "clue_bonus": bonus,
            "ambiguity_gap": gap,
            "verify_status": status,
            "landmark_corroborated": False,
        }
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

        # Snap a city-centroid pin onto a verified landmark / named water body
        # before narrowing or street refinement, so those passes search around
        # the feature rather than the regional centre.
        _snap_to_verified_feature(result)

        # Iterative narrowing: only when the map fully confirmed the scene, so
        # tightening the radius is meaningful. Tight precision -> small boost.
        if NARROW_ENABLED and status == "verified" and expected and _has_coords(result):
            precision_m, narrow_steps = _narrow_location(
                result["latitude"], result["longitude"], expected
            )
            result["precision_m"] = precision_m
            for step in narrow_steps:
                evidence.append(step)
            if precision_m <= NARROW_MIN_RADIUS_M * 2:
                result["confidence"] = _clamp(result.get("confidence", 0.0) + 0.03)

        # Street-level refinement: with a region the map did not contradict, look
        # again for the exact spot. Runs before the environmental checks so those
        # describe the refined point rather than the regional centroid.
        _apply_street_refinement(result, vision, image_path, status)

        _prefetch_context(result, vision)
        _apply_solar(result, metadata, vision)
        _apply_climate(result, vision)
        _append_clue_evidence(result, metadata, vision)
        _enrich(result, vision)

        # Independent log-linear fusion: recompute confidence from all signals so
        # the answer isn't anchored to the AI's self-reported number.
        signals = result.pop("_signals", {})
        named = (result.get("verification") or {}).get("named") or {}
        street = result.get("street_level") or {}
        signals["solar_status"] = (result.get("solar") or {}).get("status")
        signals["climate_status"] = (result.get("climate_check") or {}).get("status")
        signals["precision_m"] = result.get("precision_m")
        signals["place_status"] = named.get("place_status")
        signals["landmark_named"] = named.get("landmark_status")
        signals["water_named"] = named.get("water_status")
        if street.get("refined"):
            signals["street_status"] = (
                "model" if street.get("method") == "model_estimate" else "geocoded"
            )
        signals["synthetic_status"] = (forensics or {}).get("authenticity")
        _apply_text_signal(result, vision, signals)
        result["confidence"] = _fuse_confidence(signals)
        # Retained so later, independently-obtained evidence (vehicle
        # identification) can be added as one more voter instead of being
        # bolted on as an ad-hoc multiplier.
        result["fusion_signals"] = signals
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
                # Vision branch: fusion consumes this flag; other branches (no
                # fusion) still get the direct bump.
                if "_signals" in result:
                    result["_signals"]["landmark_corroborated"] = True
                else:
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


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _cluster_candidates(vision: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group candidates that are geographically close into regional clusters.

    Turns "three unrelated cities" into "one region we're fairly sure about",
    which is far more useful when a scene is ambiguous within a single area
    (e.g. several Pacific-Northwest guesses). Confidence is summed per cluster
    (clamped) because agreeing candidates reinforce the same region.
    """
    coords = [c for c in _candidates(vision) if _has_coords(c)]
    coords.sort(
        key=lambda c: _clamp(float(c.get("confidence", 0.0) or 0.0)), reverse=True
    )

    clusters: list[dict[str, Any]] = []
    for cand in coords:
        conf = _clamp(float(cand.get("confidence", 0.0) or 0.0))
        member = {"name": cand.get("name"), "confidence": round(conf, 3)}
        placed = False
        for cl in clusters:
            if (
                _haversine_km(cl["_lat"], cl["_lon"], cand["latitude"], cand["longitude"])
                <= CLUSTER_RADIUS_KM
            ):
                cl["members"].append(member)
                cl["confidence"] = round(_clamp(cl["confidence"] + conf), 3)
                placed = True
                break
        if not placed:
            clusters.append(
                {
                    "region": cand.get("region") or cand.get("name"),
                    "country": cand.get("country"),
                    "confidence": round(conf, 3),
                    "members": [member],
                    "_lat": cand["latitude"],
                    "_lon": cand["longitude"],
                }
            )

    for cl in clusters:
        cl.pop("_lat", None)
        cl.pop("_lon", None)
    clusters.sort(key=lambda cl: cl["confidence"], reverse=True)
    return clusters


def _narrow_location(
    latitude: float, longitude: float, expected: list[str]
) -> tuple[int, list[str]]:
    """Coarse-to-fine narrowing: shrink the search radius while the scene's
    features still all verify, to estimate how tightly they localise the point.

    Returns (precision_m, steps). `precision_m` is the smallest radius at which
    every expected feature was still confirmed. Bounded by MAX_NARROW_STEPS and
    the Overpass circuit breaker; stops immediately if the map is unavailable.
    """
    radius = VERIFY_RADIUS_M
    precision = radius
    steps: list[str] = []
    for _ in range(MAX_NARROW_STEPS):
        new_radius = radius // 2
        if new_radius < NARROW_MIN_RADIUS_M:
            break
        report = verify_location(latitude, longitude, expected, new_radius)
        status = report.get("status")
        if status == "unavailable":
            steps.append(f"Narrowing stopped: map unavailable at {new_radius} m.")
            break
        if status == "verified":
            radius = new_radius
            precision = new_radius
            steps.append(f"All features still present within {new_radius} m.")
        else:
            steps.append(
                f"Not all features within {new_radius} m; localised to ~{radius} m."
            )
            break
    return precision, steps


# Features photographed from far away: snapping the pin onto the object would
# miss the camera (a Matterhorn shot is taken from Zermatt, not the summit).
_DISTANT_VIEW_TYPES = {
    "peak",
    "volcano",
    "ridge",
    "glacier",
    "mountain",
}
# Features the camera is typically standing next to.
_CLOSE_RANGE_TYPES = {
    "attraction",
    "monument",
    "museum",
    "building",
    "house",
    "retail",
    "shop",
    "station",
    "park",
}
_WATER_FEATURE_TYPES = {
    "lake",
    "reservoir",
    "pond",
    "lagoon",
}


def _snap_targets(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Named map features we already confirmed sit near the current pin.

    Distant-view landmarks (peaks) are omitted: the photo is of them, not at
    them. Generic nearby water is omitted for the same reason (a city pin
    should not jump to the bay). A lake the scene actually named is kept,
    because that name *is* the location.
    """
    named = (result.get("verification") or {}).get("named") or {}
    targets: list[dict[str, Any]] = []

    for mark in named.get("matched_landmarks") or []:
        if mark.get("latitude") is None or mark.get("longitude") is None:
            continue
        osm = str(mark.get("osm_type") or mark.get("category") or "").lower()
        if osm in _DISTANT_VIEW_TYPES:
            continue
        targets.append(
            {
                "name": mark.get("name"),
                "latitude": mark["latitude"],
                "longitude": mark["longitude"],
                "kind": "landmark",
                "osm_type": osm,
            }
        )

    if named.get("place_status") == "matched":
        plat = named.get("place_latitude")
        plon = named.get("place_longitude")
        osm = str(named.get("place_osm_type") or "").lower()
        if plat is not None and plon is not None and osm not in _DISTANT_VIEW_TYPES:
            if osm in _CLOSE_RANGE_TYPES or osm in _WATER_FEATURE_TYPES:
                targets.append(
                    {
                        "name": named.get("place_name") or result.get("location_name"),
                        "latitude": plat,
                        "longitude": plon,
                        "kind": "place",
                        "osm_type": osm,
                    }
                )

    return targets


def _snap_to_verified_feature(result: dict[str, Any]) -> None:
    """Move a regional pin onto the nearest verified named feature.

    Vision often drops a city or county centroid (Paris, Eastern Oregon) even
    when it correctly named a landmark a few kilometres away. The 2 km product
    radius cannot be met from a city centroid, so once Nominatim has confirmed
    the landmark we treat its coordinates as the answer.
    """
    if result.get("source") == "exif_gps" or not _has_coords(result):
        return

    best: Optional[dict[str, Any]] = None
    best_dist = 1e9
    for target in _snap_targets(result):
        dist = _haversine_km(
            result["latitude"],
            result["longitude"],
            target["latitude"],
            target["longitude"],
        )
        target["distance_km"] = dist
        # Prefer landmarks over a lake centroid or the place's own geocode.
        rank = {"landmark": 0, "place": 1, "water": 2}.get(target["kind"], 3)
        current_rank = {"landmark": 0, "place": 1, "water": 2}.get(
            (best or {}).get("kind"), 3
        )
        if best is None or rank < current_rank or (rank == current_rank and dist < best_dist):
            best = target
            best_dist = dist

    # Already on the feature, or nothing confirmed.
    if best is None or best_dist < 0.05:
        return

    result["latitude"] = best["latitude"]
    result["longitude"] = best["longitude"]
    if best.get("name"):
        result["location_name"] = best["name"]
    result["precision_m"] = min(
        int(result.get("precision_m") or DEFINED_RADIUS_M),
        DEFINED_RADIUS_M,
    )
    result["snapped_to"] = {
        "name": best.get("name"),
        "kind": best["kind"],
        "distance_moved_km": round(best_dist, 3),
    }
    result["evidence"].append(
        f"Snapped the pin {best_dist:.1f} km onto the verified {best['kind']} "
        f"'{best.get('name')}' so the estimate sits inside the "
        f"{DEFINED_RADIUS_KM:g} km target radius."
    )
    # Address was for the old centroid; refresh it at the snapped point.
    rev = reverse_geocode(best["latitude"], best["longitude"])
    if rev:
        result["address"] = rev.get("display_name") or result.get("address")
        result["country"] = result.get("country") or rev.get("country")
        result["region"] = result.get("region") or rev.get("region")


def _attach_defined_radius(result: dict[str, Any]) -> None:
    """Publish the 2 km accuracy target and whether this pin claims to meet it."""
    result["defined_radius_km"] = DEFINED_RADIUS_KM
    result["defined_radius_m"] = DEFINED_RADIUS_M

    if result.get("source") == "exif_gps" and result.get("precision_m") is None:
        result["precision_m"] = 15

    if not _has_coords(result):
        result["meets_defined_radius"] = None
        return

    precision = result.get("precision_m")
    if precision is None and result.get("source") == "street_level":
        precision = DEFINED_RADIUS_M
        result["precision_m"] = precision
    if precision is None and result.get("snapped_to"):
        precision = DEFINED_RADIUS_M
        result["precision_m"] = precision

    meets = precision is not None and float(precision) <= DEFINED_RADIUS_M
    result["meets_defined_radius"] = meets
    if meets:
        return
    note = (
        f"This pin is a regional estimate, not a placement within the "
        f"{DEFINED_RADIUS_KM:g} km defined radius."
    )
    result["evidence"].append(note)
    if not result.get("warning"):
        result["warning"] = note


def _apply_street_refinement(
    result: dict[str, Any],
    vision: Optional[dict[str, Any]],
    image_path: Optional[str],
    status: Optional[str],
) -> None:
    """Move the answer from a region to an exact spot when the image supports it.

    Gated on a verification status that did not contradict the region: refining
    inside an area we already believe is wrong would just produce a precise wrong
    answer. On success the coordinates, address and precision are all replaced,
    and the source is relabelled so the client can say how the point was found.
    """
    if not _has_coords(result) or not image_path:
        return
    if status not in _REFINE_STATUSES:
        return

    city_hint = result.get("region") or result.get("country")
    street = refine_street_level(
        result["latitude"],
        result["longitude"],
        vision,
        image_path=image_path,
        place_label=result.get("address") or result.get("location_name"),
        city_hint=result.get("location_name") or city_hint,
    )
    result["street_level"] = street
    for line in street.get("evidence", []):
        result["evidence"].append(line)

    if not street.get("refined"):
        if street.get("attempted") and street.get("note"):
            result["evidence"].append(f"Street-level: {street['note']}")
        return

    result["latitude"] = street["latitude"]
    result["longitude"] = street["longitude"]
    if street.get("address"):
        result["address"] = street["address"]
        result["location_name"] = street["address"].split(",")[0].strip() or result[
            "location_name"
        ]
    result["precision_m"] = street.get("precision_m") or result.get("precision_m")
    result["source"] = "street_level"


def _warn_if_synthetic(
    result: dict[str, Any], forensics: Optional[dict[str, Any]]
) -> None:
    """Say plainly when the image may not depict a real place.

    A generated image can still produce a coherent-looking location, so this must
    be surfaced as a warning rather than left in a nested forensics field.
    """
    synthetic = (forensics or {}).get("synthetic") or {}
    status = synthetic.get("status")
    if status not in ("synthetic", "likely_synthetic", "manipulated"):
        return

    result["authenticity"] = status
    note = synthetic.get("note") or "This image may not be a genuine photograph."
    result["evidence"].append(f"Authenticity: {note}")
    # A synthetic image outranks any location warning already set.
    result["warning"] = note
    if status == "synthetic":
        # Never present a confident location for a generated image.
        result["confidence"] = min(float(result.get("confidence", 0.0)), 0.15)


def _apply_text_signal(
    result: dict[str, Any],
    vision: Optional[dict[str, Any]],
    signals: dict[str, Any],
) -> None:
    """Score the script/spelling read against the chosen place.

    Writing systems and regional spellings are close to decisive at country level
    -- Devanagari, Icelandic thorn, or traditional vs simplified Chinese rule out
    most of the map on their own -- so this is treated as its own signal rather
    than one more entry in the generic clue bonus.
    """
    text = (vision or {}).get("text_analysis") or {}
    implied = text.get("implied_countries") or []
    status = _implied_region_status(implied, result)
    if status is None:
        return

    signals["text_status"] = status
    shown = ", ".join(str(c) for c in implied[:3])
    script = text.get("primary_script")
    if status == "match":
        detail = f"{script} script" if script else "the visible text"
        result["evidence"].append(
            f"Text cross-check: {detail} is consistent with {shown}."
        )
    else:
        note = (
            f"The visible text points to {shown}, which does not match the "
            "chosen region."
        )
        result["evidence"].append(f"Text cross-check: {note}")
        if not result.get("warning"):
            result["warning"] = note


def apply_vehicle_signal(
    result: dict[str, Any], vehicle: Optional[dict[str, Any]]
) -> None:
    """Fold vehicle/plate evidence into an already-computed location.

    Vehicle identification runs concurrently with the rest of the pipeline (it is
    an independent vision call), so it arrives after fusion. Rather than applying
    an ad-hoc multiplier, the result is added as one more voter and the same
    fusion is recomputed, keeping a single definition of how confidence is built.

    Plate-implied regions are the useful part: a French plate in a scene the
    engine placed in Ontario is a genuine contradiction worth reporting.
    """
    if not vehicle or not vehicle.get("available"):
        return

    from utils.vehicle import vehicle_evidence

    for line in vehicle_evidence(vehicle):
        result["evidence"].append(line)

    result["vehicle"] = vehicle
    status = _implied_region_status(vehicle.get("implied_regions"), result)
    if status is None:
        return

    matched = status == "match"
    result["vehicle_check"] = {
        "status": status,
        "implied_regions": vehicle.get("implied_regions", [])[:4],
        "note": (
            "Vehicles and plates are consistent with the chosen region."
            if matched
            else "Vehicles/plates suggest "
            + ", ".join(vehicle.get("implied_regions", [])[:3])
            + ", which does not match the chosen region."
        ),
    }
    result["evidence"].append(f"Vehicle cross-check: {result['vehicle_check']['note']}")

    # EXIF GPS is authoritative; a plate never overrides it.
    if result.get("source") == "exif_gps":
        return

    signals = result.get("fusion_signals")
    if isinstance(signals, dict):
        signals["vehicle_status"] = status
        result["confidence"] = _fuse_confidence(signals)
    elif status == "conflict":
        result["confidence"] = _clamp(result.get("confidence", 0.0) * 0.85)

    if status == "conflict" and not result.get("warning"):
        result["warning"] = result["vehicle_check"]["note"]


def _apply_solar(
    result: dict[str, Any],
    metadata: Optional[dict[str, Any]],
    vision: Optional[dict[str, Any]],
) -> None:
    """Cross-check the chosen coordinate against the sun's computed position.

    A strong day/night contradiction tempers confidence and warns; a match
    gives a small boost. EXIF GPS is authoritative, so it is only annotated.
    """
    if not _has_coords(result):
        return
    timestamp = metadata.get("timestamp") if metadata else None
    report = sun_consistency(
        result["latitude"], result["longitude"], timestamp, vision
    )
    status = report.get("status")
    if status == "unknown" and report.get("expected_elevation") is None:
        return

    result["solar"] = report
    if report.get("note"):
        result["evidence"].append(f"Sun check: {report['note']}")

    if result.get("source") == "exif_gps":
        return  # authoritative; never penalise GPS

    if status == "inconsistent":
        result["confidence"] = _clamp(result.get("confidence", 0.0) * 0.6)
        if not result.get("warning"):
            result["warning"] = report.get("note")
    elif status == "weak_mismatch":
        result["confidence"] = _clamp(result.get("confidence", 0.0) * 0.9)
    elif status == "consistent":
        result["confidence"] = _clamp(result.get("confidence", 0.0) + 0.03)


def _prefetch_geocode(vision: Optional[dict[str, Any]]) -> None:
    """Warm the geocode cache for every candidate name in parallel.

    `_select_location` geocodes candidates sequentially; pre-fetching them
    concurrently means those cached lookups return instantly, cutting the
    dominant network latency without changing the selection order.
    """
    queries: list[str] = []
    for cand in _candidates(vision):
        name = cand.get("name")
        if name:
            queries.append(name)
            queries.append(
                ", ".join(
                    p for p in (name, cand.get("region"), cand.get("country")) if p
                )
            )
    for name in extract_named_features(vision):
        queries.append(name)
    # De-dupe while preserving order.
    seen_q: set[str] = set()
    uniq: list[str] = []
    for q in queries:
        key = q.strip().lower()
        if key and key not in seen_q:
            seen_q.add(key)
            uniq.append(q)
    if uniq:
        parallel_map(
            lambda q: forward_geocode_candidates(q, GEO_CANDIDATES_PER_NAME), uniq
        )


def _prefetch_context(
    result: dict[str, Any], vision: Optional[dict[str, Any]]
) -> None:
    """Fetch Wikipedia, elevation and climate for the final point concurrently.

    These three are independent of each other and each costs a round-trip. The
    functions that consume them stay sequential and readable; they just find the
    answers already cached.
    """
    if not _has_coords(result):
        return
    lat, lon = result["latitude"], result["longitude"]

    jobs = [lambda: enrich_location(lat, lon)]
    scene = (vision or {}).get("scene") or {}
    if any(scene.get(k) for k in ("desert", "snow", "forest")):
        jobs.append(lambda: elevation_m(lat, lon))
        jobs.append(lambda: annual_climate(lat, lon))
    parallel_map(lambda fn: fn(), jobs)


def _apply_climate(result: dict[str, Any], vision: Optional[dict[str, Any]]) -> None:
    """Cross-check the scene's climate/terrain claims against real-world data.

    Only runs when the image describes a climate-checkable scene (snow, desert,
    forest). Elevation and climate are fetched concurrently. A mismatch tempers
    confidence; a match nudges it up. EXIF GPS is authoritative (info only).
    """
    if not _has_coords(result):
        return
    scene = (vision or {}).get("scene") or {}
    if not any(scene.get(k) for k in ("desert", "snow", "forest")):
        return

    lat, lon = result["latitude"], result["longitude"]
    elev, clim = parallel_map(lambda fn: fn(lat, lon), [elevation_m, annual_climate])
    if elev is None and clim is None:
        return

    if elev is not None:
        result["elevation_m"] = elev

    report = climate_consistency(scene, elev, clim)
    if report.get("status") == "unknown":
        return

    result["climate_check"] = report
    if report.get("note"):
        result["evidence"].append(f"Climate check: {report['note']}")

    if result.get("source") == "exif_gps":
        return
    if report["status"] == "weak_mismatch":
        result["confidence"] = _clamp(result.get("confidence", 0.0) * 0.92)
    elif report["status"] == "consistent":
        result["confidence"] = _clamp(result.get("confidence", 0.0) + 0.02)


def _build_reasoning_trace(
    result: dict[str, Any],
    metadata: Optional[dict[str, Any]],
    vision: Optional[dict[str, Any]],
    forensics: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Assemble an ordered, auditable narrative of how the answer was reached."""
    trace: list[dict[str, Any]] = []

    meta_lines: list[str] = []
    if metadata:
        if metadata.get("timestamp"):
            meta_lines.append(f"Photo timestamp: {metadata['timestamp']}.")
        cam = metadata.get("camera", {}) or {}
        if cam.get("model"):
            label = f"{cam.get('make') or ''} {cam.get('model')}".strip()
            meta_lines.append(f"Camera: {label}.")
    if forensics:
        meta_lines.extend(forensics.get("notes", []))
        synthetic = forensics.get("synthetic") or {}
        for signal in synthetic.get("signals", [])[:3]:
            meta_lines.append(signal)
    if meta_lines:
        trace.append({"stage": "Metadata & forensics", "details": meta_lines})

    vehicle = result.get("vehicle") or {}
    if vehicle.get("available") and vehicle.get("vehicles_present"):
        from utils.vehicle import vehicle_evidence

        vehicle_lines = vehicle_evidence(vehicle)
        check = result.get("vehicle_check") or {}
        if check.get("note"):
            vehicle_lines.append(check["note"])
        if vehicle_lines:
            trace.append({"stage": "Vehicle analysis", "details": vehicle_lines})

    clue_lines: list[str] = []
    if vision:
        road_side = vision.get("road_side")
        if road_side and road_side != "unknown":
            clue_lines.append(f"Traffic drives on the {road_side}.")
        ta = vision.get("text_analysis") or {}
        if ta.get("primary_script"):
            clue_lines.append(f"Script/alphabet: {ta['primary_script']}.")
        if ta.get("regional_spelling"):
            clue_lines.append(
                "Regional spelling: " + ", ".join(ta["regional_spelling"]) + "."
            )
        if ta.get("implied_countries"):
            clue_lines.append(
                "Text implies: " + ", ".join(ta["implied_countries"]) + "."
            )
        if vision.get("architecture_style"):
            line = f"Architecture: {vision['architecture_style']}"
            regions = vision.get("architecture_regions") or []
            if regions:
                line += " (implies " + ", ".join(regions) + ")"
            clue_lines.append(line + ".")
        scene = vision.get("scene") or {}
        present = [k for k, v in scene.items() if v]
        if present:
            clue_lines.append("Scene features: " + ", ".join(present) + ".")
        for key in ("vegetation", "terrain", "climate"):
            if vision.get(key):
                clue_lines.append(f"{key.capitalize()}: {vision[key]}.")
    if clue_lines:
        trace.append({"stage": "Visual clues", "details": clue_lines})

    cand_lines: list[str] = []
    for cand in _candidates(vision):
        conf = _clamp(float(cand.get("confidence", 0.0) or 0.0))
        why = cand.get("why") or ""
        cand_lines.append(f"{cand.get('name')} ({round(conf, 2)}): {why}".strip())
    clusters = result.get("alternative_clusters") or []
    if any(len(cl.get("members", [])) > 1 for cl in clusters):
        for cl in clusters:
            cand_lines.append(
                f"Cluster {cl.get('region')} (~{cl.get('confidence')}): "
                + ", ".join(m["name"] for m in cl.get("members", []) if m.get("name"))
            )
    if cand_lines:
        trace.append({"stage": "Candidate regions", "details": cand_lines})

    verification = result.get("verification") or {}
    status = verification.get("status")
    named = verification.get("named") or {}
    if verification and (status not in (None, "skipped") or named.get("note")):
        verify_lines: list[str] = []
        if verification.get("confirmed"):
            verify_lines.append(
                "Confirmed nearby: " + ", ".join(verification["confirmed"]) + "."
            )
        if verification.get("missing"):
            verify_lines.append(
                "Not found nearby: " + ", ".join(verification["missing"]) + "."
            )
        if verification.get("context"):
            verify_lines.append(
                "Context features: " + ", ".join(verification["context"]) + "."
            )
        if named.get("note"):
            verify_lines.append(named["note"])
        for rej in (result.get("rejected") or [])[:5]:
            verify_lines.append(f"Rejected {rej.get('name')}: {rej.get('reason')}.")
        if status == "unavailable":
            verify_lines.append("Map service unavailable, so this is unverified.")
        if verify_lines:
            trace.append({"stage": "Map verification", "details": verify_lines})

    street = result.get("street_level") or {}
    if street.get("attempted"):
        street_lines: list[str] = list(street.get("evidence") or [])
        if not street.get("refined") and street.get("note"):
            street_lines.append(street["note"])
        for cand in (street.get("candidates") or [])[:3]:
            street_lines.append(
                f"Map match for '{cand.get('query')}': {cand.get('name')} "
                f"({cand.get('distance_km')} km, {cand.get('provider')})."
            )
        if street_lines:
            trace.append({"stage": "Street-level refinement", "details": street_lines})

    refine: list[str] = []
    solar = result.get("solar")
    if solar and solar.get("note"):
        refine.append(f"Sun geometry: {solar['note']}")
    climate_check = result.get("climate_check")
    if climate_check and climate_check.get("note"):
        refine.append(f"Climate/elevation: {climate_check['note']}")
    if result.get("elevation_m") is not None:
        refine.append(f"Ground elevation ~{result['elevation_m']} m.")
    if result.get("precision_m"):
        refine.append(
            f"Scene features localise to within ~{result['precision_m']} m."
        )
    if refine:
        trace.append({"stage": "Sun, climate & narrowing", "details": refine})

    conclusion: list[str] = []
    if result.get("location_name"):
        conclusion.append(
            f"{result['location_name']} - confidence "
            f"{round(float(result.get('confidence', 0.0)), 2)} "
            f"(source: {result.get('source')})."
        )
    else:
        conclusion.append("Location could not be determined from this image.")
    if result.get("defined_radius_km") is not None:
        if result.get("meets_defined_radius") is True:
            conclusion.append(
                f"Estimate is within the {result['defined_radius_km']:g} km "
                "defined radius."
            )
        elif result.get("meets_defined_radius") is False:
            conclusion.append(
                f"Estimate does not claim the {result['defined_radius_km']:g} km "
                "defined radius (regional pin only)."
            )
        else:
            conclusion.append(
                f"Defined radius is {result['defined_radius_km']:g} km; "
                "no coordinates were produced."
            )
    if result.get("warning"):
        conclusion.append(result["warning"])
    trace.append({"stage": "Conclusion", "details": conclusion})

    return trace


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
        named = (report or {}).get("named") or {}
        if named.get("note"):
            result["evidence"].append(f"Named-feature check: {named['note']}")
        return
    evidence: list[str] = result["evidence"]
    named = report.get("named") or {}
    if named.get("note"):
        evidence.append(f"Named-feature check: {named['note']}")
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
    if expected:
        report = verify_location(lat, lon, expected)
    else:
        report = {
            "status": "skipped",
            "note": "No checkable features described in the image.",
            "expected": [],
            "confirmed": [],
            "missing": [],
            "match_score": 1.0,
        }
    report = _augment_verification(
        lat, lon, report, vision, expected, place_name=result.get("location_name")
    )
    result["verification"] = report
    status = report.get("status")
    if status == "skipped" and not (report.get("named") or {}).get("note"):
        return

    _emit_verification_evidence(result, report)
    is_gps = result.get("source") == "exif_gps"
    if not is_gps and status != "skipped":
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
