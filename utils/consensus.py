"""Combine several independent vision answers for ONE image into a consensus.

The vision model's self-reported confidence is poorly calibrated: it says 0.6
whether it is right or wrong. How often independent samples AGREE is a much
better signal. If three samples all put the photo in the same place, that place
is very likely right. If they scatter across three countries, the scene is
genuinely ambiguous.

`merge_samples` groups candidates from all samples by location (same name, or
coordinates within AGREE_KM), scores each group by its mean confidence across
samples, and records per-group support: the share of samples whose TOP pick
landed in that group. The OSINT fusion step uses that support as an
independent confidence signal.

The result has the same shape `analyze_image` returns, plus a `consensus`
block, so the rest of the pipeline consumes it unchanged.
"""

from __future__ import annotations

import math
import os
import re
from typing import Any, Optional

# Two candidates within this distance count as the same answer.
AGREE_KM = float(os.getenv("CONSENSUS_AGREE_KM", "75"))

_LIST_KEYS = (
    "ocr_text",
    "languages",
    "landmarks",
    "named_landmarks",
    "flags",
    "signage",
    "vehicles_plates",
    "architecture_regions",
)


def _clamp(v: Any) -> float:
    try:
        return max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return 0.0


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _norm_name(name: Optional[str]) -> str:
    text = (name or "").lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def _has_coords(c: dict[str, Any]) -> bool:
    return c.get("latitude") is not None and c.get("longitude") is not None


def _same_place(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if _has_coords(a) and _has_coords(b):
        return (
            _haversine_km(a["latitude"], a["longitude"], b["latitude"], b["longitude"])
            <= AGREE_KM
        )
    na, nb = _norm_name(a.get("name")), _norm_name(b.get("name"))
    return bool(na) and na == nb


def _union_list(samples: list[dict[str, Any]], key: str) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for s in samples:
        for item in s.get(key) or []:
            norm = str(item).strip().lower()
            if norm and norm not in seen:
                seen.add(norm)
                out.append(item)
    return out


def _majority_scene(samples: list[dict[str, Any]]) -> dict[str, bool]:
    """A scene feature counts only if most samples saw it.

    Scene booleans drive map verification, so a single sample hallucinating a
    lake should not make every candidate fail the "is there water nearby?"
    check.
    """
    keys: set[str] = set()
    for s in samples:
        keys.update((s.get("scene") or {}).keys())
    need = len(samples) / 2.0
    return {
        k: sum(1 for s in samples if (s.get("scene") or {}).get(k)) > need
        for k in sorted(keys)
    }


def _group_candidates(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Cluster candidates across samples and score each cluster."""
    n = len(samples)
    groups: list[dict[str, Any]] = []

    for idx, sample in enumerate(samples):
        cands = [c for c in sample.get("candidates") or [] if c.get("name")]
        for rank, cand in enumerate(cands):
            conf = _clamp(cand.get("confidence"))
            group = next((g for g in groups if _same_place(g["rep"], cand)), None)
            if group is None:
                group = {"rep": dict(cand), "best_conf": conf, "by_sample": {}, "top_votes": 0}
                groups.append(group)
            # One vote per sample per group: keep that sample's best confidence.
            prev = group["by_sample"].get(idx, 0.0)
            group["by_sample"][idx] = max(prev, conf)
            if rank == 0:
                group["top_votes"] += 1
            if conf > group["best_conf"] or (
                not _has_coords(group["rep"]) and _has_coords(cand)
            ):
                group["rep"] = dict(cand)
                group["best_conf"] = max(group["best_conf"], conf)

    merged: list[dict[str, Any]] = []
    for g in groups:
        mean_conf = sum(g["by_sample"].values()) / n  # missing samples count as 0
        cand = dict(g["rep"])
        cand["confidence"] = round(_clamp(mean_conf), 3)
        cand["samples_agreeing"] = len(g["by_sample"])
        cand["top_votes"] = round(g["top_votes"] / n, 3)
        merged.append(cand)
    merged.sort(key=lambda c: (c["top_votes"], c["confidence"]), reverse=True)
    return merged


_COUNTRY_ALIASES = {
    "usa": "united states", "us": "united states", "u s": "united states",
    "united states of america": "united states", "uk": "united kingdom",
    "great britain": "united kingdom", "england": "united kingdom",
}


def _norm_country(name: Optional[str]) -> str:
    n = _norm_name(name)
    return _COUNTRY_ALIASES.get(n, n)


def _country_agreement(samples: list[dict[str, Any]], top: Optional[dict[str, Any]]) -> Optional[float]:
    """Share of samples whose top pick is in the same country as the winner."""
    target = _norm_country((top or {}).get("country"))
    if not target:
        return None
    firsts = [(s.get("candidates") or [{}])[0] for s in samples]
    return round(sum(_norm_country(f.get("country")) == target for f in firsts) / len(samples), 3)


def merge_samples(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge N vision analyses of the same image into one consensus analysis."""
    usable = [s for s in samples if s and s.get("available")]
    if not usable:
        return samples[0] if samples else {}
    if len(usable) == 1:
        single = dict(usable[0])
        cands = [dict(c) for c in single.get("candidates") or []]
        single["candidates"] = cands
        single["consensus"] = {"samples": 1, "top_agreement": None}
        return single

    candidates = _group_candidates(usable)
    top = candidates[0] if candidates else None

    # Clue/description fields come from the sample that best supports the
    # winning answer, so the reasoning matches the chosen location.
    def supports_top(s: dict[str, Any]) -> float:
        first = (s.get("candidates") or [None])[0]
        if top is None or not first:
            return _clamp(s.get("confidence"))
        return (1.0 if _same_place(first, top) else 0.0) + _clamp(first.get("confidence"))

    base = dict(max(usable, key=supports_top))
    for key in _LIST_KEYS:
        base[key] = _union_list(usable, key)
    base["scene"] = _majority_scene(usable)
    base["candidates"] = candidates

    if top is not None:
        base["best_guess_location"] = {
            "name": top.get("name"),
            "country": top.get("country"),
            "latitude": top.get("latitude"),
            "longitude": top.get("longitude"),
        }
        base["confidence"] = top["confidence"]

    base["consensus"] = {
        "samples": len(usable),
        "top_agreement": top["top_votes"] if top is not None else 0.0,
        "groups": len(candidates),
        "country_agreement": _country_agreement(usable, top),
    }
    return base
