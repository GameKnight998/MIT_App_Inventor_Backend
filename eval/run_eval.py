"""Evaluation harness for the Image Locator OSINT pipeline.

Measures whether a location guess is in the right country/region, within a
labelled radius, and whether confidence is calibrated (high when correct, low
when wrong). Default mode is fully offline: vision/metadata come from fixtures
in labels.json and Nominatim/Overpass are stubbed from the same gazetteer.

Usage:
  venv\\Scripts\\python.exe eval\\run_eval.py
  venv\\Scripts\\python.exe eval\\run_eval.py --live     # real Nominatim/Overpass
  venv\\Scripts\\python.exe eval\\run_eval.py --json     # machine-readable summary

Add new labelled cases to eval/labels.json. Optional image_path on a case is
reserved for a future live-vision pass; fixtures do not require photos.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

LABELS_PATH = Path(__file__).resolve().parent / "labels.json"


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def _country_hit(pred: Optional[str], expected: dict[str, Any]) -> bool:
    if expected.get("expect_unknown"):
        return not pred
    target = [_norm(expected.get("country"))] + [
        _norm(a) for a in (expected.get("country_aliases") or [])
    ]
    target = [t for t in target if t]
    got = _norm(pred)
    if not target:
        return True
    return any(t in got or got in t for t in target if got)


def _region_hit(pred: Optional[str], expected: dict[str, Any]) -> bool:
    if expected.get("expect_unknown"):
        return not pred
    want = _norm(expected.get("region"))
    if not want:
        return True
    got = _norm(pred)
    return bool(got) and (want in got or got in want)


def _name_hit(pred: Optional[str], expected: dict[str, Any]) -> Optional[bool]:
    needles = [_norm(n) for n in (expected.get("name_contains") or []) if n]
    if not needles:
        return None
    if expected.get("expect_unknown"):
        return not pred
    got = _norm(pred)
    return any(n in got for n in needles)


def _within_radius(pred_lat, pred_lon, expected: dict[str, Any]) -> Optional[bool]:
    if expected.get("expect_unknown"):
        return pred_lat is None and pred_lon is None
    elat, elon = expected.get("latitude"), expected.get("longitude")
    if elat is None or elon is None:
        return None
    if pred_lat is None or pred_lon is None:
        return False
    radius = float(expected.get("radius_km") or 2)
    return _haversine_km(float(pred_lat), float(pred_lon), float(elat), float(elon)) <= radius


def _distance_km(pred_lat, pred_lon, expected: dict[str, Any]) -> Optional[float]:
    elat, elon = expected.get("latitude"), expected.get("longitude")
    if None in (pred_lat, pred_lon, elat, elon):
        return None
    return round(_haversine_km(float(pred_lat), float(pred_lon), float(elat), float(elon)), 1)


def _confidence_band(conf: float, expected: dict[str, Any]) -> Optional[bool]:
    """Whether confidence sits inside the band this case demands.

    A separate check from accuracy: getting the place right but claiming 95%
    certainty on an ambiguous forest, or 90% on a generated image, is a
    calibration failure even though the coordinates pass.
    """
    low = expected.get("min_confidence")
    high = expected.get("max_confidence")
    if low is None and high is None:
        return None
    if low is not None and conf < float(low):
        return False
    if high is not None and conf > float(high):
        return False
    return True


def score_case(pred: dict[str, Any], expected: dict[str, Any]) -> dict[str, Any]:
    country = _country_hit(pred.get("country"), expected)
    region = _region_hit(pred.get("region"), expected)
    name = _name_hit(pred.get("location_name"), expected)
    within = _within_radius(pred.get("latitude"), pred.get("longitude"), expected)
    dist = _distance_km(pred.get("latitude"), pred.get("longitude"), expected)
    conf = float(pred.get("confidence") or 0.0)
    band = _confidence_band(conf, expected)
    # Brier vs "within radius" (or unknown-success). Skip if that metric is N/A.
    outcome = None
    if within is not None:
        outcome = 1.0 if within else 0.0
    elif expected.get("expect_unknown"):
        outcome = 1.0 if not pred.get("location_name") else 0.0
    brier = None if outcome is None else round((conf - outcome) ** 2, 4)
    return {
        "country": country,
        "region": region,
        "name": name,
        "within_radius": within,
        "distance_km": dist,
        "confidence": round(conf, 3),
        "confidence_band": band,
        "brier": brier,
        "location_name": pred.get("location_name"),
        "named": (pred.get("verification") or {}).get("named"),
    }


def _lookup_gazetteer(gazetteer: dict[str, Any], query: str) -> list[dict[str, Any]]:
    q = (query or "").strip().lower()
    if not q:
        return []
    hits: list[dict[str, Any]] = []
    for key, rec in gazetteer.items():
        k = key.lower()
        if k == q or k in q or q in k:
            hits.append(dict(rec))
    return hits[:3]


def install_offline_stubs(gazetteer: dict[str, Any], water_near: dict[str, Any]) -> None:
    """Replace network helpers so eval can run without Nominatim/Overpass/OpenAI."""
    import utils.landmarks as L
    import utils.osint as O

    def fake_forward(query: str, limit: int = 3) -> list[dict[str, Any]]:
        return _lookup_gazetteer(gazetteer, query)[: max(1, int(limit))]

    def fake_reverse(lat: float, lon: float) -> Optional[dict[str, Any]]:
        best = None
        best_d = 1e12
        for rec in gazetteer.values():
            d = _haversine_km(lat, lon, rec["latitude"], rec["longitude"])
            if d < best_d:
                best_d = d
                best = rec
        if best is None or best_d > 80:
            return {
                "name": "Unknown",
                "country": "United States",
                "region": None,
                "display_name": f"{lat}, {lon}",
            }
        return {
            "name": best.get("name"),
            "country": best.get("country"),
            "region": best.get("region"),
            "display_name": best.get("display_name"),
        }

    def fake_search(query: str, latitude: float, longitude: float, radius_km: float = 25, limit: int = 5):
        key = f"{round(float(latitude), 3)},{round(float(longitude), 3)}"
        # Also try slightly coarser keys used in labels.
        for k, hits in water_near.items():
            plat, plon = (float(x) for x in k.split(","))
            if _haversine_km(latitude, longitude, plat, plon) <= radius_km:
                return list(hits)[:limit]
        return list(water_near.get(key) or [])[:limit]

    def fake_verify(lat, lon, expected, radius=None):
        expected = expected or []
        return {
            "status": "verified" if expected else "skipped",
            "note": "ok (stub)",
            "expected": expected,
            "confirmed": list(expected),
            "missing": [],
            "context": [],
            "nearest_m": {c: 400.0 for c in expected},
            "anchors": {},
            "match_score": 1.0 if expected else 1.0,
            "radius_m": radius or 3000,
        }

    O.forward_geocode_candidates = fake_forward
    O.forward_geocode = lambda q: (fake_forward(q, 1) or [None])[0]
    O.reverse_geocode = fake_reverse
    O.verify_location = fake_verify
    O.enrich_location = lambda lat, lon: {"nearby_places": []}
    O.elevation_m = lambda lat, lon: None
    O.annual_climate = lambda lat, lon: None
    L.forward_geocode_candidates = fake_forward
    L.search_nearby = fake_search


def run_eval(live: bool = False) -> dict[str, Any]:
    from utils.osint import determine_location

    data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    if not live:
        install_offline_stubs(data.get("gazetteer") or {}, data.get("water_near") or {})

    rows: list[dict[str, Any]] = []
    for case in data.get("cases") or []:
        pred = determine_location(
            case.get("metadata") or {},
            case.get("vision") or {},
            forensics=case.get("forensics"),
        )
        scored = score_case(pred, case.get("expected") or {})
        scored["id"] = case["id"]
        scored["description"] = case.get("description", "")
        rows.append(scored)

    def rate(key: str) -> Optional[float]:
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return None
        return round(sum(1 for v in vals if v) / len(vals), 3)

    briers = [r["brier"] for r in rows if r["brier"] is not None]
    correct_conf = [r["confidence"] for r in rows if r.get("within_radius") is True]
    wrong_conf = [r["confidence"] for r in rows if r.get("within_radius") is False]
    summary = {
        "n": len(rows),
        "mode": "live" if live else "offline",
        "country_accuracy": rate("country"),
        "region_accuracy": rate("region"),
        "name_accuracy": rate("name"),
        "within_radius_accuracy": rate("within_radius"),
        "confidence_band_accuracy": rate("confidence_band"),
        "mean_brier": round(sum(briers) / len(briers), 4) if briers else None,
        "mean_confidence_when_correct": round(sum(correct_conf) / len(correct_conf), 3)
        if correct_conf
        else None,
        "mean_confidence_when_wrong": round(sum(wrong_conf) / len(wrong_conf), 3)
        if wrong_conf
        else None,
        "cases": rows,
    }
    summary["calibration_gap"] = None
    if summary["mean_confidence_when_correct"] is not None and summary[
        "mean_confidence_when_wrong"
    ] is not None:
        summary["calibration_gap"] = round(
            summary["mean_confidence_when_correct"] - summary["mean_confidence_when_wrong"],
            3,
        )
    return summary


def _print_human(summary: dict[str, Any]) -> None:
    print(f"Eval ({summary['mode']})  n={summary['n']}")
    print(
        f"  country={summary['country_accuracy']}  region={summary['region_accuracy']}  "
        f"name={summary['name_accuracy']}  within_radius={summary['within_radius_accuracy']}"
    )
    print(
        f"  conf_band={summary['confidence_band_accuracy']}  "
        f"mean Brier={summary['mean_brier']}  "
        f"conf|correct={summary['mean_confidence_when_correct']}  "
        f"conf|wrong={summary['mean_confidence_when_wrong']}  "
        f"gap={summary['calibration_gap']}"
    )
    print()
    print(f"{'id':<28} {'ok?':<5} {'band':<6} {'dist_km':<8} {'conf':<6} {'name'}")
    for row in summary["cases"]:
        ok = "Y" if row.get("within_radius") else ("-" if row.get("within_radius") is None else "N")
        if row.get("within_radius") is None and row.get("country"):
            ok = "c"
        band = {True: "ok", False: "OUT", None: "-"}[row.get("confidence_band")]
        dist = "-" if row["distance_km"] is None else str(row["distance_km"])
        print(
            f"{row['id']:<28} {ok:<5} {band:<6} {dist:<8} {row['confidence']:<6} "
            f"{row.get('location_name')}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the location-eval harness.")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use real Nominatim/Overpass (still uses fixture vision, no OpenAI).",
    )
    parser.add_argument("--json", action="store_true", help="Print JSON instead of a table.")
    args = parser.parse_args()
    summary = run_eval(live=args.live)
    if args.json:
        # Drop bulky named blocks from JSON default print? Keep them; useful.
        print(json.dumps(summary, indent=2, default=str))
    else:
        _print_human(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
