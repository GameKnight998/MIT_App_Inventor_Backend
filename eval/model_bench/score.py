"""Score every model on the remote-terrain benchmark against GPT-4o.

Turns each model's raw output into one coordinate (JSON lat/lon, GeoCLIP's top
prediction, or a geocoded place name for text-only models), measures the
great-circle error to the Commons camera location, and reports:

  standalone accuracy   median error and share within 25/200/750/2500 km
  complement to GPT-4o  how often the model is right (<= 200 km) when GPT-4o
                        is not, and the best-of-two median error
  best trio             greedy pick of the 3 models that most often put at
                        least one candidate within 200 km alongside GPT-4o

  python eval/model_bench/score.py [--json summary.json]
"""

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from utils import geocode  # noqa: E402

THRESHOLDS = (25, 200, 750, 2500)
HIT_KM = 200
BASE = "gpt4o"
GEOCODE_CACHE = HERE / "geocode_cache.json"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    a = (
        math.sin((p2 - p1) / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lon2 - lon1) / 2) ** 2
    )
    return 2 * 6371.0 * math.asin(math.sqrt(a))


class Geocoder:
    """Nominatim with a JSON cache, retrying ever-coarser versions of a place."""

    def __init__(self) -> None:
        self.cache = (
            json.loads(GEOCODE_CACHE.read_text(encoding="utf-8")) if GEOCODE_CACHE.exists() else {}
        )

    def __call__(self, parts: list[str]) -> tuple[float, float] | None:
        parts = [p.strip() for p in parts if p and p.strip() and p.strip().lower() not in ("unknown", "n/a")]
        for i in range(len(parts)):
            query = ", ".join(parts[i:])
            if query not in self.cache:
                hit = geocode.forward_geocode(query)
                geocode.polite_pause()
                if not hit:
                    # A miss may be a transient rate limit, so it is not persisted.
                    continue
                self.cache[query] = [hit["latitude"], hit["longitude"]]
                GEOCODE_CACHE.write_text(json.dumps(self.cache, indent=1, ensure_ascii=False), encoding="utf-8")
            return tuple(self.cache[query])
        return None


def _float(value) -> float | None:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def coordinates(model: str, result: dict, geocoder: Geocoder) -> tuple[float, float] | None:
    """One (lat, lon) per model output, or None if the model gave nothing usable."""
    if not result or result.get("error"):
        return None
    parsed = result.get("parsed") or {}
    if model == "geoclip":
        best = (parsed.get("predictions") or [None])[0]
        return (best["latitude"], best["longitude"]) if best else None

    lat, lon = _float(parsed.get("latitude")), _float(parsed.get("longitude"))
    if lat is not None and lon is not None and (lat, lon) != (0.0, 0.0):
        return lat, lon

    if parsed.get("FinalAnswer"):  # GeoAgent: "Country; Region; Specific Location"
        country, *rest = [p.strip() for p in str(parsed["FinalAnswer"]).split(";")]
        return geocoder(list(reversed(rest)) + [country])
    if any(parsed.get(k) for k in ("specific_location", "city", "region", "country")):
        return geocoder([str(parsed.get(k) or "") for k in ("specific_location", "city", "region", "country")])

    answer = result.get("answer") or ""
    if model == "gaea":
        answer = re.sub(r"(?i)^.*?\b(taken|located|captured) (in|at|near)\s+", "", answer)
        answer = re.split(r"(?i)[.;]|, specifically", answer)[0]
    return geocoder(answer.split(",")) if answer else None


def load_results() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in sorted(HERE.glob("results_*.json")):
        for model, per_image in json.loads(path.read_text(encoding="utf-8")).items():
            if not all(r.get("error") for r in per_image.values()):
                merged[model] = per_image
    return merged


def summarise(errors: dict[str, dict[str, float | None]], ids: list[str]) -> dict:
    summary = {}
    base = errors.get(BASE, {})
    for model, per_image in errors.items():
        km = [per_image.get(i) for i in ids]
        answered = [e for e in km if e is not None]
        filled = [e if e is not None else 20000.0 for e in km]  # a non-answer counts as a miss
        row = {
            "answered": len(answered),
            "median_km": round(statistics.median(filled)),
            "mean_km": round(statistics.mean(filled)),
            **{f"acc@{t}": round(sum(e <= t for e in filled) / len(ids), 3) for t in THRESHOLDS},
        }
        if model != BASE and base:
            base_km = [base.get(i) if base.get(i) is not None else 20000.0 for i in ids]
            row["rescues"] = sum(m <= HIT_KM < b for m, b in zip(filled, base_km))
            row["closer_than_gpt4o"] = sum(m < b for m, b in zip(filled, base_km))
            row["best_of_two_median_km"] = round(
                statistics.median(min(m, b) for m, b in zip(filled, base_km))
            )
        summary[model] = row
    return summary


def best_trio(errors: dict[str, dict], ids: list[str], exclude: set[str]) -> list[dict]:
    """Greedily add the model that most raises 'any candidate within HIT_KM'."""
    def filled(model: str) -> list[float]:
        return [errors[model].get(i) if errors[model].get(i) is not None else 20000.0 for i in ids]

    chosen: list[str] = []
    steps = []
    current = filled(BASE)
    pool = [m for m in errors if m != BASE and m not in exclude]
    for _ in range(3):
        def gain(model: str) -> tuple[int, float]:
            combo = [min(c, e) for c, e in zip(current, filled(model))]
            return sum(e <= HIT_KM for e in combo), -statistics.median(combo)

        pool = [m for m in pool if m not in chosen]
        if not pool:
            break
        pick = max(pool, key=gain)
        current = [min(c, e) for c, e in zip(current, filled(pick))]
        chosen.append(pick)
        steps.append(
            {
                "model": pick,
                "any_within_200km": round(sum(e <= HIT_KM for e in current) / len(ids), 3),
                "best_of_median_km": round(statistics.median(current)),
            }
        )
    return steps


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write the summary here as JSON too")
    args = ap.parse_args()

    manifest = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    truth = {r["id"]: (r["latitude"], r["longitude"]) for r in manifest}
    ids = [r["id"] for r in manifest]
    results = load_results()
    geocoder = Geocoder()

    errors: dict[str, dict[str, float | None]] = {}
    points: dict[str, dict[str, list[float] | None]] = {}
    for model, per_image in results.items():
        errors[model], points[model] = {}, {}
        for i in ids:
            coord = coordinates(model, per_image.get(i), geocoder)
            points[model][i] = list(coord) if coord else None
            errors[model][i] = round(haversine_km(*truth[i], *coord), 1) if coord else None

    summary = summarise(errors, ids)
    order = sorted(summary, key=lambda m: (-summary[m]["acc@200"], summary[m]["median_km"]))
    cols = ["answered", "median_km", *[f"acc@{t}" for t in THRESHOLDS], "rescues", "closer_than_gpt4o", "best_of_two_median_km"]
    print(f"{len(ids)} remote-terrain images; misses count as 20,000 km\n")
    print(f"{'model':<14}" + "".join(f"{c:>12}" for c in cols))
    for model in order:
        print(f"{model:<14}" + "".join(f"{str(summary[model].get(c, '-')):>12}" for c in cols))

    trio = best_trio(errors, ids, exclude={"api_pipeline"})
    print("\nGreedy trio alongside GPT-4o (any candidate within 200 km):")
    for step in trio:
        print(f"  + {step['model']:<12} -> {step['any_within_200km']:.0%} covered, best-of median {step['best_of_median_km']} km")

    print("\nPer-image error (km):")
    print(f"{'image':<20}" + "".join(f"{m[:10]:>11}" for m in order))
    for i in ids:
        cells = "".join(
            f"{'-' if errors[m].get(i) is None else round(errors[m][i]):>11}" for m in order
        )
        print(f"{i:<20}{cells}")

    if args.json:
        Path(args.json).write_text(
            json.dumps({"summary": summary, "trio": trio, "errors": errors, "points": points}, indent=2),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
