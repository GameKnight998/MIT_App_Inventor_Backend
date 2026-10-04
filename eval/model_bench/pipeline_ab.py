"""A/B: does adding open-model candidates improve the real pipeline?

Both arms run `determine_location` on the SAME cached GPT-4o scene analysis, so
the only difference is the candidate list:

  baseline  GPT-4o's candidates only (today's production behaviour)
  ensemble  GPT-4o's candidates + one candidate each from the open models
            (appended after GPT-4o's, at a fixed modest confidence)

Street-level refinement is skipped in both arms (it would make a second,
non-deterministic GPT-4o call). Needs score.py to have been run first, since
open-model coordinates come from summary.json.

  python eval/model_bench/pipeline_ab.py [--models geoclip,gaea,globe] [--conf 0.3]
"""

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from score import THRESHOLDS, haversine_km, load_results  # noqa: E402
from utils.osint import determine_location  # noqa: E402
from utils.vision import analyze_image  # noqa: E402

VISION_CACHE = HERE / "vision_cache.json"


def _vision(rows: list[dict]) -> dict[str, dict]:
    cache = json.loads(VISION_CACHE.read_text(encoding="utf-8")) if VISION_CACHE.exists() else {}
    for row in rows:
        if row["id"] not in cache:
            cache[row["id"]] = analyze_image(str(HERE / "images" / row["file"]))
            VISION_CACHE.write_text(json.dumps(cache, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"vision {row['id']}: cached")
    return cache


def _extra_candidates(image_id: str, models: list[str], points: dict, raw: dict, conf: float) -> list[dict]:
    extra = []
    for model in models:
        point = points.get(model, {}).get(image_id)
        if not point:
            continue
        answer = (raw.get(model, {}).get(image_id) or {}).get("answer") or ""
        name = None if model == "geoclip" else answer or None
        country = answer.split(",")[-1].strip() if name and "," in answer else None
        extra.append(
            {
                "name": name,
                "country": country,
                "latitude": point[0],
                "longitude": point[1],
                "confidence": conf,
                "source_model": model,
            }
        )
    return extra


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="geoclip,gaea,globe")
    ap.add_argument("--conf", type=float, default=0.3)
    args = ap.parse_args()
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    rows = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))
    points = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))["points"]
    raw = load_results()
    vision = _vision(rows)

    errors: dict[str, dict[str, float]] = {"baseline": {}, "ensemble": {}}
    for row in rows:
        base_vision = vision[row["id"]]
        aug_vision = copy.deepcopy(base_vision)
        aug_vision["candidates"] = list(aug_vision.get("candidates") or []) + _extra_candidates(
            row["id"], models, points, raw, args.conf
        )
        for arm, v in (("baseline", base_vision), ("ensemble", aug_vision)):
            out = determine_location({}, copy.deepcopy(v), None, image_path=None)
            lat, lon = out.get("latitude"), out.get("longitude")
            err = haversine_km(row["latitude"], row["longitude"], lat, lon) if lat is not None else 20000.0
            errors[arm][row["id"]] = round(err, 1)
        b, e = errors["baseline"][row["id"]], errors["ensemble"][row["id"]]
        print(f"{row['id']:<20} baseline {b:>9.1f} km   ensemble {e:>9.1f} km   {'better' if e < b * 0.8 else 'worse' if e > b * 1.25 else 'same'}")

    print(f"\nmodels added: {', '.join(models)} at confidence {args.conf}")
    for arm, per in errors.items():
        km = list(per.values())
        accs = "  ".join(f"@{t}: {sum(x <= t for x in km) / len(km):.0%}" for t in THRESHOLDS)
        print(f"{arm:<9} median {statistics.median(km):>7.0f} km  {accs}")
    better = sum(errors["ensemble"][i] < errors["baseline"][i] * 0.8 for i in errors["baseline"])
    worse = sum(errors["ensemble"][i] > errors["baseline"][i] * 1.25 for i in errors["baseline"])
    print(f"ensemble better on {better}, worse on {worse}, same on {len(rows) - better - worse}")
    (HERE / f"ab_{'_'.join(models)}_{args.conf}.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
