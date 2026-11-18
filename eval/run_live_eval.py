"""Full-pipeline live geolocation eval on real photos (eval/live_labels.json).

  python eval/run_live_eval.py --split dev
  python eval/run_live_eval.py --split holdout
  python eval/run_live_eval.py --split all --repeats 3

Vision responses are cached under eval/cache/ by (image hash, model, prompt
hash, settings) so later OSINT-only changes do not re-bill the vision API.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

LABELS = ROOT / "eval" / "live_labels.json"
IMAGES = ROOT / "eval" / "images"
CACHE = ROOT / "eval" / "cache"
RESULTS = ROOT / "eval" / "results"

THRESHOLDS_KM = (1, 25, 200, 750, 2500)


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
            ).strip()
        )
    except Exception:
        return "unknown"


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()[:16]


def _prompt_sha() -> str:
    from utils import vision as V

    blob = (getattr(V, "_SYSTEM_PROMPT", "") or "") + (getattr(V, "_ROUTER_SYSTEM", "") or "")
    return hashlib.sha256(blob.encode()).hexdigest()[:12]


def _cache_key(image_path: Path) -> str:
    from utils.visionclient import model_name, provider

    settings = f"{provider()}|{model_name()}|{os.getenv('VISION_MAX_TOKENS','')}"
    raw = f"{_file_sha(image_path)}|{model_name()}|{_prompt_sha()}|{settings}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _load_cache(key: str) -> Optional[dict]:
    path = CACHE / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _save_cache(key: str, vision: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / f"{key}.json").write_text(
        json.dumps(vision, ensure_ascii=False), encoding="utf-8"
    )


def _analyze_cached(image_path: Path) -> tuple[int, dict[str, Any]]:
    """Same path as /analyze, but reuse cached vision JSON when possible."""
    import main
    from utils.exif import extract_exif
    from utils.forensics import assess_forensics
    from utils.osint import apply_vehicle_signal, determine_location
    from utils.preprocess import enhance_image
    from utils.response import build_response
    from utils.vehicle import identify_vehicles, should_identify
    from utils.vision import analyze_image
    from utils.visionclient import model_name, provider
    from utils.cache import parallel_map

    data = image_path.read_bytes()
    # Persist through main's temp path logic by writing a temp file.
    suffix = image_path.suffix or ".jpg"
    saved = main.UPLOAD_DIR / f"live_{_file_sha(image_path)}{suffix}"
    saved.write_bytes(data)
    enhanced_path = None
    try:
        metadata = extract_exif(str(saved))
        # Blind eval: drop any residual GPS.
        if metadata.get("gps"):
            metadata = {**metadata, "gps": None}
        forensics = assess_forensics(str(saved), "JPEG", metadata)
        quality = enhance_image(str(saved))
        vision_path = quality["path"]
        if quality.get("enhanced"):
            enhanced_path = vision_path

        key = _cache_key(Path(vision_path) if Path(vision_path).exists() else saved)
        # Prefer hashing the original bytes for stable keys across enhance toggles.
        key = _cache_key(saved)
        cached = _load_cache(key)
        if cached is not None:
            vision = cached
        else:
            vision = analyze_image(vision_path)
            vision["_cache_meta"] = {
                "provider": provider(),
                "model": model_name(),
                "cached_at": datetime.now(timezone.utc).isoformat(),
            }
            _save_cache(key, vision)

        jobs = [
            lambda: determine_location(
                metadata, vision, forensics, image_path=vision_path
            )
        ]
        wants_vehicle = should_identify(vision)
        if wants_vehicle:
            jobs.append(lambda: identify_vehicles(vision_path))
        outputs = parallel_map(lambda fn: fn(), jobs)
        location = outputs[0]
        vehicle = outputs[1] if wants_vehicle and len(outputs) > 1 else None
        apply_vehicle_signal(location, vehicle)
        payload = build_response(
            filename=image_path.name,
            metadata=metadata,
            vision=vision,
            location=location,
            forensics=forensics,
            image_quality={k: v for k, v in quality.items() if k != "path"},
        )
        payload["vision_provider"] = provider()
        payload["vision_model"] = model_name()
        payload["vision_cached"] = cached is not None
        return 200, payload
    except Exception as exc:
        return 500, {"success": False, "error": str(exc), "filename": image_path.name}
    finally:
        try:
            saved.unlink(missing_ok=True)
        except Exception:
            pass
        if enhanced_path and enhanced_path != str(saved):
            try:
                os.unlink(enhanced_path)
            except OSError:
                pass


def _brier(pairs: list[tuple[float, int]]) -> Optional[float]:
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def _ece(pairs: list[tuple[float, int]], bins: int = 10) -> Optional[float]:
    if not pairs:
        return None
    total = 0.0
    n = len(pairs)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        bucket = [(p, y) for p, y in pairs if (lo <= p < hi) or (b == bins - 1 and p == 1.0)]
        if not bucket:
            continue
        conf = sum(p for p, _ in bucket) / len(bucket)
        acc = sum(y for _, y in bucket) / len(bucket)
        total += (len(bucket) / n) * abs(acc - conf)
    return total


def _country_of(lat: float, lon: float) -> Optional[str]:
    try:
        from utils.geocode import reverse_geocode

        rev = reverse_geocode(lat, lon)
        return (rev or {}).get("country")
    except Exception:
        return None


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [r["error_km"] for r in rows if r.get("error_km") is not None]
    unanswered = sum(1 for r in rows if r.get("error_km") is None)
    api_errors = sum(1 for r in rows if r.get("api_error"))
    n = len(rows) or 1

    def rate(thresh: float) -> float:
        return sum(1 for e in errors if e <= thresh) / len(rows)

    cal_pairs = [
        (float(r["confidence"]), 1 if r.get("error_km") is not None and r["error_km"] <= 25 else 0)
        for r in rows
        if r.get("confidence") is not None
    ]
    country_ok = [
        r for r in rows if r.get("country_match") is not None
    ]
    by_cat: dict[str, list] = {}
    for r in rows:
        by_cat.setdefault(r.get("category") or "other", []).append(r)

    cat_stats = {}
    for cat, items in by_cat.items():
        ce = [x["error_km"] for x in items if x.get("error_km") is not None]
        cat_stats[cat] = {
            "n": len(items),
            "median_km": round(statistics.median(ce), 3) if ce else None,
            "acc_25km": round(sum(1 for e in ce if e <= 25) / len(items), 3) if items else None,
        }

    return {
        "n": len(rows),
        "answered": len(errors),
        "unanswered_rate": round(unanswered / n, 3),
        "api_error_rate": round(api_errors / n, 3),
        "median_km": round(statistics.median(errors), 3) if errors else None,
        "mean_km": round(statistics.fmean(errors), 3) if errors else None,
        "accuracy": {f"{t}km": round(rate(t), 3) for t in THRESHOLDS_KM},
        "country_accuracy": (
            round(sum(1 for r in country_ok if r["country_match"]) / len(country_ok), 3)
            if country_ok
            else None
        ),
        "mean_confidence": (
            round(statistics.fmean(r["confidence"] for r in rows if r.get("confidence") is not None), 3)
            if any(r.get("confidence") is not None for r in rows)
            else None
        ),
        "brier_25km": round(_brier(cal_pairs), 4) if cal_pairs else None,
        "ece_25km": round(_ece(cal_pairs), 4) if cal_pairs else None,
        "by_category": cat_stats,
    }


def run_split(split: str, limit: Optional[int] = None) -> dict[str, Any]:
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    if split != "all":
        labels = [r for r in labels if r.get("split") == split]
    if limit:
        labels = labels[:limit]

    from utils.visionclient import model_name, provider

    rows = []
    for i, lab in enumerate(labels, 1):
        path = IMAGES / lab["file"]
        if not path.exists():
            rows.append(
                {
                    "id": lab["id"],
                    "file": lab["file"],
                    "category": lab.get("category"),
                    "api_error": True,
                    "error": "missing image",
                    "error_km": None,
                    "confidence": None,
                }
            )
            continue
        t0 = time.monotonic()
        status, payload = _analyze_cached(path)
        dt = time.monotonic() - t0
        lat, lon = payload.get("latitude"), payload.get("longitude")
        err = None
        if lat is not None and lon is not None:
            err = round(_haversine_km(lab["latitude"], lab["longitude"], lat, lon), 3)
        pred_country = payload.get("country")
        truth_country = lab.get("country")
        if truth_country is None and err is not None:
            truth_country = _country_of(lab["latitude"], lab["longitude"])
            lab["country"] = truth_country  # cache into memory for this run
        country_match = None
        if truth_country and pred_country and pred_country not in (None, "Unknown"):
            country_match = truth_country.lower() in str(pred_country).lower() or str(
                pred_country
            ).lower() in truth_country.lower()

        row = {
            "id": lab["id"],
            "file": lab["file"],
            "category": lab.get("category"),
            "split": lab.get("split"),
            "true_lat": lab["latitude"],
            "true_lon": lab["longitude"],
            "pred_lat": lat,
            "pred_lon": lon,
            "error_km": err,
            "confidence": payload.get("confidence"),
            "location_name": payload.get("location_name"),
            "country_pred": pred_country,
            "country_true": truth_country,
            "country_match": country_match,
            "verified": payload.get("verified"),
            "vision_model": payload.get("vision_model") or model_name(),
            "vision_provider": payload.get("vision_provider") or provider(),
            "vision_cached": payload.get("vision_cached"),
            "api_error": status != 200 or not payload.get("success", True),
            "error": payload.get("error"),
            "seconds": round(dt, 1),
        }
        rows.append(row)
        mark = f"{err:.1f} km" if err is not None else "FAIL"
        print(
            f"[{i}/{len(labels)}] {lab['file']} {lab.get('category')} -> {mark} "
            f"conf={row['confidence']} {row['vision_model']} "
            f"{'(cache)' if row['vision_cached'] else ''} {dt:.0f}s",
            flush=True,
        )

    summary = summarise(rows)
    return {
        "split": split,
        "git_sha": _git_sha(),
        "vision_provider": provider(),
        "vision_model": model_name(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "cases": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "holdout", "all"), default="dev")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    if not LABELS.exists() or not IMAGES.exists():
        print("Missing live labels/images. Run: python eval/build_live_bench.py", file=sys.stderr)
        return 2

    RESULTS.mkdir(parents=True, exist_ok=True)
    runs = []
    for r in range(args.repeats):
        print(f"\n=== repeat {r + 1}/{args.repeats} split={args.split} ===", flush=True)
        runs.append(run_split(args.split, args.limit))

    # Aggregate means across repeats for headline metrics.
    def mean_of(key_path: list[str]) -> Optional[float]:
        vals = []
        for run in runs:
            cur: Any = run
            for k in key_path:
                cur = cur.get(k) if isinstance(cur, dict) else None
            if isinstance(cur, (int, float)):
                vals.append(float(cur))
        return round(statistics.mean(vals), 4) if vals else None

    aggregate = {
        "repeats": args.repeats,
        "split": args.split,
        "git_sha": _git_sha(),
        "mean": {
            "acc_1km": mean_of(["summary", "accuracy", "1km"]),
            "acc_25km": mean_of(["summary", "accuracy", "25km"]),
            "acc_200km": mean_of(["summary", "accuracy", "200km"]),
            "acc_750km": mean_of(["summary", "accuracy", "750km"]),
            "median_km": mean_of(["summary", "median_km"]),
            "country_accuracy": mean_of(["summary", "country_accuracy"]),
            "ece_25km": mean_of(["summary", "ece_25km"]),
            "brier_25km": mean_of(["summary", "brier_25km"]),
            "mean_confidence": mean_of(["summary", "mean_confidence"]),
        },
        "runs": runs,
    }

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = args.out or (RESULTS / f"{stamp}_{_git_sha()}_{args.split}.json")
    out.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")

    m = aggregate["mean"]
    print("\n=== SUMMARY ===")
    print(
        f"split={args.split} n≈{runs[0]['summary']['n']} model={runs[0].get('vision_model')} "
        f"sha={aggregate['git_sha']}"
    )
    print(
        f"acc@1={m['acc_1km']}  @25={m['acc_25km']}  @200={m['acc_200km']}  "
        f"@750={m['acc_750km']}  median_km={m['median_km']}"
    )
    print(
        f"country={m['country_accuracy']}  ECE@25={m['ece_25km']}  "
        f"Brier@25={m['brier_25km']}  mean_conf={m['mean_confidence']}"
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
