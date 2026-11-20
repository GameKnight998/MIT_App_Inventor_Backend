"""Benchmark photo geolocation across AI models.

Runs every photo in a manifest through the full pipeline once per model, with
the photo's own GPS ignored, and measures how far each answer lands from the
true location. Results are appended to a CSV as they arrive, so an interrupted
run resumes where it stopped.

The headline metric is the standard one in photo-geolocation research: the
share of photos placed within 1 km (street), 25 km (city), 200 km (region),
750 km (country) and 2,500 km (continent) of the truth, plus 5 km (district)
and 10 km (town), which tell a neighbouring city apart from the right one.

Manifest (CSV, one row per photo):
    file,latitude,longitude,country,place,category
    eiffel.jpg,48.8584,2.2945,France,Eiffel Tower,landmark
    farm_road.jpg,44.05,-121.31,United States,Bend OR,rural

`file` is relative to the manifest's folder. `category` is free text used to
break results down (e.g. landmark / urban / rural / nature / indoor).

Usage:
  python eval/benchmark.py run --manifest photos/manifest.csv \\
      --models gpt-4o,gpt-6.1-terra,claude-opus-5-5,gemini-3-pro --out results/
  python eval/benchmark.py report --out results/

`run` also writes the report at the end. Outputs in --out:
  results.csv          one row per photo x model (raw data)
  summary_by_model.csv accuracy, error and calibration per model
  summary_by_category.csv  accuracy at 25 km per model x category
  accuracy_by_threshold.png, median_error.png, calibration.png  slide charts
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent
# BENCH_CODE_DIR runs another checkout of the pipeline (e.g. the original
# version) so before/after results come from the same harness and photos.
import os  # noqa: E402

CODE_DIR = Path(os.getenv("BENCH_CODE_DIR") or ROOT).resolve()
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

# 5 and 10 km separate "the right town" from "the nearest big city": when every
# photo comes from one suburb, 25 km cannot tell Bellevue from Seattle.
THRESHOLDS_KM = (1, 5, 10, 25, 200, 750, 2500)
THRESHOLD_LABELS = ("Street\n1 km", "District\n5 km", "Town\n10 km", "City\n25 km", "Region\n200 km", "Country\n750 km", "Continent\n2,500 km")
# An answer counts as "correct" for calibration when within this distance.
CORRECT_KM = 25

FIELDS = [
    "file", "category", "model", "true_lat", "true_lon", "true_country", "true_place",
    "pred_place", "pred_country", "pred_lat", "pred_lon", "error_km", "country_match",
    "confidence", "conf_1km", "conf_25km", "conf_750km", "ai_top_place", "ai_top_error_km", "ai_raw_confidence",
    "agreement", "samples", "map_check", "seconds", "error",
]


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _num(value: Any) -> Optional[float]:
    try:
        f = float(value)
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _err(lat: Any, lon: Any, tlat: float, tlon: float) -> Optional[float]:
    lat, lon = _num(lat), _num(lon)
    if lat is None or lon is None:
        return None
    return round(haversine_km(lat, lon, tlat, tlon), 2)


def _country_match(pred: Optional[str], truth: str) -> Optional[bool]:
    p, t = (pred or "").strip().lower(), (truth or "").strip().lower()
    if not t:
        return None
    return bool(p) and (p in t or t in p)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            lat, lon = _num(row.get("latitude")), _num(row.get("longitude"))
            if not row.get("file") or lat is None or lon is None:
                print(f"Skipping manifest row without file/latitude/longitude: {row}")
                continue
            row["path"] = (path.parent / row["file"]).resolve()
            row["lat"], row["lon"] = lat, lon
            rows.append(row)
    return rows


def run_one(photo: dict[str, Any], model: str) -> dict[str, Any]:
    from utils.exif import extract_exif
    from utils.forensics import assess_forensics
    from utils.osint import determine_location
    from utils.vision import analyze_image

    out: dict[str, Any] = {
        "file": photo["file"], "category": photo.get("category", ""), "model": model,
        "true_lat": photo["lat"], "true_lon": photo["lon"],
        "true_country": photo.get("country", ""), "true_place": photo.get("place", ""),
    }
    start = time.monotonic()
    try:
        path = str(photo["path"])
        metadata = {**extract_exif(path), "gps": None}  # judge the pixels, not EXIF
        forensics = assess_forensics(path, None, metadata)
        try:
            vision = analyze_image(path, model=model)
        except TypeError:  # older pipeline without per-call model choice
            import utils.vision as vision_module

            vision_module.VISION_MODEL = model
            vision = analyze_image(path)
        if not vision.get("available"):
            raise RuntimeError(vision.get("note") or "vision unavailable")
        loc = determine_location(metadata, vision, forensics)
    except Exception as exc:  # record and keep going
        out.update(seconds=round(time.monotonic() - start, 1), error=str(exc)[:300])
        return out

    top = (vision.get("candidates") or [{}])[0]
    consensus = vision.get("consensus") or {}
    out.update(
        pred_place=loc.get("location_name"),
        pred_country=loc.get("country"),
        pred_lat=loc.get("latitude"),
        pred_lon=loc.get("longitude"),
        error_km=_err(loc.get("latitude"), loc.get("longitude"), photo["lat"], photo["lon"]),
        country_match=_country_match(loc.get("country"), photo.get("country", "")),
        confidence=round(float(loc.get("confidence") or 0.0), 3),
        conf_1km=(loc.get("confidence_by_distance") or {}).get("1km"),
        conf_25km=(loc.get("confidence_by_distance") or {}).get("25km"),
        conf_750km=(loc.get("confidence_by_distance") or {}).get("750km"),
        # The AI's own top answer before map checks: shows what the pipeline adds.
        ai_top_place=top.get("name"),
        ai_top_error_km=_err(top.get("latitude"), top.get("longitude"), photo["lat"], photo["lon"]),
        ai_raw_confidence=top.get("confidence"),
        agreement=consensus.get("top_agreement"),
        samples=consensus.get("samples"),
        map_check=(loc.get("verification") or {}).get("status"),
        seconds=round(time.monotonic() - start, 1),
        error="",
    )
    return out


def run(manifest: Path, models: list[str], out_dir: Path, label: str = "") -> None:
    photos = load_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.csv"
    done: set[tuple[str, str]] = set()
    if results_path.exists():
        with results_path.open(newline="", encoding="utf-8") as f:
            done = {(r["file"], r["model"]) for r in csv.DictReader(f) if not r.get("error")}

    new_file = not results_path.exists()
    with results_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        total = len(photos) * len(models)
        i = 0
        for photo in photos:
            for model in models:
                i += 1
                if (photo["file"], model + label) in done:
                    continue
                row = run_one(photo, model)
                row["model"] = model + label
                writer.writerow(row)
                f.flush()
                status = row.get("error") or (
                    f"{row.get('pred_place')} | {row.get('error_km')} km | "
                    f"conf {row.get('confidence')} | {row.get('seconds')} s"
                )
                print(f"[{i}/{total}] {photo['file']} x {model}: {status}")
    report(out_dir)


# --- reporting --------------------------------------------------------------


def _read_results(out_dir: Path) -> list[dict[str, Any]]:
    with (out_dir / "results.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    # Keep the latest row per photo x model (a failed row may have been retried).
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        latest[(r["file"], r["model"])] = r
    return list(latest.values())


def _conf_25(r: dict[str, Any]) -> float:
    """Confidence that the answer is within CORRECT_KM.

    The improved pipeline's headline confidence means "this exact spot", so
    compare like with like using its within-25 km estimate when present.
    The original pipeline has only one number, which is used as is.
    """
    v = _num(r.get("conf_25km"))
    return v if v is not None else (_num(r.get("confidence")) or 0.0)


def _rate(values: list[bool]) -> Optional[float]:
    return round(sum(values) / len(values), 3) if values else None


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    models = list(dict.fromkeys(r["model"] for r in rows))
    out = []
    for model in models:
        mine = [r for r in rows if r["model"] == model]
        # A failed or unanswered photo counts as a miss, never as excluded.
        errs = [_num(r["error_km"]) for r in mine]
        dist = [e if e is not None else float("inf") for e in errs]
        ai_errs = [_num(r["ai_top_error_km"]) for r in mine]
        ai_dist = [e if e is not None else float("inf") for e in ai_errs]
        conf = [_conf_25(r) for r in mine]
        correct = [d <= CORRECT_KM for d in dist]
        row: dict[str, Any] = {"model": model, "photos": len(mine), "failed": sum(1 for r in mine if r.get("error"))}
        for t in THRESHOLDS_KM:
            row[f"acc_{t}km"] = _rate([d <= t for d in dist])
        for t in (25, 750):
            row[f"ai_only_acc_{t}km"] = _rate([d <= t for d in ai_dist])
        finite = [d for d in dist if math.isfinite(d)]
        row["median_error_km"] = round(statistics.median(dist), 1) if finite and len(finite) * 2 > len(dist) else None
        row["country_accuracy"] = _rate([r["country_match"] == "True" for r in mine])
        row["mean_confidence"] = round(statistics.mean(conf), 3) if conf else None
        right = [c for c, ok in zip(conf, correct) if ok]
        wrong = [c for c, ok in zip(conf, correct) if not ok]
        row["confidence_when_correct"] = round(statistics.mean(right), 3) if right else None
        row["confidence_when_wrong"] = round(statistics.mean(wrong), 3) if wrong else None
        row["brier_25km"] = round(statistics.mean((c - ok) ** 2 for c, ok in zip(conf, correct)), 4) if conf else None
        secs = [_num(r["seconds"]) for r in mine if _num(r["seconds"]) is not None]
        row["median_seconds"] = round(statistics.median(secs), 1) if secs else None
        out.append(row)
    return out


def by_category(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for model in dict.fromkeys(r["model"] for r in rows):
        for cat in sorted({r.get("category") or "uncategorised" for r in rows}):
            mine = [r for r in rows if r["model"] == model and (r.get("category") or "uncategorised") == cat]
            if not mine:
                continue
            dist = [(_num(r["error_km"]) if _num(r["error_km"]) is not None else float("inf")) for r in mine]
            out.append({"model": model, "category": cat, "photos": len(mine),
                        f"acc_{CORRECT_KM}km": _rate([d <= CORRECT_KM for d in dist]),
                        "acc_750km": _rate([d <= 750 for d in dist])})
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# Validated categorical palette (fixed order; a model keeps its colour).
_SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
_INK, _MUTED, _GRID, _SURFACE = "#1f1f1e", "#6b6a64", "#e4e3dd", "#ffffff"


def _style(ax: Any) -> None:
    ax.set_facecolor(_SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.tick_params(colors=_MUTED, length=0)
    ax.yaxis.grid(True, color=_GRID, linewidth=1)
    ax.set_axisbelow(True)


def _chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i:i + size] for i in range(0, len(items), size)] or [[]]


def _footnote(fig: Any, skipped: list[str]) -> None:
    if skipped:
        fig.text(0.01, 0.01, "Not charted (no answers, every run failed): " + ", ".join(skipped),
                 fontsize=10, color=_MUTED, ha="left", va="bottom")


def charts(summary: list[dict[str, Any]], rows: list[dict[str, Any]], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 13, "font.family": "DejaVu Sans", "text.color": _INK,
                         "axes.labelcolor": _MUTED, "figure.dpi": 150})
    # A model that never answered (e.g. refused every photo) has no accuracy or
    # confidence to show: drawing it as 0% bars would read as "always wrong".
    # It stays in the summary CSVs and the printed table, and is named in a footnote.
    skipped = [s["model"] for s in summary if s["failed"] >= s["photos"]]
    summary = [s for s in summary if s["failed"] < s["photos"]]
    for old in list(out_dir.glob("accuracy_by_threshold*.png")) + list(out_dir.glob("calibration*.png")):
        old.unlink()
    if not summary:
        return
    # The palette has 8 validated hues and a colour is never reused within a
    # chart, so more than 8 models are split into separate charts.
    groups = _chunks(summary, len(_SERIES))
    if len(groups) > 1:
        print(f"Note: {len(summary)} models; multi-series charts split into {len(groups)} parts of up to {len(_SERIES)}.")

    for part, group in enumerate(groups, 1):
        suffix = f"_part{part}" if len(groups) > 1 else ""
        color = {s["model"]: _SERIES[i] for i, s in enumerate(group)}
        n = len(group)

        # 1. Accuracy at distance thresholds (grouped bars).
        fig, ax = plt.subplots(figsize=(15, 6.6))
        _style(ax)
        width = 0.8 / max(n, 1)
        for i, s in enumerate(group):
            xs = [k + (i - (n - 1) / 2) * width for k in range(len(THRESHOLDS_KM))]
            ys = [100 * (s[f"acc_{t}km"] or 0) for t in THRESHOLDS_KM]
            ax.bar(xs, ys, width * 0.9, color=color[s["model"]], label=s["model"],
                   edgecolor=_SURFACE, linewidth=2)
            if s is group[0] or n <= 3:  # label one series' values, not every bar
                for x, y in zip(xs, ys):
                    ax.text(x, y + 1.2, f"{y:.0f}%", ha="center", va="bottom", fontsize=10, color=_INK)
        ax.set_xticks(range(len(THRESHOLDS_KM)), THRESHOLD_LABELS)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Photos located within distance (%)")
        title = "Share of photos located within each distance"
        ax.set_title(title + (f" (part {part} of {len(groups)})" if suffix else ""),
                     loc="left", fontsize=16, color=_INK, pad=14)
        ax.legend(frameon=False, ncol=min(n, 4), loc="upper left", bbox_to_anchor=(0, -0.13))
        fig.tight_layout(rect=(0, 0.04 if skipped else 0, 1, 1))
        _footnote(fig, skipped)
        fig.savefig(out_dir / f"accuracy_by_threshold{suffix}.png", facecolor=_SURFACE)
        plt.close(fig)

        # 4. Calibration: does a stated confidence match how often it is right?
        bins = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
        fig, ax = plt.subplots(figsize=(8, 7.4))
        _style(ax)
        ax.xaxis.grid(True, color=_GRID)
        ax.plot([0, 100], [0, 100], color=_MUTED, linewidth=1.5, linestyle="--", label="Perfect calibration")
        for s in group:
            m = s["model"]
            mine = [r for r in rows if r["model"] == m]
            xs, ys = [], []
            for lo, hi in bins:
                inside = [r for r in mine if lo <= _conf_25(r) < hi]
                if len(inside) < 3:  # too few photos to plot a rate
                    continue
                ok = [(_num(r["error_km"]) if _num(r["error_km"]) is not None else float("inf")) <= CORRECT_KM
                      for r in inside]
                xs.append(100 * statistics.mean(_conf_25(r) for r in inside))
                ys.append(100 * sum(ok) / len(ok))
            if xs:
                ax.plot(xs, ys, color=color[m], linewidth=2, marker="o", markersize=8,
                        markeredgecolor=_SURFACE, markeredgewidth=2, label=m)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_xlabel(f"Stated chance of being within {CORRECT_KM} km (%)")
        ax.set_ylabel(f"Actually within {CORRECT_KM} km (%)")
        ax.set_title("Confidence calibration" + (f" (part {part} of {len(groups)})" if suffix else ""),
                     loc="left", fontsize=16, color=_INK, pad=14)
        ax.legend(frameon=False, loc="upper left", fontsize=11)
        fig.tight_layout(rect=(0, 0.04 if skipped else 0, 1, 1))
        _footnote(fig, skipped)
        fig.savefig(out_dir / f"calibration{suffix}.png", facecolor=_SURFACE)
        plt.close(fig)

    # 2. Median error per model (single measure -> one hue, sorted).
    med = sorted((s for s in summary if s["median_error_km"] is not None), key=lambda s: s["median_error_km"])
    if med:
        fig, ax = plt.subplots(figsize=(10, 0.7 * len(med) + 2.1))
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=_GRID)
        vals = [s["median_error_km"] for s in med]
        ax.barh([s["model"] for s in med], vals, color=_SERIES[0], height=0.55)
        for y, v in enumerate(vals):
            ax.text(v, y, f"  {v:,.0f} km", va="center", fontsize=11, color=_INK)
        ax.invert_yaxis()
        ax.set_xlabel("Median distance from the true location (km), lower is better")
        ax.set_title("Median location error by model", loc="left", fontsize=16, color=_INK, pad=14)
        ax.set_xlim(0, max(vals) * 1.25 or 1)
        fig.tight_layout(rect=(0, 0.05 if skipped else 0, 1, 1))
        _footnote(fig, skipped)
        fig.savefig(out_dir / "median_error.png", facecolor=_SURFACE)
        plt.close(fig)

    # 3. Confidence on right vs wrong answers: a trustworthy score is high
    # when right and low when wrong. Horizontal bars keep long model names
    # readable however many models there are.
    sep = [s for s in summary if s["confidence_when_correct"] is not None]
    if sep:
        fig, ax = plt.subplots(figsize=(11, 1.0 * len(sep) + 2.4))
        _style(ax)
        ax.yaxis.grid(False)
        ax.xaxis.grid(True, color=_GRID)
        ys = list(range(len(sep)))
        h = 0.36
        right = [100 * (s["confidence_when_correct"] or 0) for s in sep]
        wrong = [100 * (s["confidence_when_wrong"] or 0) for s in sep]
        ax.barh([y - h / 2 for y in ys], right, h * 0.92, color=_SERIES[0], label=f"Answer within {CORRECT_KM} km")
        ax.barh([y + h / 2 for y in ys], wrong, h * 0.92, color=_SERIES[1], label=f"Answer more than {CORRECT_KM} km off")
        for y, (r_, w_, s_) in enumerate(zip(right, wrong, sep)):
            ax.text(r_ + 0.8, y - h / 2, f"{r_:.0f}%", va="center", fontsize=10, color=_INK)
            if s_["confidence_when_wrong"] is not None:
                ax.text(w_ + 0.8, y + h / 2, f"{w_:.0f}%", va="center", fontsize=10, color=_INK)
            else:
                ax.text(0.8, y + h / 2, "no misses", va="center", fontsize=9, color=_MUTED)
        ax.set_yticks(ys, [s["model"] for s in sep], fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, 105)
        ax.set_xlabel("Average stated confidence (%)")
        ax.set_title("Confidence when right vs. when wrong (bigger gap = more trustworthy)",
                     loc="left", fontsize=16, color=_INK, pad=14)
        ax.legend(frameon=False, ncol=2, loc="upper left", bbox_to_anchor=(0, -0.05 - 0.35 / len(sep)), fontsize=11)
        fig.tight_layout(rect=(0, 0.04 if skipped else 0, 1, 1))
        _footnote(fig, skipped)
        fig.savefig(out_dir / "confidence_right_vs_wrong.png", facecolor=_SURFACE)
        plt.close(fig)


def report(out_dir: Path) -> None:
    rows = _read_results(out_dir)
    if not rows:
        print("No results yet.")
        return
    summary = summarize(rows)
    _write_csv(out_dir / "summary_by_model.csv", summary)
    _write_csv(out_dir / "summary_by_category.csv", by_category(rows))
    charts(summary, rows, out_dir)

    pct = lambda v: "-" if v is None else f"{100 * v:.0f}%"
    head = "".join(f"{str(t) + 'km':>7}" for t in THRESHOLDS_KM)
    print(f"\n{'model':<22} {'n':>3}{head} {'median km':>10} {'conf ok/wrong':>14} {'sec':>5}")
    for s in summary:
        accs = "".join(f"{pct(s[f'acc_{t}km']):>7}" for t in THRESHOLDS_KM)
        cw = f"{s['confidence_when_correct'] or 0:.2f}/{s['confidence_when_wrong'] or 0:.2f}"
        print(f"{s['model']:<22} {s['photos']:>3}{accs} "
              f"{s['median_error_km'] if s['median_error_km'] is not None else '-':>10} {cw:>14} "
              f"{s['median_seconds'] or '-':>5}")
    print(f"\nWrote summaries and charts to {out_dir}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Benchmark geolocation across models.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="Run photos through models, then report.")
    r.add_argument("--manifest", type=Path, required=True)
    r.add_argument("--models", required=True, help="Comma-separated model names.")
    r.add_argument("--label", default="", help="Suffix for the model column, e.g. ' (original)'.")
    r.add_argument("--out", type=Path, default=Path("benchmark_results"))
    rep = sub.add_parser("report", help="Rebuild summaries and charts from results.csv.")
    rep.add_argument("--out", type=Path, default=Path("benchmark_results"))
    args = parser.parse_args(argv)
    if args.cmd == "run":
        run(args.manifest, [m.strip() for m in args.models.split(",") if m.strip()], args.out, args.label)
    else:
        report(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
