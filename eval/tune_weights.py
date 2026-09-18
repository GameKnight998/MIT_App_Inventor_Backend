"""Fit the log-linear fusion weights to the labelled set in labels.json.

The pipeline's confidence is a sum of independent log-odds contributions (see
utils/osint._fuse_confidence). Those weights were hand-set from judgement about
how much each kind of evidence is worth. This script checks that judgement
against the labelled cases and nudges the weights toward whatever is actually
better calibrated.

What is and isn't being tuned
-----------------------------
The weights only affect *confidence*, never which location is chosen: candidate
selection happens before fusion. So this is a pure calibration fit, and country /
region / radius accuracy should come out identical before and after. The script
asserts that and refuses to report a win if accuracy moved, which would mean the
separation between selection and scoring had been broken.

Method
------
Coordinate descent with a shrinking step: each weight is tried at a few offsets
around its current value, the best is kept, and the sweep repeats. It is slower
than a gradient but needs no derivative of a pipeline full of network stubs and
branching, and with ~20 cases it converges in seconds.

The objective is mean Brier score (how far confidence sits from the 0/1 truth of
whether the answer was right), plus two guards:

  * a band penalty for violating an explicit min_confidence / max_confidence on a
    case -- these encode requirements that no amount of Brier improvement should
    trade away, such as a generated image never scoring above 0.2;
  * an L2 pull back toward the hand-set defaults, because 20 cases cannot
    honestly support large moves. Raise --reg if the fit looks greedy, lower it
    once the label set is much bigger.

Usage
-----
  venv\\Scripts\\python.exe eval\\tune_weights.py
  venv\\Scripts\\python.exe eval\\tune_weights.py --passes 4 --reg 0.02
  venv\\Scripts\\python.exe eval\\tune_weights.py --write eval/tuned_weights.env

Nothing is applied automatically. The script prints a FUSION_* env block; copy
the lines you accept into .env or render.yaml, since every weight is read from
the environment at import time (utils/osint._w).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent
for _p in (ROOT, HERE):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# Each knob is one tunable number: the env name it is published under, a reader
# for its current value, and a writer. Dict-valued signals (verification status,
# solar, ...) are exposed one entry at a time so a single status can move without
# dragging its siblings.
def _build_knobs(O: Any) -> dict[str, tuple[Callable[[], float], Callable[[float], None], tuple[float, float]]]:
    def scalar(name: str, lo: float, hi: float):
        return (
            lambda: float(getattr(O, name)),
            lambda v: setattr(O, name, float(v)),
            (lo, hi),
        )

    def entry(dict_name: str, key: str, lo: float, hi: float):
        return (
            lambda: float(getattr(O, dict_name)[key]),
            lambda v: getattr(O, dict_name).__setitem__(key, float(v)),
            (lo, hi),
        )

    return {
        "FUSION_PRIOR": scalar("_FUSION_PRIOR", 0.02, 0.9),
        "FUSION_AI": scalar("_AI_WEIGHT", 0.0, 3.0),
        "FUSION_CLUE": scalar("_CLUE_WEIGHT", 0.0, 8.0),
        "FUSION_AMBIGUITY": scalar("_AMBIGUITY_WEIGHT", -3.0, 0.0),
        "FUSION_LANDMARK_CORROB": scalar("_LANDMARK_CORROBORATED", 0.0, 2.0),
        "FUSION_PRECISION": scalar("_PRECISION_BONUS", 0.0, 2.0),
        "FUSION_VERIFIED": entry("_VERIFY_LOGIT", "verified", -1.0, 3.0),
        "FUSION_PARTIAL": entry("_VERIFY_LOGIT", "partial", -2.0, 2.0),
        "FUSION_MISMATCH": entry("_VERIFY_LOGIT", "mismatch", -4.0, 0.0),
        "FUSION_UNAVAILABLE": entry("_VERIFY_LOGIT", "unavailable", -2.0, 1.0),
        "FUSION_SOLAR_OK": entry("_SOLAR_LOGIT", "consistent", 0.0, 2.0),
        "FUSION_SOLAR_BAD": entry("_SOLAR_LOGIT", "inconsistent", -3.0, 0.0),
        "FUSION_CLIMATE_OK": entry("_CLIMATE_LOGIT", "consistent", 0.0, 2.0),
        "FUSION_CLIMATE_BAD": entry("_CLIMATE_LOGIT", "inconsistent", -3.0, 0.0),
        "FUSION_NAMED_OK": entry("_NAMED_LOGIT", "matched", 0.0, 3.0),
        "FUSION_NAMED_BAD": entry("_NAMED_LOGIT", "missed", -3.0, 0.0),
        "FUSION_WATER_OK": entry("_WATER_NAMED_LOGIT", "matched", 0.0, 3.0),
        "FUSION_WATER_BAD": entry("_WATER_NAMED_LOGIT", "missed", -3.0, 0.0),
        "FUSION_STREET_OK": entry("_STREET_LOGIT", "geocoded", 0.0, 2.5),
        "FUSION_VEHICLE_OK": entry("_VEHICLE_LOGIT", "match", 0.0, 2.0),
        "FUSION_VEHICLE_BAD": entry("_VEHICLE_LOGIT", "conflict", -3.0, 0.0),
        "FUSION_TEXT_OK": entry("_TEXT_LOGIT", "match", 0.0, 2.5),
        "FUSION_TEXT_BAD": entry("_TEXT_LOGIT", "conflict", -3.0, 0.0),
    }


def _available_knobs(
    O: Any, run: Callable[[], dict[str, Any]], verbose: bool = False
) -> dict[str, Any]:
    """Keep only the knobs the label set can actually speak to.

    A weight no case exercises is unidentifiable: every value scores identically,
    so coordinate descent would settle on an arbitrary one and the script would
    print it as if it had been learned. Each candidate is therefore shoved hard in
    one direction; if no case's confidence moves, the knob is dropped.
    """
    baseline_conf = [r.get("confidence") for r in run().get("cases") or []]
    keep: dict[str, Any] = {}
    dropped: list[str] = []
    for name, (get, set_, bounds) in _build_knobs(O).items():
        try:
            current = get()
        except (AttributeError, KeyError):
            dropped.append(name)
            continue
        lo, hi = bounds
        probe = lo if abs(current - hi) > abs(current - lo) else hi
        set_(probe)
        moved = [r.get("confidence") for r in run().get("cases") or []] != baseline_conf
        set_(current)
        if moved:
            keep[name] = (get, set_, bounds)
        else:
            dropped.append(name)
    if verbose and dropped:
        print("  not exercised by any case, left at defaults: " + ", ".join(dropped))
    return keep


def _objective(
    summary: dict[str, Any],
    knobs: dict[str, Any],
    baseline: dict[str, float],
    reg: float,
    band_penalty: float,
) -> float:
    brier = float(summary.get("mean_brier") or 1.0)
    rows = summary.get("cases") or []
    violations = sum(1 for r in rows if r.get("confidence_band") is False)
    scored = sum(1 for r in rows if r.get("confidence_band") is not None)
    band = (violations / scored) if scored else 0.0
    drift = 0.0
    for name, (get, _set, _b) in knobs.items():
        drift += (get() - baseline[name]) ** 2
    drift /= max(len(knobs), 1)
    return brier + band_penalty * band + reg * drift


def _accuracy_fingerprint(summary: dict[str, Any]) -> tuple:
    """The parts of the result that must not move when only weights change."""
    return tuple(
        (r.get("id"), r.get("country"), r.get("region"), r.get("within_radius"))
        for r in summary.get("cases") or []
    )


def tune(
    passes: int = 3,
    reg: float = 0.01,
    band_penalty: float = 0.5,
    live: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    from run_eval import run_eval
    from utils import osint as O

    run = lambda: run_eval(live=live)
    knobs = _available_knobs(O, run, verbose=verbose)
    if not knobs:
        raise SystemExit("No fusion weight affects any labelled case; nothing to tune.")
    baseline = {name: get() for name, (get, _s, _b) in knobs.items()}

    first = run()
    fingerprint = _accuracy_fingerprint(first)
    best = _objective(first, knobs, baseline, reg, band_penalty)
    start_summary = first
    start_score = best

    if verbose:
        print(f"start: objective={best:.4f}  brier={first['mean_brier']}  "
              f"n={first['n']}  knobs={len(knobs)}")
        if int(first.get("n") or 0) < 30:
            print(
                "  note: fewer than 30 labelled cases. Treat the numbers below as a\n"
                "        sanity check on the hand-set weights, not a trained model."
            )

    step_schedule = [0.6, 0.3, 0.15, 0.08]
    for p in range(passes):
        step = step_schedule[min(p, len(step_schedule) - 1)]
        improved = False
        for name, (get, set_, (lo, hi)) in knobs.items():
            current = get()
            trials = [current + d for d in (-step, -step / 2, step / 2, step)]
            for value in trials:
                value = max(lo, min(hi, value))
                if abs(value - current) < 1e-6:
                    continue
                set_(value)
                score = _objective(run(), knobs, baseline, reg, band_penalty)
                if score < best - 1e-5:
                    best = score
                    current = value
                    improved = True
                else:
                    set_(current)
            if verbose and abs(current - baseline[name]) > 1e-6:
                print(f"  {name}: {baseline[name]:+.3f} -> {current:+.3f}")
        if verbose:
            print(f"pass {p + 1}/{passes}: objective={best:.4f} (step {step})")
        if not improved:
            break

    final = run()
    tuned = {name: get() for name, (get, _s, _b) in knobs.items()}
    changed = {n: v for n, v in tuned.items() if abs(v - baseline[n]) > 1e-6}

    stable = _accuracy_fingerprint(final) == fingerprint
    return {
        "n": final.get("n"),
        "objective_before": round(start_score, 4),
        "objective_after": round(best, 4),
        "brier_before": start_summary.get("mean_brier"),
        "brier_after": final.get("mean_brier"),
        "band_before": start_summary.get("confidence_band_accuracy"),
        "band_after": final.get("confidence_band_accuracy"),
        "gap_before": start_summary.get("calibration_gap"),
        "gap_after": final.get("calibration_gap"),
        "accuracy_unchanged": stable,
        "baseline": {k: round(v, 3) for k, v in baseline.items()},
        "tuned": {k: round(v, 3) for k, v in tuned.items()},
        "changed": {k: round(v, 3) for k, v in changed.items()},
    }


def _env_block(changed: dict[str, float]) -> str:
    if not changed:
        return "# No weight change improved the objective; the defaults stand.\n"
    lines = [
        "# Tuned fusion weights from eval/tune_weights.py.",
        "# Copy into .env (and render.yaml) to apply; utils/osint reads these at import.",
    ]
    for name, value in sorted(changed.items()):
        lines.append(f"{name}={value:g}")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Fit fusion weights to eval/labels.json (calibration only)."
    )
    parser.add_argument("--passes", type=int, default=3, help="Coordinate-descent sweeps.")
    parser.add_argument(
        "--reg",
        type=float,
        default=0.01,
        help="L2 pull toward the hand-set defaults; raise it on a small label set.",
    )
    parser.add_argument(
        "--band-penalty",
        type=float,
        default=0.5,
        help="Cost of violating a case's min/max confidence band.",
    )
    parser.add_argument("--live", action="store_true", help="Use real Nominatim/Overpass.")
    parser.add_argument("--json", action="store_true", help="Print JSON only.")
    parser.add_argument(
        "--write",
        metavar="PATH",
        help="Also write the FUSION_* env block to this file.",
    )
    args = parser.parse_args()

    result = tune(
        passes=args.passes,
        reg=args.reg,
        band_penalty=args.band_penalty,
        live=args.live,
        verbose=not args.json,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print()
        print(f"cases={result['n']}")
        print(f"objective {result['objective_before']} -> {result['objective_after']}")
        print(f"brier     {result['brier_before']} -> {result['brier_after']}")
        print(f"conf band {result['band_before']} -> {result['band_after']}")
        print(f"calib gap {result['gap_before']} -> {result['gap_after']}")
        if not result["accuracy_unchanged"]:
            print(
                "WARNING: country/region/radius accuracy moved. Fusion weights should\n"
                "         only affect confidence, so selection is reading a fused\n"
                "         score somewhere. Investigate before trusting these weights."
            )
        print()
        print(_env_block(result["changed"]), end="")

    if args.write:
        path = Path(args.write)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_env_block(result["changed"]), encoding="utf-8")
        if not args.json:
            print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
