"""Fit the confidence-fusion weights to labelled results.

The weights in utils/osint.py (_DEFAULT_WEIGHTS) were set by hand, so a
reported "0.3" does not mean "right 30% of the time". This script runs the
eval, records each case's confidence signals and whether the answer landed
within the labelled radius, then fits a logistic regression:

    P(correct) = sigmoid(bias + sum(weight[f] * feature[f]))

The fit is pulled toward the current weights (ridge penalty `--strength`), so
a small labelled set only nudges them instead of overfitting. Write the result
to utils/fusion_weights.json, which the pipeline loads at startup.

Usage:
  python eval/calibrate.py --photos           # real photos + live vision
  python eval/calibrate.py --photos --write   # also save utils/fusion_weights.json
  python eval/calibrate.py                    # offline fixtures (smoke test)

Aim for 50+ labelled photos before trusting the fitted weights.
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

from eval.run_eval import run_eval  # noqa: E402
from utils.osint import (  # noqa: E402
    FUSION_WEIGHTS_PATH,
    _fuse_confidence,
    fusion_features,
    load_fusion_weights,
)


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def fit_weights(
    samples: list[tuple[dict[str, float], float]],
    prior: dict[str, float],
    strength: float = 1.0,
    steps: int = 4000,
    lr: float = 0.05,
) -> dict[str, float]:
    """Gradient descent on log-loss + strength * ||w - prior||^2 / n."""
    w = dict(prior)
    n = max(len(samples), 1)
    for _ in range(steps):
        grad = {k: 2 * strength * (w[k] - prior[k]) / n for k in w}
        for feats, y in samples:
            z = w["bias"] + sum(w.get(f, 0.0) * v for f, v in feats.items())
            err = _sigmoid(z) - y
            grad["bias"] += err / n
            for f, v in feats.items():
                if f in grad:
                    grad[f] += err * v / n
        for k in w:
            w[k] -= lr * grad[k]
    return {k: round(v, 4) for k, v in w.items()}


def _brier(rows: list[tuple[dict[str, Any], float]], weights: dict[str, float]) -> float:
    errs = [(_fuse_confidence(sig, weights) - y) ** 2 for sig, y in rows]
    return round(sum(errs) / len(errs), 4) if errs else float("nan")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fit confidence-fusion weights.")
    parser.add_argument("--photos", action="store_true", help="Use real photos + live vision.")
    parser.add_argument("--live", action="store_true", help="Real Nominatim/Overpass.")
    parser.add_argument("--labels", type=Path, help="Labels file (default eval/labels.json).")
    parser.add_argument("--strength", type=float, default=1.0, help="Pull toward current weights.")
    parser.add_argument("--write", action="store_true", help="Save utils/fusion_weights.json.")
    args = parser.parse_args(argv)

    summary = run_eval(
        live=args.live or args.photos, use_photos=args.photos, labels_path=args.labels
    )
    rows: list[tuple[dict[str, Any], float]] = []
    for case in summary["cases"]:
        if case.get("signals") is None or case.get("within_radius") is None:
            continue  # GPS / name-only / unknown answers have no fused score
        rows.append((case["signals"], 1.0 if case["within_radius"] else 0.0))

    print(f"Cases with fused confidence: {len(rows)} of {summary['n']}")
    if len(rows) < 2:
        print("Not enough labelled cases to fit. Add photos to the labels file.")
        return 1

    prior = load_fusion_weights()
    fitted = fit_weights(
        [(fusion_features(sig), y) for sig, y in rows], prior, strength=args.strength
    )
    print(f"Brier score  current={_brier(rows, prior)}  fitted={_brier(rows, fitted)}")
    print("Weight changes (current -> fitted):")
    for key in sorted(fitted):
        if abs(fitted[key] - prior[key]) >= 0.01:
            print(f"  {key:<24} {prior[key]:>7.3f} -> {fitted[key]:>7.3f}")
    if len(rows) < 50:
        print("Warning: fewer than 50 cases; treat the fitted weights as provisional.")

    if args.write:
        Path(FUSION_WEIGHTS_PATH).write_text(
            json.dumps({"n_cases": len(rows), "weights": fitted}, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {FUSION_WEIGHTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
