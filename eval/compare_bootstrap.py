"""Paired bootstrap compare of two live-eval result JSONs (dev cases).

  python eval/compare_bootstrap.py baseline.json candidate.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def _cases(path: Path) -> dict[str, dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = data.get("runs") or [data]
    rows = runs[0].get("cases") or data.get("cases") or []
    return {r["id"]: r for r in rows if r.get("id")}


def _metric(rows: list[dict[str, Any]], kind: str) -> float:
    errs = [r.get("error_km") for r in rows]
    if kind == "acc25":
        vals = [e for e in errs if e is not None]
        return sum(1 for e in vals if e <= 25) / len(vals) if vals else 0.0
    if kind == "median":
        vals = sorted(e for e in errs if e is not None)
        if not vals:
            return float("inf")
        mid = len(vals) // 2
        return float(vals[mid] if len(vals) % 2 else 0.5 * (vals[mid - 1] + vals[mid]))
    raise ValueError(kind)


def bootstrap(
    base: dict[str, dict[str, Any]],
    cand: dict[str, dict[str, Any]],
    kind: str,
    n: int = 1000,
    seed: int = 0,
) -> dict[str, float]:
    ids = sorted(set(base) & set(cand))
    rng = random.Random(seed)
    b0 = _metric([base[i] for i in ids], kind)
    c0 = _metric([cand[i] for i in ids], kind)
    # Higher acc is better; lower median is better.
    better = (c0 - b0) if kind == "acc25" else (b0 - c0)
    wins = 0
    for _ in range(n):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        bb = _metric([base[i] for i in sample], kind)
        cc = _metric([cand[i] for i in sample], kind)
        delta = (cc - bb) if kind == "acc25" else (bb - cc)
        if delta > 0:
            wins += 1
    # one-sided p: probability improvement is noise (not > 0)
    p = 1.0 - wins / n
    return {
        "n_ids": len(ids),
        "base": b0,
        "cand": c0,
        "delta": better,
        "p_one_sided": p,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("candidate")
    ap.add_argument("--n", type=int, default=1000)
    args = ap.parse_args()
    base = _cases(Path(args.baseline))
    cand = _cases(Path(args.candidate))
    for kind in ("acc25", "median"):
        r = bootstrap(base, cand, kind, n=args.n)
        print(
            f"{kind}: base={r['base']:.4f} cand={r['cand']:.4f} "
            f"delta={r['delta']:.4f} p={r['p_one_sided']:.4f} n={r['n_ids']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
