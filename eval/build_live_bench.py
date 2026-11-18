"""Assemble eval/images + eval/live_labels.json from terrain_bench Commons photos.

Copies JPEGs with neutral names, strips EXIF GPS (labels keep ground truth),
and writes a fixed 70/30 dev/holdout split (seed 42).

  python eval/build_live_bench.py [--per-category 22] [--seed 42]
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "eval" / "terrain_bench"
OUT_IMG = ROOT / "eval" / "images"
LABELS = ROOT / "eval" / "live_labels.json"

# Map terrain folders -> Im2GPS-style categories used in reporting.
TERRAIN_TO_CATEGORY = {
    "urban": "urban",
    "coast": "coast",
    "mountains": "mountain",
    "canyon": "mountain",
    "desert": "rural",
    "forest": "rural",
    "grassland": "rural",
    "tropical": "rural",
    "arctic": "other",
    "wetland": "other",
}


def _strip_to(src: Path, dest: Path) -> None:
    with Image.open(src) as img:
        img.convert("RGB").save(dest, format="JPEG", quality=90)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-category", type=int, default=22)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-total", type=int, default=150)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    by_cat: dict[str, list[dict]] = {}
    for terrain, category in TERRAIN_TO_CATEGORY.items():
        man = SRC / terrain / "manifest.json"
        if not man.exists():
            continue
        rows = json.loads(man.read_text(encoding="utf-8"))
        usable = []
        for row in rows:
            path = SRC / terrain / row["file"]
            if path.exists() and row.get("latitude") is not None:
                usable.append({**row, "src": str(path), "category": category})
        rng.shuffle(usable)
        by_cat.setdefault(category, []).extend(usable)

    selected: list[dict] = []
    for category, pool in sorted(by_cat.items()):
        take = pool[: args.per_category]
        print(f"{category}: taking {len(take)} of {len(pool)}")
        selected.extend(take)

    if len(selected) < args.min_total:
        # Fill from leftovers to reach min_total.
        leftovers = []
        chosen_ids = {id(r) for r in selected}
        for pool in by_cat.values():
            for row in pool:
                if id(row) not in chosen_ids:
                    leftovers.append(row)
        rng.shuffle(leftovers)
        need = args.min_total - len(selected)
        selected.extend(leftovers[:need])
        print(f"filled +{min(need, len(leftovers))} -> {len(selected)}")

    rng.shuffle(selected)
    if OUT_IMG.exists():
        shutil.rmtree(OUT_IMG)
    OUT_IMG.mkdir(parents=True)

    labels = []
    for i, row in enumerate(selected, 1):
        fname = f"img_{i:04d}.jpg"
        _strip_to(Path(row["src"]), OUT_IMG / fname)
        # Stable id from content path + coords (not the place name).
        digest = hashlib.sha1(
            f"{row['latitude']}|{row['longitude']}|{row['src']}".encode()
        ).hexdigest()[:12]
        labels.append(
            {
                "id": f"live_{digest}",
                "file": fname,
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "country": None,  # filled optionally later; country metric uses reverse geocode of truth
                "region": row.get("anchor") or row.get("terrain"),
                "category": row["category"],
                "source_terrain": row.get("terrain"),
                "license": row.get("license"),
                "commons_title": row.get("title"),
            }
        )

    # 70/30 split by index after shuffle (fixed seed).
    n_dev = int(round(len(labels) * 0.7))
    for i, row in enumerate(labels):
        row["split"] = "dev" if i < n_dev else "holdout"

    LABELS.write_text(json.dumps(labels, indent=2, ensure_ascii=False), encoding="utf-8")
    n_hold = len(labels) - n_dev
    print(f"Wrote {len(labels)} images -> {OUT_IMG}")
    print(f"Labels -> {LABELS}  (dev={n_dev}, holdout={n_hold})")
    cats = {}
    for row in labels:
        cats[row["category"]] = cats.get(row["category"], 0) + 1
    print("categories:", cats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
