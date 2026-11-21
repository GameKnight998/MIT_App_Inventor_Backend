"""Turn a folder of GPS-tagged photos into a benchmark set.

Input: one folder with a subfolder per category, holding original photos
(JPEG, PNG or iPhone HEIC) that still carry their GPS location:

    team-photos/city/  team-photos/suburb/  team-photos/nature/
    team-photos/wilderness/  team-photos/landmark/  ...

Output, in OUT:
    photos/<category>/<name>.jpg   copies with ALL metadata removed (no GPS,
                                   no camera, no time), turned upright
    manifest.csv                   file,latitude,longitude,country,place,category
                                   for eval/benchmark.py
    needs_location.csv             photos with no GPS: fill in their latitude
                                   and longitude (from a map) in manifest.csv,
                                   or delete their rows

The true location lives only in manifest.csv, so the photo copies can be
shared without revealing where each one was taken. Keep manifest.csv private
(not in GitHub: the repo is public).

Usage:
    pip install pillow-heif          # once, for iPhone HEIC photos
    python eval/make_manifest.py team-photos/ benchmark-set/
    python eval/benchmark.py run --manifest benchmark-set/manifest.csv \\
        --models gpt-4o,gpt-6.1-terra --out benchmark-set/results/

Add --no-lookup to skip filling in country and place names (one map lookup
per photo, about 1 second each).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image, ImageOps  # noqa: E402

from utils.exif import extract_exif  # noqa: E402

try:  # iPhone photos are HEIC; Pillow opens them only with this plugin.
    from pillow_heif import register_heif_opener

    register_heif_opener()
    HEIC_OK = True
except ImportError:
    HEIC_OK = False

IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff"}
FIELDS = ["file", "latitude", "longitude", "country", "place", "category"]


def _clean_copy(src: Path, dest: Path) -> None:
    """Save an upright JPEG with no metadata at all."""
    with Image.open(src) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        clean = Image.new("RGB", img.size)
        clean.paste(img)  # a fresh image carries no EXIF, GPS or XMP
        dest.parent.mkdir(parents=True, exist_ok=True)
        clean.save(dest, "JPEG", quality=92)


def build(photos_dir: Path, out_dir: Path, lookup: bool = True) -> int:
    from utils.geocode import reverse_geocode

    rows, missing, skipped = [], [], []
    sources = sorted(p for p in photos_dir.rglob("*") if p.suffix.lower() in IMAGE_TYPES and p.is_file())
    for src in sources:
        rel = src.relative_to(photos_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "uncategorised"
        if src.suffix.lower() in (".heic", ".heif") and not HEIC_OK:
            skipped.append(f"{rel} (HEIC: run `pip install pillow-heif` first)")
            continue
        meta = extract_exif(str(src))
        gps = meta.get("gps") or {}
        name = f"{category}/{src.stem}.jpg"
        try:
            _clean_copy(src, out_dir / "photos" / name)
        except Exception as exc:
            skipped.append(f"{rel} ({exc})")
            continue
        row = {"file": f"photos/{name}", "latitude": gps.get("latitude", ""),
               "longitude": gps.get("longitude", ""), "country": "", "place": "",
               "category": category}
        if row["latitude"] == "":
            missing.append(row)
        elif lookup:
            hit = reverse_geocode(row["latitude"], row["longitude"]) or {}
            row["country"] = hit.get("country") or ""
            row["place"] = hit.get("name") or hit.get("city") or ""
        rows.append(row)
        print(f"{rel}: {row['latitude'] or 'NO GPS'} {row['longitude']} {row['place']}")

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    with (out_dir / "needs_location.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(missing)

    # A fingerprint of the set, so everyone can check they test the same photos.
    digest = hashlib.sha256()
    for row in rows:
        digest.update((out_dir / row["file"]).read_bytes())
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1

    print(f"\n{len(rows)} photos written to {out_dir}/manifest.csv")
    for cat, n in sorted(counts.items()):
        print(f"  {cat}: {n}")
    print(f"Fingerprint: {digest.hexdigest()[:12]}")
    if missing:
        print(f"{len(missing)} photos have no GPS: see needs_location.csv")
    for s in skipped:
        print(f"Skipped {s}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Build a benchmark set from GPS-tagged photos.")
    ap.add_argument("photos_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--no-lookup", action="store_true", help="Don't fill in country and place names.")
    args = ap.parse_args(argv)
    return build(args.photos_dir, args.out_dir, lookup=not args.no_lookup)


if __name__ == "__main__":
    sys.exit(main())
