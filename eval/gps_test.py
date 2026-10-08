"""Accuracy test on a folder of geotagged photos.

Every image's EXIF GPS is the ground truth. The image is stripped of its
location metadata, located by the backend, and scored by how far the answer
lands from the real camera position.

  python eval/gps_test.py PHOTOS_DIR                    # in-process pipeline
  python eval/gps_test.py PHOTOS_DIR --url https://<app>.onrender.com
  python eval/gps_test.py PHOTOS_DIR --mode all --out report.json

`--mode location` (default) removes GPS, XMP, IPTC and captions but keeps the
camera and timestamp; `--mode all` removes every tag. `--save-stripped DIR`
keeps the exact files that were analysed so a case can be re-run by hand.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from utils import gpstest  # noqa: E402

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def _collect(folder: Path, recursive: bool) -> list[tuple[str, bytes]]:
    paths = folder.rglob("*") if recursive else folder.iterdir()
    files = sorted(p for p in paths if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    return [(str(p.relative_to(folder)), p.read_bytes()) for p in files]


def _remote(url: str, timeout: float) -> gpstest.AnalyzeFn:
    import requests

    endpoint = url.rstrip("/") + "/analyze"

    def analyze(data: bytes, name: str) -> tuple[int, dict[str, Any]]:
        resp = requests.post(
            endpoint, files={"image": (Path(name).name, data)}, timeout=timeout
        )
        try:
            return resp.status_code, resp.json()
        except ValueError:
            return resp.status_code, {"success": False, "error": resp.text[:200]}

    return analyze


def _local() -> gpstest.AnalyzeFn:
    from main import _analyze_image_bytes

    return lambda data, name: _analyze_image_bytes(data, name)


def _km(value: Any) -> str:
    return "-" if value is None else f"{value:,.1f}"


def _print_case(case: dict[str, Any]) -> None:
    if case["status"] == "skipped":
        print(f"  SKIP  {case['filename']}: {case['error']}", flush=True)
        return
    if case["status"] == "error":
        print(f"  FAIL  {case['filename']}: {case.get('error')}", flush=True)
        return
    hit = case.get("search_area_hit_rank")
    print(
        f"  {_km(case['error_km']):>10} km  {case['filename']}  -> "
        f"{case.get('location_name')}, {case.get('country')}  "
        f"(conf {case.get('confidence')}, search area hit: "
        f"{'#' + str(hit) if hit else 'none'}, {case['seconds']} s)",
        flush=True,
    )


def _print_report(report: dict[str, Any]) -> None:
    s = report["summary"]
    print()
    print(f"Strip mode: {report['strip_mode']}")
    print(
        f"Images: {s['images']}  tested: {s['tested']}  answered: {s['answered']}  "
        f"skipped (no GPS / unreadable): {s['skipped']}  failed: {s['failed']}"
    )
    if not s["tested"]:
        return
    print(f"Median error: {_km(s['median_error_km'])} km   mean: {_km(s['mean_error_km'])} km")
    print(
        f"Within the {s['defined_radius_km']:g} km defined radius: "
        f"{s['within_defined_radius']:.0%}"
    )
    print("Accuracy: " + "  ".join(f"@{k} {v:.0%}" for k, v in s["accuracy"].items()))
    print(f"True spot inside one of the ranked search areas: {s['search_area_hit_rate']:.0%}")

    print("\nPer image (worst first):")
    scored = [c for c in report["cases"] if c["status"] != "skipped"]
    scored.sort(key=lambda c: -1 if c.get("error_km") is None else c["error_km"], reverse=True)
    for case in scored:
        print(
            f"  {_km(case.get('error_km')):>10} km  {case['filename']}  "
            f"true ({case['true_latitude']:.5f}, {case['true_longitude']:.5f})  "
            f"pred ({case.get('predicted_latitude')}, {case.get('predicted_longitude')})"
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("folder", type=Path)
    ap.add_argument("--url", help="test a deployed backend instead of the local code")
    ap.add_argument("--mode", choices=gpstest.STRIP_MODES, default="location")
    ap.add_argument("--parallel", type=int, default=gpstest.GPS_TEST_PARALLEL)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--timeout", type=float, default=300, help="per-image seconds (--url)")
    ap.add_argument("--out", type=Path, help="write the full JSON report here")
    ap.add_argument("--save-stripped", type=Path, metavar="DIR")
    args = ap.parse_args()

    if not args.folder.is_dir():
        print(f"Not a folder: {args.folder}", file=sys.stderr)
        return 2
    images = _collect(args.folder, args.recursive)
    if not images:
        print(f"No images in {args.folder}", file=sys.stderr)
        return 2

    keep = None
    if args.save_stripped:
        args.save_stripped.mkdir(parents=True, exist_ok=True)

        def keep(name: str, data: bytes) -> None:
            target = args.save_stripped / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)

    analyze = _remote(args.url, args.timeout) if args.url else _local()
    print(f"Testing {len(images)} image(s) against {args.url or 'the local pipeline'}...")
    report = gpstest.run_test(
        images,
        analyze,
        mode=args.mode,
        parallel=args.parallel,
        on_case=_print_case,
        keep_stripped=keep,
    )
    _print_report(report)
    if args.out:
        args.out.write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nReport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
