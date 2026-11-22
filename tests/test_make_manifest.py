"""Tests for eval/make_manifest.py (building a benchmark set from GPS photos).

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

from eval.make_manifest import build  # noqa: E402
from utils.exif import extract_exif  # noqa: E402


def _photo_with_gps(path: Path, lat: float, lon: float) -> None:
    def dms(v):
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        return (float(d), float(m), round(((v - d) * 60 - m) * 60, 2))
    img = Image.new("RGB", (64, 48), "green")
    exif = img.getexif()
    exif[0x8825] = {1: "N" if lat >= 0 else "S", 2: dms(lat), 3: "E" if lon >= 0 else "W", 4: dms(lon)}
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, exif=exif)


class MakeManifestTests(unittest.TestCase):
    def test_location_goes_to_manifest_and_copies_are_clean(self):
        with tempfile.TemporaryDirectory() as d:
            src, out = Path(d) / "team", Path(d) / "set"
            _photo_with_gps(src / "wilderness" / "ridge.jpg", 36.2719, -115.6957)
            (src / "city").mkdir()
            Image.new("RGB", (64, 48), "blue").save(src / "city" / "street.jpg")  # no GPS
            build(src, out, lookup=False)
            rows = list(csv.DictReader((out / "manifest.csv").open()))
            by_file = {r["file"]: r for r in rows}
            ridge = by_file["photos/wilderness/ridge.jpg"]
            self.assertAlmostEqual(float(ridge["latitude"]), 36.2719, places=3)
            self.assertAlmostEqual(float(ridge["longitude"]), -115.6957, places=3)
            self.assertEqual(ridge["category"], "wilderness")
            self.assertIsNone(extract_exif(str(out / ridge["file"])).get("gps"))  # copy has no GPS
            missing = list(csv.DictReader((out / "needs_location.csv").open()))
            self.assertEqual([m["file"] for m in missing], ["photos/city/street.jpg"])


if __name__ == "__main__":
    unittest.main()
