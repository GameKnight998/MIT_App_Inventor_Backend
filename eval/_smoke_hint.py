"""Live smoke: rerun two misses from the dev split with a location hint."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

import main  # noqa: E402

CASES = [
    ("img_0025.jpg", "Banff", 51.453933, -116.140626),
]


def km(a, b, c, d):
    p1, p2 = math.radians(a), math.radians(c)
    h = math.sin((p2 - p1) / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(math.radians(d - b) / 2) ** 2
    return 12742 * math.asin(math.sqrt(h))


for name, hint, lat, lon in CASES:
    data = (ROOT / "eval" / "images" / name).read_bytes()
    _, out = main._analyze_image_bytes(data, name, "image/jpeg", location_hint=hint)
    plat, plon = out.get("latitude"), out.get("longitude")
    err = km(lat, lon, plat, plon) if plat is not None else None
    print(name, repr(hint), "->", out.get("location_name"), plat, plon,
          f"err={err:.1f}km" if err is not None else "no pin")
    print("   hint:", out.get("location_hint"))
