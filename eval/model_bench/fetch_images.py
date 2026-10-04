"""Build the remote-terrain benchmark set from geotagged Wikimedia Commons photos.

For each target region, finds photos whose Commons coordinates lie within
10 km of the target, keeps landscape-looking JPEGs, and writes the first
acceptable one to images/ plus a row in manifest.json (ground-truth camera
coordinates, attribution, licence). Re-running reuses the manifest; pass
--refresh to re-query, or --skip FILE_TITLE to reject a pick.

  python eval/model_bench/fetch_images.py [--refresh] [--skip "File:..."]
"""

import argparse
import json
import re
import time
from pathlib import Path

import requests

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"
MANIFEST = HERE / "manifest.json"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "ImageLocatorBenchmark/1.0 (capstone research; contact via GitHub)"}

TARGETS = [
    ("atacama", "Atacama Desert, Chile", -22.9, -67.9),
    ("altiplano", "Bolivian Altiplano", -21.9, -67.6),
    ("patagonia", "Patagonian steppe, Argentina", -45.9, -71.2),
    ("tierra_del_fuego", "Tierra del Fuego", -54.84, -68.5),
    ("damaraland", "Damaraland, Namibia", -20.6, 14.3),
    ("namib", "Namib Desert, Namibia", -24.73, 15.34),
    ("tassili", "Tassili n'Ajjer, Algeria", 24.5, 9.5),
    ("simien", "Simien Mountains, Ethiopia", 13.2, 38.0),
    ("maasai_mara", "Maasai Mara, Kenya", -1.5, 35.1),
    ("iceland_highlands", "Icelandic Highlands", 63.99, -19.06),
    ("glen_coe", "Scottish Highlands", 56.67, -5.0),
    ("svalbard", "Svalbard", 78.2, 15.6),
    ("lapland", "Finnish Lapland", 69.04, 20.85),
    ("altai", "Altai Mountains, Russia", 50.0, 87.8),
    ("charyn", "Kazakh steppe canyon", 43.35, 79.08),
    ("song_kol", "Tian Shan, Kyrgyzstan", 41.83, 75.12),
    ("ladakh", "Ladakh, India", 34.0, 77.9),
    ("mongolia_steppe", "Mongolian steppe", 47.7, 105.9),
    ("gobi", "Gobi Desert, Mongolia", 43.49, 104.07),
    ("wahiba", "Wahiba Sands, Oman", 22.2, 58.8),
    ("denali_hwy", "Interior Alaska tundra", 63.1, -147.5),
    ("tombstone", "Yukon, Canada", 64.5, -138.2),
    ("kangerlussuaq", "Western Greenland", 67.0, -50.7),
    ("colca", "Andes, Peru", -15.6, -71.9),
    ("khibiny", "Kola Peninsula, Russia", 67.6, 33.7),
]

# Titles that are rarely a plain outdoor photo of the terrain.
_REJECT = re.compile(
    r"map|karte|carte|panorama|diagram|logo|sign|plaque|museum|interior|church|"
    r"hotel|portrait|selfie|sketch|drawing|flag|coat|svg|screenshot|\bbus\b|train|"
    r"station|airport|market|mosque|temple|monument|statue|"
    r"ISS\d|view of earth|landsat|sentinel|satellite|nasa|astronaut",
    re.I,
)


def _get(params: dict) -> dict:
    params = {"format": "json", "formatversion": 2, **params}
    for attempt in range(6):
        resp = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if resp.status_code == 429:
            time.sleep(int(resp.headers.get("Retry-After", 0)) or 15 * (attempt + 1))
            continue
        resp.raise_for_status()
        return resp.json()
    resp.raise_for_status()
    return {}


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text or "").strip()


def _candidates(lat: float, lon: float) -> list[dict]:
    hits = _get(
        {
            "action": "query",
            "list": "geosearch",
            "gscoord": f"{lat}|{lon}",
            "gsradius": 10000,
            "gsnamespace": 6,
            "gslimit": 60,
        }
    )["query"]["geosearch"]
    titles = [h["title"] for h in hits if not _REJECT.search(h["title"])]
    if not titles:
        return []
    pages = _get(
        {
            "action": "query",
            "titles": "|".join(titles[:50]),
            "prop": "imageinfo|coordinates",
            "iiprop": "url|size|mime|extmetadata",
            "iiurlwidth": 1024,
            "coprop": "type",
        }
    )["query"]["pages"]
    out = []
    for page in pages:
        info = (page.get("imageinfo") or [{}])[0]
        coords = page.get("coordinates") or []
        if info.get("mime") != "image/jpeg" or not coords:
            continue
        if min(info.get("width", 0), info.get("height", 0)) < 700:
            continue
        meta = info.get("extmetadata", {})
        out.append(
            {
                "title": page["title"],
                "latitude": coords[0]["lat"],
                "longitude": coords[0]["lon"],
                "thumb_url": info.get("thumburl"),
                "page_url": info.get("descriptionurl"),
                "artist": _strip_html(meta.get("Artist", {}).get("value", "")),
                "license": meta.get("LicenseShortName", {}).get("value", ""),
                "pixels": info.get("width", 0) * info.get("height", 0),
            }
        )
    out.sort(key=lambda c: -c["pixels"])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--skip", action="append", default=[], help="Commons file title to reject")
    args = ap.parse_args()

    IMAGES.mkdir(parents=True, exist_ok=True)
    manifest = {} if args.refresh or not MANIFEST.exists() else {
        row["id"]: row for row in json.loads(MANIFEST.read_text(encoding="utf-8"))
    }
    skipped = set(args.skip) | {
        t for row in manifest.values() for t in row.get("rejected", [])
    }

    for tid, region, lat, lon in TARGETS:
        row = manifest.get(tid)
        if row and row.get("title") not in skipped and (IMAGES / row["file"]).exists():
            continue
        rejected = sorted(set((row or {}).get("rejected", [])) | ({row["title"]} & skipped if row else set()))
        pick = next(
            (c for c in _candidates(lat, lon) if c["title"] not in skipped | set(rejected)),
            None,
        )
        if pick is None:
            print(f"{tid}: no usable photo")
            manifest.pop(tid, None)
            continue
        file = f"{tid}.jpg"
        data = requests.get(pick["thumb_url"], headers=HEADERS, timeout=60).content
        _save_stripped(data, IMAGES / file)
        manifest[tid] = {
            "id": tid,
            "region": region,
            "file": file,
            **{k: pick[k] for k in ("title", "latitude", "longitude", "page_url", "artist", "license")},
            "rejected": rejected,
        }
        print(f"{tid}: {pick['title']} ({pick['latitude']:.4f}, {pick['longitude']:.4f})")
        _save(manifest)
        time.sleep(3)

    print(f"{_save(manifest)} images in manifest")


def _save_stripped(data: bytes, path: Path) -> None:
    """Re-encode without EXIF so no model (or the pipeline) can read GPS tags."""
    import io

    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img.convert("RGB").save(path, format="JPEG", quality=92)


def strip_all() -> None:
    for path in IMAGES.glob("*.jpg"):
        _save_stripped(path.read_bytes(), path)


def _save(manifest: dict) -> int:
    rows = [manifest[t[0]] for t in TARGETS if t[0] in manifest]
    MANIFEST.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(rows)


if __name__ == "__main__":
    main()
