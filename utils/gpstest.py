"""Ground-truth accuracy testing: strip GPS from geotagged photos, locate them
blind, and score each answer by its distance from the real camera position.

Stripping modes:
  location  (default) removes everything that gives the answer away -- the GPS
            IFD, XMP and IPTC blocks (which carry GPS and city/country fields)
            and free-text captions -- but keeps camera make/model and the
            timestamp, so forensics and the sun-position check still run as
            they would on a real photo that lost its location.
  all       removes all metadata, like a photo re-shared through social media.

JPEGs are rewritten segment by segment, so the pixels are never re-compressed
(re-compression would itself look like an edit to the forensics pass). Other
formats are re-encoded by Pillow. Every stripped image is re-read to confirm
the GPS is really gone before it is analysed.
"""

from __future__ import annotations

import io
import os
import statistics
import struct
import threading
import time
import uuid
from typing import Any, Callable, Optional

from PIL import Image

from utils.cache import parallel_map
from utils.exif import extract_exif
from utils.osint import DEFINED_RADIUS_KM, _haversine_km

STRIP_MODES = ("location", "all")
THRESHOLDS_KM = tuple(sorted({1.0, DEFINED_RADIUS_KM, 5.0, 25.0, 200.0, 750.0}))
GPS_TEST_MAX_IMAGES = int(os.getenv("GPS_TEST_MAX_IMAGES", "50"))
GPS_TEST_PARALLEL = int(os.getenv("GPS_TEST_PARALLEL", "2"))
GPS_TEST_MAX_JOBS = int(os.getenv("GPS_TEST_MAX_JOBS", "20"))

_GPS_IFD = 0x8825
_EXIF_IFD = 0x8769
# ImageDescription, Windows XP title/comment/keywords/subject, and (in the Exif
# sub-IFD) UserComment: captions such as "Lake Chelan 2019" leak the answer.
_CAPTION_TAGS = (0x010E, 0x9C9B, 0x9C9C, 0x9C9E, 0x9C9F)
_USER_COMMENT = 0x9286

AnalyzeFn = Callable[[bytes, str], tuple[int, dict[str, Any]]]


def read_gps(data: bytes) -> Optional[dict[str, float]]:
    """The photo's EXIF GPS position, or None."""
    gps = extract_exif(io.BytesIO(data)).get("gps")  # type: ignore[arg-type]
    if gps and gps.get("latitude") is not None and gps.get("longitude") is not None:
        return gps
    return None


def _clean_exif(payload: bytes) -> bytes:
    exif = Image.Exif()
    exif.load(payload)
    exif.pop(_GPS_IFD, None)
    for tag in _CAPTION_TAGS:
        exif.pop(tag, None)
    if _EXIF_IFD in exif:
        exif.get_ifd(_EXIF_IFD).pop(_USER_COMMENT, None)
    return exif.tobytes()


def _strip_jpeg(data: bytes, mode: str) -> bytes:
    """Rewrite a JPEG's metadata segments, copying the image data untouched."""
    out = bytearray(data[:2])
    pos = 2
    while pos + 4 <= len(data):
        if data[pos] != 0xFF:
            raise ValueError("malformed JPEG segment")
        marker = data[pos + 1]
        if marker == 0xFF:  # fill byte
            pos += 1
            continue
        if marker == 0xDA:  # start of scan: the rest is image data
            out += data[pos:]
            return bytes(out)
        if 0xD0 <= marker <= 0xD7 or marker == 0x01:
            out += data[pos : pos + 2]
            pos += 2
            continue
        (length,) = struct.unpack(">H", data[pos + 2 : pos + 4])
        segment = data[pos : pos + 2 + length]
        payload = segment[4:]
        pos += 2 + length

        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            if mode == "all":
                continue
            cleaned = _clean_exif(payload)
            if len(cleaned) + 2 > 0xFFFF:
                continue
            out += b"\xff\xe1" + struct.pack(">H", len(cleaned) + 2) + cleaned
        elif marker == 0xE1 or marker == 0xED:  # XMP / IPTC
            continue
        elif mode == "all" and 0xE3 <= marker <= 0xEF and marker != 0xEE:
            continue  # other APPn vendor blocks; keep ICC (E2) and Adobe (EE)
        elif marker == 0xFE and mode == "all":  # comment
            continue
        else:
            out += segment
    raise ValueError("JPEG ended before image data")


def _strip_other(data: bytes, mode: str) -> bytes:
    with Image.open(io.BytesIO(data)) as img:
        fmt = img.format or "PNG"
        params: dict[str, Any] = {}
        if mode != "all":
            exif = img.getexif()
            if exif:
                params["exif"] = _clean_exif(exif.tobytes())
        if "icc_profile" in img.info:
            params["icc_profile"] = img.info["icc_profile"]
        if fmt in ("JPEG", "WEBP"):
            params["quality"] = 95
        buf = io.BytesIO()
        img.save(buf, format=fmt, **params)
    return buf.getvalue()


def strip_location(data: bytes, mode: str = "location") -> bytes:
    """Return the image with its location metadata removed (see module doc)."""
    if mode not in STRIP_MODES:
        raise ValueError(f"mode must be one of {STRIP_MODES}")
    if data[:2] == b"\xff\xd8":
        return _strip_jpeg(data, mode)
    return _strip_other(data, mode)


def score_case(truth: dict[str, float], payload: dict[str, Any]) -> dict[str, Any]:
    """Distance of the pin and of the nearest search area from the true spot."""
    lat, lon = payload.get("latitude"), payload.get("longitude")
    error_km = (
        round(_haversine_km(truth["latitude"], truth["longitude"], lat, lon), 3)
        if lat is not None and lon is not None
        else None
    )
    hit_rank = None
    nearest_area_km = None
    for area in payload.get("search_areas") or []:
        dist = _haversine_km(
            truth["latitude"], truth["longitude"], area["latitude"], area["longitude"]
        )
        if nearest_area_km is None or dist < nearest_area_km:
            nearest_area_km = dist
        if hit_rank is None and dist <= float(area.get("search_radius_km") or 0):
            hit_rank = area.get("rank")
    return {
        "error_km": error_km,
        "within_defined_radius": error_km is not None and error_km <= DEFINED_RADIUS_KM,
        "search_area_hit_rank": hit_rank,
        "nearest_search_area_km": (
            round(nearest_area_km, 3) if nearest_area_km is not None else None
        ),
    }


def _case_record(
    name: str, truth: dict[str, float], status: int, payload: dict[str, Any], seconds: float
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "filename": name,
        "status": "ok" if status == 200 and payload.get("success") else "error",
        "true_latitude": truth["latitude"],
        "true_longitude": truth["longitude"],
        "predicted_latitude": payload.get("latitude"),
        "predicted_longitude": payload.get("longitude"),
        "location_name": payload.get("location_name"),
        "country": payload.get("country"),
        "confidence": payload.get("confidence"),
        "source": payload.get("source"),
        "verified": payload.get("verified"),
        "search_areas": len(payload.get("search_areas") or []),
        "seconds": round(seconds, 1),
    }
    if record["status"] == "error":
        record["error"] = payload.get("error") or f"HTTP {status}"
    record.update(score_case(truth, payload))
    return record


def _skipped(name: str, reason: str) -> dict[str, Any]:
    return {"filename": name, "status": "skipped", "error": reason}


def summarize(cases: list[dict[str, Any]]) -> dict[str, Any]:
    """Accuracy over the scored cases. Unanswered images count as misses."""
    scored = [c for c in cases if c["status"] in ("ok", "error")]
    errors = [c.get("error_km") for c in scored]
    answered = [e for e in errors if e is not None]
    n = len(scored)

    def rate(hits: int) -> Optional[float]:
        return round(hits / n, 3) if n else None

    return {
        "images": len(cases),
        "tested": n,
        "skipped": sum(c["status"] == "skipped" for c in cases),
        "failed": sum(c["status"] == "error" for c in cases),
        "answered": len(answered),
        "defined_radius_km": DEFINED_RADIUS_KM,
        "within_defined_radius": rate(sum(bool(c.get("within_defined_radius")) for c in scored)),
        "median_error_km": round(statistics.median(answered), 3) if answered else None,
        "mean_error_km": round(statistics.fmean(answered), 3) if answered else None,
        "accuracy": {
            f"{t:g}km": rate(sum(e is not None and e <= t for e in errors))
            for t in THRESHOLDS_KM
        },
        "search_area_hit_rate": rate(
            sum(c.get("search_area_hit_rank") is not None for c in scored)
        ),
        "mean_confidence": (
            round(statistics.fmean(c["confidence"] for c in scored if c.get("confidence") is not None), 3)
            if any(c.get("confidence") is not None for c in scored)
            else None
        ),
    }


def run_test(
    images: list[tuple[str, bytes]],
    analyze: AnalyzeFn,
    *,
    mode: str = "location",
    parallel: int = GPS_TEST_PARALLEL,
    on_case: Optional[Callable[[dict[str, Any]], None]] = None,
    keep_stripped: Optional[Callable[[str, bytes], None]] = None,
) -> dict[str, Any]:
    """Strip, analyse and score every image; returns the summary and cases.

    Images without GPS are reported as skipped and never analysed, so they cost
    nothing. Results come back in input order.
    """
    if mode not in STRIP_MODES:
        raise ValueError(f"mode must be one of {STRIP_MODES}")

    def one(entry: tuple[str, bytes]) -> dict[str, Any]:
        name, data = entry
        try:
            truth = read_gps(data)
        except Exception as exc:
            truth = None
            record = _skipped(name, f"unreadable image: {exc}")
        else:
            record = _skipped(name, "no EXIF GPS to test against") if truth is None else None
        if record is None:
            try:
                stripped = strip_location(data, mode)
                if read_gps(stripped) is not None:
                    raise ValueError("GPS survived stripping")
            except Exception as exc:
                record = _skipped(name, f"could not strip metadata: {exc}")
            else:
                if keep_stripped:
                    keep_stripped(name, stripped)
                start = time.monotonic()
                try:
                    status, payload = analyze(stripped, name)
                except Exception as exc:
                    status, payload = 500, {"success": False, "error": str(exc)}
                record = _case_record(name, truth, status, payload, time.monotonic() - start)
        if on_case:
            on_case(record)
        return record

    cases = parallel_map(one, images, max(1, parallel))
    return {"strip_mode": mode, "summary": summarize(cases), "cases": cases}


class JobStore:
    """In-memory background test jobs. Lost on restart, which is acceptable for
    a test harness; the oldest finished jobs are evicted beyond the cap."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def start(
        self, images: list[tuple[str, bytes]], analyze: AnalyzeFn, mode: str
    ) -> str:
        job_id = uuid.uuid4().hex[:12]
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "strip_mode": mode,
            "total": len(images),
            "completed": 0,
            "started_at": time.time(),
            "finished_at": None,
            "summary": None,
            "cases": [],
            "error": None,
        }
        with self._lock:
            self._evict()
            self._jobs[job_id] = job

        def progress(_record: dict[str, Any]) -> None:
            with self._lock:
                job["completed"] += 1

        def work() -> None:
            try:
                report = run_test(images, analyze, mode=mode, on_case=progress)
                with self._lock:
                    job.update(summary=report["summary"], cases=report["cases"], status="done")
            except Exception as exc:
                with self._lock:
                    job.update(status="failed", error=str(exc))
            finally:
                with self._lock:
                    job["finished_at"] = time.time()

        threading.Thread(target=work, name=f"gps-test-{job_id}", daemon=True).start()
        return job_id

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job, cases=list(job["cases"])) if job else None

    def _evict(self) -> None:
        done = sorted(
            (j for j in self._jobs.values() if j["status"] != "running"),
            key=lambda j: j["started_at"],
        )
        while len(self._jobs) >= GPS_TEST_MAX_JOBS and done:
            self._jobs.pop(done.pop(0)["job_id"], None)


jobs = JobStore()
