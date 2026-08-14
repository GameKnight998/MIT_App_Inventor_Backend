"""Turn a short video into a handful of high-value still frames.

A video is only useful for geolocation if different moments show different
clues (a sign that flashes by, a landmark revealed by a pan, several angles of
the same street). Analysing every frame would be slow and expensive AND mostly
redundant, so this module does what a human OSINT analyst does: scrub the clip
and keep only a few frames that are (a) sharp and (b) visually distinct from
each other. Those frames are then fed through the normal image pipeline.

It also makes a best-effort attempt to read GPS out of the video container
(QuickTime/MP4 store it in a `©xyz` atom as an ISO-6709 string), which -- when
present -- is far stronger than any visual guess.

Everything fails soft: unreadable or codec-unsupported files return no frames
so the caller can surface a clean error instead of crashing.
"""

from __future__ import annotations

import os
import re
from typing import Any, Optional

# Tunables (env-overridable). Kept conservative so a single free-tier worker
# isn't overwhelmed and the vision-API cost per video stays bounded.
VIDEO_MAX_FRAMES = int(os.getenv("VIDEO_MAX_FRAMES", "5"))
VIDEO_MAX_SCAN = int(os.getenv("VIDEO_MAX_SCAN", "40"))
VIDEO_MIN_BLUR = float(os.getenv("VIDEO_MIN_BLUR", "40"))
VIDEO_FRAME_DIFF_MIN = float(os.getenv("VIDEO_FRAME_DIFF_MIN", "8"))
VIDEO_FRAME_JPEG_QUALITY = int(os.getenv("VIDEO_FRAME_JPEG_QUALITY", "90"))

# ISO-6709 has two (optionally three) signed decimal numbers: lat then lon.
_ISO6709_NUM = re.compile(r"[+-]\d+(?:\.\d+)?")


def _import_cv2():
    """Import OpenCV lazily so the image-only path never needs the dependency."""
    try:
        import cv2  # type: ignore

        return cv2
    except Exception:
        return None


def video_support_available() -> bool:
    return _import_cv2() is not None


def extract_video_gps(video_path: str) -> Optional[dict[str, float]]:
    """Best-effort read of embedded GPS from an MP4/MOV `©xyz` atom.

    Returns {"latitude", "longitude"[, "altitude_m"]} or None. Never raises.
    """
    try:
        with open(video_path, "rb") as fh:
            data = fh.read()
    except Exception:
        return None

    marker = b"\xa9xyz"
    idx = data.find(marker)
    if idx < 0:
        return None
    try:
        pos = idx + len(marker)
        str_len = int.from_bytes(data[pos : pos + 2], "big")
        # 2 bytes size + 2 bytes language code, then the string.
        start = pos + 4
        raw = data[start : start + str_len].decode("utf-8", errors="ignore")
    except Exception:
        return None

    nums = _ISO6709_NUM.findall(raw)
    if len(nums) < 2:
        return None
    try:
        lat = float(nums[0])
        lon = float(nums[1])
    except ValueError:
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None

    gps: dict[str, float] = {"latitude": round(lat, 6), "longitude": round(lon, 6)}
    if len(nums) >= 3:
        try:
            gps["altitude_m"] = round(float(nums[2]), 2)
        except ValueError:
            pass
    return gps


def probe_video(video_path: str) -> dict[str, Any]:
    """Return basic properties (duration, fps, frame count) or an error."""
    cv2 = _import_cv2()
    if cv2 is None:
        return {"ok": False, "error": "Video support (OpenCV) is not installed."}
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return {"ok": False, "error": "Could not open video (unsupported codec?)."}
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
        return {
            "ok": frame_count > 0 or fps > 0,
            "fps": round(fps, 2),
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_s": round(duration, 2),
            "error": None if (frame_count > 0 or fps > 0) else "Video has no frames.",
        }
    finally:
        cap.release()


def _blur_score(cv2, gray) -> float:
    """Sharpness = variance of the Laplacian (higher = crisper)."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _thumb(cv2, gray):
    """Tiny grayscale signature used to measure inter-frame difference."""
    return cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)


def _thumb_diff(cv2, a, b) -> float:
    """Mean absolute per-pixel difference of two 32x32 thumbnails (0..255)."""
    import numpy as np  # bundled with opencv

    return float(np.mean(np.abs(a.astype("int16") - b.astype("int16"))))


def extract_key_frames(
    video_path: str, output_dir: str, max_frames: Optional[int] = None
) -> list[dict[str, Any]]:
    """Select up to `max_frames` sharp, visually-distinct frames.

    Two passes to keep memory bounded regardless of clip length:
      1. Sample ~VIDEO_MAX_SCAN positions evenly; record blur + a 32x32 thumb.
      2. Greedily keep the sharpest frames that are also different enough from
         those already kept (visual diversity), then re-read and save them.

    Returns a list of dicts: {path, index, timestamp_s, blur_score}. Empty on
    failure. Never raises.
    """
    cv2 = _import_cv2()
    if cv2 is None:
        return []
    max_frames = max_frames or VIDEO_MAX_FRAMES

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # Build the list of frame indices to inspect.
        if frame_count > 0:
            scan = min(VIDEO_MAX_SCAN, frame_count)
            step = max(1, frame_count // scan)
            positions = list(range(0, frame_count, step))[:VIDEO_MAX_SCAN]
        else:
            positions = []  # unknown length -> sequential scan below

        # --- Pass 1: score sampled frames (blur + thumbnail) ---
        scored: list[dict[str, Any]] = []
        if positions:
            for idx in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                scored.append(
                    {
                        "index": idx,
                        "timestamp_s": round(idx / fps, 2) if fps > 0 else None,
                        "blur": _blur_score(cv2, gray),
                        "thumb": _thumb(cv2, gray),
                    }
                )
        else:
            # Fallback for containers with no frame count: read sequentially and
            # subsample so we still inspect at most VIDEO_MAX_SCAN frames.
            i = 0
            read_step = 5
            while len(scored) < VIDEO_MAX_SCAN:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                if i % read_step == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    scored.append(
                        {
                            "index": i,
                            "timestamp_s": round(i / fps, 2) if fps > 0 else None,
                            "blur": _blur_score(cv2, gray),
                            "thumb": _thumb(cv2, gray),
                        }
                    )
                i += 1

        if not scored:
            return []

        # --- Pass 2: pick sharp + diverse frames ---
        sharp = [s for s in scored if s["blur"] >= VIDEO_MIN_BLUR] or scored
        sharp.sort(key=lambda s: s["blur"], reverse=True)

        selected: list[dict[str, Any]] = []
        for cand in sharp:
            if len(selected) >= max_frames:
                break
            if all(
                _thumb_diff(cv2, cand["thumb"], s["thumb"]) >= VIDEO_FRAME_DIFF_MIN
                for s in selected
            ):
                selected.append(cand)
        # If diversity filtering left room, top up with the next-sharpest frames.
        if len(selected) < max_frames:
            for cand in sharp:
                if cand in selected:
                    continue
                selected.append(cand)
                if len(selected) >= max_frames:
                    break

        selected.sort(key=lambda s: s["index"])  # chronological order

        # --- Pass 3: re-read the chosen frames and save as JPEG ---
        os.makedirs(output_dir, exist_ok=True)
        saved: list[dict[str, Any]] = []
        for n, sel in enumerate(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, sel["index"])
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            out_path = os.path.join(output_dir, f"frame_{n:02d}.jpg")
            cv2.imwrite(
                out_path,
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_FRAME_JPEG_QUALITY],
            )
            saved.append(
                {
                    "path": out_path,
                    "index": sel["index"],
                    "timestamp_s": sel["timestamp_s"],
                    "blur_score": round(sel["blur"], 1),
                }
            )
        return saved
    finally:
        cap.release()
