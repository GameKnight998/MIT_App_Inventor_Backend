"""Sun-position consistency check for geolocation (a small piece of #6).

Pure-Python, no dependencies. Given an EXIF timestamp and a candidate
coordinate, we compute where the sun WOULD be (azimuth + elevation) using the
low-precision NOAA solar-position algorithm, then compare that against what the
image actually shows (day vs. night, high vs. low sun, shadow direction). A
gross contradiction -- e.g. the photo is clearly daylight but the sun would be
below the horizon there at that moment -- is an independent reason to distrust a
candidate, exactly like a failed map-feature check.

This is a *forward consistency test*, not a locator: it only ever confirms or
contradicts a coordinate we already have. It fails soft (returns "unknown")
whenever inputs are missing or ambiguous -- which is often, because EXIF
timestamps are local time with no timezone, so we approximate the offset from
longitude.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Any, Optional


def parse_exif_timestamp(value: Optional[str]) -> Optional[datetime]:
    """Parse an EXIF 'YYYY:MM:DD HH:MM:SS' (or ISO-ish) local timestamp."""
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S", "%Y:%m:%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _julian_day(dt_utc: datetime) -> float:
    y, m = dt_utc.year, dt_utc.month
    day = dt_utc.day + (dt_utc.hour + dt_utc.minute / 60 + dt_utc.second / 3600) / 24
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + day + b - 1524.5


def sun_position(dt_utc: datetime, latitude: float, longitude: float) -> tuple[float, float]:
    """Return (azimuth_deg_from_north_CW, elevation_deg) for a UTC instant.

    Low-precision NOAA algorithm; accurate to a fraction of a degree, which is
    ample for day/night and elevation-band consistency checks.
    """
    jd = _julian_day(dt_utc)
    n = jd - 2451545.0
    mean_long_deg = (280.460 + 0.9856474 * n) % 360
    mean_anom = math.radians((357.528 + 0.9856003 * n) % 360)
    ecl_long = math.radians(
        (mean_long_deg + 1.915 * math.sin(mean_anom) + 0.020 * math.sin(2 * mean_anom))
        % 360
    )
    obliquity = math.radians(23.439 - 0.0000004 * n)

    decl = math.asin(math.sin(obliquity) * math.sin(ecl_long))
    right_asc = math.atan2(
        math.cos(obliquity) * math.sin(ecl_long), math.cos(ecl_long)
    )

    gmst = (280.46061837 + 360.98564736629 * n) % 360
    local_sidereal = math.radians((gmst + longitude) % 360)
    hour_angle = local_sidereal - right_asc

    lat = math.radians(latitude)
    elevation = math.asin(
        math.sin(lat) * math.sin(decl)
        + math.cos(lat) * math.cos(decl) * math.cos(hour_angle)
    )
    azimuth = math.atan2(
        -math.sin(hour_angle),
        math.tan(decl) * math.cos(lat) - math.sin(lat) * math.cos(hour_angle),
    )
    return (math.degrees(azimuth) + 360) % 360, math.degrees(elevation)


def _local_to_utc_by_longitude(dt_local: datetime, longitude: float) -> datetime:
    """Approximate UTC from a naive local timestamp using longitude/15 as the
    timezone offset. Coarse (ignores DST and political boundaries) but good
    enough for a day/night and rough elevation check."""
    offset_hours = round(longitude / 15.0)
    return (dt_local - timedelta(hours=offset_hours)).replace(tzinfo=timezone.utc)


def _observed_elevation_band(sun_obs: dict[str, Any]) -> Optional[str]:
    raw = (sun_obs.get("approx_solar_elevation") or "").strip().lower()
    if raw in ("low", "high", "medium"):
        return raw
    return None


def _observed_daylight(sun_obs: dict[str, Any], vision: dict[str, Any]) -> Optional[bool]:
    """True if the image clearly shows daylight, False if clearly night."""
    if sun_obs.get("sun_visible") or sun_obs.get("shadows_visible"):
        return True
    if _observed_elevation_band(sun_obs):
        return True
    tod = (vision.get("time_of_day") or "").lower()
    if any(w in tod for w in ("day", "noon", "morning", "afternoon", "sunrise", "sunset")):
        return True
    if any(w in tod for w in ("night", "midnight", "dark")):
        return False
    return None


def sun_consistency(
    latitude: float,
    longitude: float,
    exif_timestamp: Optional[str],
    vision: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Compare computed sun position at (lat, lon, time) with the image.

    Returns a report dict with a `status` of:
      - "consistent"     computed sun matches what the photo shows
      - "inconsistent"   a strong day/night contradiction
      - "weak_mismatch"  elevation band disagrees (soft signal only)
      - "unknown"        not enough information to judge
    """
    report: dict[str, Any] = {
        "status": "unknown",
        "note": "",
        "expected_azimuth": None,
        "expected_elevation": None,
    }
    if latitude is None or longitude is None:
        report["note"] = "No coordinates to check."
        return report
    dt_local = parse_exif_timestamp(exif_timestamp)
    if dt_local is None:
        report["note"] = "No usable timestamp; skipped."
        return report

    sun_obs = (vision or {}).get("sun") or {}
    dt_utc = _local_to_utc_by_longitude(dt_local, longitude)
    azimuth, elevation = sun_position(dt_utc, latitude, longitude)
    report["expected_azimuth"] = round(azimuth, 1)
    report["expected_elevation"] = round(elevation, 1)

    daylight = _observed_daylight(sun_obs, vision or {})
    if daylight is True and elevation < -6.0:
        report["status"] = "inconsistent"
        report["note"] = (
            f"Photo looks like daylight, but the sun would be {elevation:.0f}deg "
            f"(below the horizon) here at that time."
        )
        return report
    if daylight is False and elevation > 5.0:
        report["status"] = "inconsistent"
        report["note"] = (
            f"Photo looks like night, but the sun would be up ({elevation:.0f}deg) "
            f"here at that time."
        )
        return report

    band = _observed_elevation_band(sun_obs)
    if band == "high" and elevation < 25.0:
        report["status"] = "weak_mismatch"
        report["note"] = (
            f"Image suggests a high sun, but computed elevation is only "
            f"{elevation:.0f}deg."
        )
        return report
    if band == "low" and elevation > 45.0:
        report["status"] = "weak_mismatch"
        report["note"] = (
            f"Image suggests a low sun, but computed elevation is {elevation:.0f}deg."
        )
        return report

    if daylight is not None or band:
        report["status"] = "consistent"
        report["note"] = (
            f"Sun geometry consistent (elevation ~{elevation:.0f}deg, "
            f"azimuth ~{azimuth:.0f}deg)."
        )
    else:
        report["note"] = "No sun/shadow observations to compare."
    return report
