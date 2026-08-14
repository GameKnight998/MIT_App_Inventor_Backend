"""Elevation and climate cross-checks (independent physical signals).

Uses free, key-less APIs:
  - Open-Elevation for ground elevation at a coordinate.
  - Open-Meteo's historical archive for a coarse annual climate (mean
    temperature and total precipitation over a recent year).

We compare those against the scene the image shows (snow, desert, forest) and
report consistent / weak_mismatch / unknown. This is an independent corroborating
signal, not a locator. Fail-soft: any error yields "unknown" and no effect.
"""

from __future__ import annotations

import datetime as _dt
import os
from typing import Any, Optional

import requests

from utils.cache import cached

ELEV_URL = os.getenv("OPEN_ELEVATION_URL", "https://api.open-elevation.com/api/v1/lookup")
CLIMATE_URL = os.getenv(
    "OPEN_METEO_ARCHIVE_URL", "https://archive-api.open-meteo.com/v1/archive"
)
GEOENV_TIMEOUT = float(os.getenv("GEOENV_TIMEOUT", "8"))
ELEVATION_ENABLED = os.getenv("ELEVATION_ENABLED", "1") not in ("0", "false", "False")
CLIMATE_ENABLED = os.getenv("CLIMATE_ENABLED", "1") not in ("0", "false", "False")
_USER_AGENT = os.getenv("GEOCODER_USER_AGENT", "ImageLocatorBackend/1.0")


@cached(cache_empty=False)
def elevation_m(latitude: float, longitude: float) -> Optional[float]:
    """Ground elevation in metres at the coordinate (or None)."""
    if not ELEVATION_ENABLED or latitude is None or longitude is None:
        return None
    try:
        resp = requests.get(
            ELEV_URL,
            params={"locations": f"{round(latitude, 4)},{round(longitude, 4)}"},
            headers={"User-Agent": _USER_AGENT},
            timeout=GEOENV_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("results") or []
        if results and results[0].get("elevation") is not None:
            return round(float(results[0]["elevation"]), 1)
    except Exception:
        return None
    return None


@cached(cache_empty=False)
def annual_climate(latitude: float, longitude: float) -> Optional[dict[str, float]]:
    """Coarse annual climate: mean temperature (C) and total precip (mm)."""
    if not CLIMATE_ENABLED or latitude is None or longitude is None:
        return None
    # A recent full-year window that the archive is guaranteed to have.
    end = _dt.date.today().replace(day=1) - _dt.timedelta(days=1)
    start = end - _dt.timedelta(days=365)
    try:
        resp = requests.get(
            CLIMATE_URL,
            params={
                "latitude": round(latitude, 4),
                "longitude": round(longitude, 4),
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "daily": "temperature_2m_mean,precipitation_sum",
                "timezone": "UTC",
            },
            timeout=GEOENV_TIMEOUT,
        )
        resp.raise_for_status()
        daily = resp.json().get("daily") or {}
        temps = [t for t in (daily.get("temperature_2m_mean") or []) if t is not None]
        precs = [p for p in (daily.get("precipitation_sum") or []) if p is not None]
        if not temps:
            return None
        return {
            "mean_temp_c": round(sum(temps) / len(temps), 1),
            "annual_precip_mm": round(sum(precs), 1),
        }
    except Exception:
        return None


def climate_consistency(
    scene: Optional[dict[str, Any]],
    elevation: Optional[float],
    climate: Optional[dict[str, float]],
) -> dict[str, Any]:
    """Compare scene claims (snow/desert/forest) with elevation + climate."""
    scene = scene or {}
    climate = climate or {}
    mean_temp = climate.get("mean_temp_c")
    precip = climate.get("annual_precip_mm")
    report: dict[str, Any] = {
        "status": "unknown",
        "note": "",
        "elevation_m": elevation,
        "mean_temp_c": mean_temp,
        "annual_precip_mm": precip,
    }

    checks: list[str] = []
    weak = False
    any_check = False

    if scene.get("desert") and precip is not None:
        any_check = True
        if precip < 400:
            checks.append("arid climate matches a desert scene")
        else:
            weak = True
            checks.append(f"desert claimed but ~{precip:.0f} mm/yr precip (not arid)")

    if scene.get("snow") and (mean_temp is not None or elevation is not None):
        any_check = True
        cold = mean_temp is not None and mean_temp < 10
        high = elevation is not None and elevation > 1500
        if cold or high:
            checks.append("cold/high terrain matches snow")
        else:
            weak = True
            checks.append("snow claimed but mild climate and low elevation")

    if scene.get("forest") and precip is not None:
        any_check = True
        if precip >= 300:
            checks.append("precipitation consistent with forest")
        else:
            weak = True
            checks.append("forest claimed but very dry climate")

    if not any_check:
        report["note"] = "No climate-checkable scene features."
        return report

    report["status"] = "weak_mismatch" if weak else "consistent"
    report["note"] = "; ".join(checks)
    return report
