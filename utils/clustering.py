"""Combine several images' results into one answer for a case.

One photo gives one guess. Several photos of the same place give something
stronger: agreement. If four images independently land within a few kilometres of
each other, that convergence is far better evidence than any single image's
confidence score, because the errors of independent guesses do not usually
coincide.

Two things are computed here:

  * clusters - primary results grouped by proximity, with confidence combined as
    noisy-OR (1 - product of misses). That is the right form for independent
    corroborating evidence: two 0.6 guesses agreeing give 0.84, not 1.2, and the
    value can never exceed certainty.
  * a heatmap - every candidate point from every image, weighted, ready for a map
    layer. Alternatives are included at reduced weight so the client can show the
    whole search space rather than only the winners.

A cluster containing one image is reported too, and clearly marked, so a single
outlier is visible instead of silently dropped.
"""

from __future__ import annotations

import math
import os
from typing import Any, Optional

# Primary results within this distance are treated as the same place.
CASE_CLUSTER_RADIUS_KM = float(os.getenv("CASE_CLUSTER_RADIUS_KM", "10"))
# Weight applied to non-winning candidates when building the heatmap.
_ALTERNATE_WEIGHT = 0.4


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(min(1.0, math.sqrt(a)))


def _noisy_or(confidences: list[float]) -> float:
    """Combine independent agreeing confidences without ever exceeding 1.0."""
    miss = 1.0
    for conf in confidences:
        miss *= 1.0 - max(0.0, min(1.0, conf))
    return round(1.0 - miss, 3)


def _centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Mean position, computed in 3D so it behaves near the date line and poles."""
    x = y = z = 0.0
    for lat, lon in points:
        rlat, rlon = math.radians(lat), math.radians(lon)
        x += math.cos(rlat) * math.cos(rlon)
        y += math.cos(rlat) * math.sin(rlon)
        z += math.sin(rlat)
    n = len(points)
    x, y, z = x / n, y / n, z / n
    lon = math.atan2(y, x)
    lat = math.atan2(z, math.sqrt(x * x + y * y))
    return round(math.degrees(lat), 6), round(math.degrees(lon), 6)


def cluster_multi_image_candidates(
    items: list[dict[str, Any]], radius_km: Optional[float] = None
) -> dict[str, Any]:
    """Group per-image results into geographic clusters and pick a consensus.

    Each item needs `latitude`, `longitude`, `confidence` and a label; anything
    without coordinates is reported separately as unlocated rather than dropped.
    """
    radius = float(radius_km or CASE_CLUSTER_RADIUS_KM)

    located = [
        item
        for item in items
        if item.get("latitude") is not None and item.get("longitude") is not None
    ]
    unlocated = [
        item.get("filename")
        for item in items
        if item.get("latitude") is None or item.get("longitude") is None
    ]

    # Highest-confidence results seed clusters, so a cluster's centre starts from
    # the best evidence rather than whichever image was uploaded first.
    located.sort(key=lambda i: float(i.get("confidence") or 0.0), reverse=True)

    clusters: list[dict[str, Any]] = []
    for item in located:
        lat = float(item["latitude"])
        lon = float(item["longitude"])
        placed = False
        for cluster in clusters:
            if _haversine_km(cluster["latitude"], cluster["longitude"], lat, lon) <= radius:
                cluster["_points"].append((lat, lon))
                cluster["_confidences"].append(float(item.get("confidence") or 0.0))
                cluster["members"].append(
                    {
                        "filename": item.get("filename"),
                        "location_name": item.get("location_name"),
                        "confidence": round(float(item.get("confidence") or 0.0), 3),
                        "latitude": lat,
                        "longitude": lon,
                    }
                )
                clat, clon = _centroid(cluster["_points"])
                cluster["latitude"] = clat
                cluster["longitude"] = clon
                placed = True
                break
        if placed:
            continue
        clusters.append(
            {
                "latitude": lat,
                "longitude": lon,
                "location_name": item.get("location_name"),
                "country": item.get("country"),
                "region": item.get("region"),
                "members": [
                    {
                        "filename": item.get("filename"),
                        "location_name": item.get("location_name"),
                        "confidence": round(float(item.get("confidence") or 0.0), 3),
                        "latitude": lat,
                        "longitude": lon,
                    }
                ],
                "_points": [(lat, lon)],
                "_confidences": [float(item.get("confidence") or 0.0)],
            }
        )

    for cluster in clusters:
        points = cluster.pop("_points")
        confidences = cluster.pop("_confidences")
        cluster["images_agreeing"] = len(points)
        cluster["combined_confidence"] = _noisy_or(confidences)
        cluster["spread_km"] = round(
            max(
                (
                    _haversine_km(cluster["latitude"], cluster["longitude"], lat, lon)
                    for lat, lon in points
                ),
                default=0.0,
            ),
            2,
        )
        cluster["map_url"] = (
            f"https://www.google.com/maps?q={cluster['latitude']},{cluster['longitude']}"
        )

    # Agreement first, then combined confidence: three images agreeing beats one
    # confident image, which is the entire point of analysing a case together.
    clusters.sort(
        key=lambda c: (c["images_agreeing"], c["combined_confidence"]), reverse=True
    )

    consensus: Optional[dict[str, Any]] = None
    if clusters:
        top = clusters[0]
        needed = max(2, math.ceil(len(located) / 2)) if len(located) > 1 else 1
        if top["images_agreeing"] >= needed:
            consensus = {
                "latitude": top["latitude"],
                "longitude": top["longitude"],
                "location_name": top.get("location_name"),
                "country": top.get("country"),
                "images_agreeing": top["images_agreeing"],
                "of_images": len(located),
                "combined_confidence": top["combined_confidence"],
                "spread_km": top["spread_km"],
                "map_url": top["map_url"],
                "note": (
                    f"{top['images_agreeing']} of {len(located)} located image(s) "
                    f"agree within {radius:.0f} km"
                    + (f" near {top['location_name']}" if top.get("location_name") else "")
                    + "."
                ),
            }

    heatmap: list[list[float]] = []
    for item in located:
        heatmap.append(
            [
                round(float(item["latitude"]), 6),
                round(float(item["longitude"]), 6),
                round(float(item.get("confidence") or 0.0), 3),
            ]
        )
        for alt in item.get("alternatives") or []:
            if alt.get("latitude") is None or alt.get("longitude") is None:
                continue
            heatmap.append(
                [
                    round(float(alt["latitude"]), 6),
                    round(float(alt["longitude"]), 6),
                    round(
                        float(alt.get("confidence") or 0.0) * _ALTERNATE_WEIGHT, 3
                    ),
                ]
            )

    return {
        "radius_km": radius,
        "images_total": len(items),
        "images_located": len(located),
        "images_unlocated": unlocated,
        "clusters": clusters,
        "consensus": consensus,
        "heatmap": heatmap,
    }
