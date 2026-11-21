"""Tests for choosing a place when the map check cannot confirm any candidate.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.osint as osint  # noqa: E402
from utils.verify import detect_expected_features  # noqa: E402

LAKE = {"name": "Sunset Beach, Lake Sammamish State Park", "country": "United States",
        "latitude": 47.566, "longitude": -122.077, "confidence": 0.25, "top_votes": 1.0}
OTHER = {"name": "Rocky Point Park", "country": "Canada",
         "latitude": 49.2839, "longitude": -122.8462, "confidence": 0.09, "top_votes": 0.0}


def _named(place_status="unknown"):
    return {"place_status": place_status, "landmark_status": "unknown",
            "water_status": "unknown", "note": ""}


class SelectionTests(unittest.TestCase):
    def _select(self, candidates, status_for, named_for):
        """Run _select_location with map checks faked per latitude."""
        def fake_verify(lat, lon, expected, radius=None):
            return {"status": status_for(lat), "note": "", "match_score": 0.5,
                    "missing": [], "confirmed": []}

        with mock.patch.object(osint, "forward_geocode_candidates", return_value=[]), \
                mock.patch.object(osint, "verify_location", fake_verify), \
                mock.patch.object(osint, "verify_named_and_water",
                                  lambda lat, lon, *a, **k: named_for(lat)):
            return osint._select_location(candidates, 0.0, ["water"], {})

    def test_map_outage_does_not_promote_a_place_no_look_chose(self):
        # Overpass is down for both; only the unsupported place's name is found.
        sel = self._select(
            [LAKE, OTHER],
            status_for=lambda lat: "unavailable",
            named_for=lambda lat: _named("matched" if lat > 49 else "unknown"),
        )
        self.assertEqual(sel["chosen"]["name"], LAKE["name"])

    def test_map_mismatch_does_not_promote_a_place_no_look_chose(self):
        sel = self._select(
            [LAKE, OTHER],
            status_for=lambda lat: "mismatch" if lat < 49 else "partial",
            named_for=lambda lat: _named(),
        )
        self.assertEqual(sel["chosen"]["name"], LAKE["name"])

    def test_close_support_still_lets_the_map_decide(self):
        a = {**LAKE, "top_votes": 0.67}
        b = {**OTHER, "top_votes": 0.33}  # within the margin of the leader
        sel = self._select(
            [a, b],
            status_for=lambda lat: "mismatch" if lat < 49 else "partial",
            named_for=lambda lat: _named(),
        )
        self.assertEqual(sel["chosen"]["name"], OTHER["name"])

    def test_lake_beach_does_not_expect_a_sea_coast(self):
        lake_beach = {"scene": {"beach": True, "lake": True, "forest": True}}
        sea_beach = {"scene": {"beach": True, "ocean_or_sea": True}}
        self.assertEqual(detect_expected_features(lake_beach), ["water", "forest"])
        self.assertIn("coast", detect_expected_features(sea_beach))


if __name__ == "__main__":
    unittest.main()
