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

    def test_region_centre_does_not_replace_the_models_coordinates(self):
        # Seen on a benchmark photo of San Pedro de Atacama: the model put the
        # town within 0.2 km, but the geocoder's first hit for the name is the
        # commune, a 279 km-wide area centred 58 km away in the mountains. The
        # mountain in the photo (a volcano 30 km off) is not near the town, so
        # the map check passed the commune centre and the right answer lost.
        town = {"name": "San Pedro de Atacama", "country": "Chile",
                "latitude": -22.9087, "longitude": -68.1997,
                "confidence": 0.99, "top_votes": 1.0}
        commune = {"name": "San Pedro de Atacama", "latitude": -23.354549,
                   "longitude": -67.902657, "addresstype": "city",
                   "osm_type": "administrative", "extent_km": 279.16}
        town_hit = {"name": "San Pedro de Atacama", "latitude": -22.910832,
                    "longitude": -68.200138, "addresstype": "town",
                    "osm_type": "town", "extent_km": 12.09}

        for far_status in ("verified", "unavailable"):
            def fake_verify(lat, lon, expected, radius=None, far_status=far_status):
                status = far_status if lat < -23 else "mismatch"
                return {"status": status, "note": "", "match_score": 0.5,
                        "missing": [], "confirmed": []}

            with self.subTest(region_centre=far_status), \
                    mock.patch.object(osint, "forward_geocode_candidates",
                                      return_value=[commune, town_hit]), \
                    mock.patch.object(osint, "verify_location", fake_verify), \
                    mock.patch.object(osint, "verify_named_and_water",
                                      lambda lat, lon, *a, **k: _named()):
                sel = osint._select_location([town], 0.0, ["mountain"], {})
                self.assertGreater(sel["chosen"]["latitude"], -23.0)
                self.assertNotIn((commune["latitude"], commune["longitude"]),
                                 [(a["latitude"], a["longitude"]) for a in sel["alternatives"]])

    def test_name_only_candidate_still_uses_the_geocoded_place(self):
        # Without model coordinates the geocoder is the only way to a point,
        # however large the place is.
        name_only = {"name": "San Pedro de Atacama", "country": "Chile",
                     "confidence": 0.6, "top_votes": 1.0}
        commune = {"name": "San Pedro de Atacama", "latitude": -23.354549,
                   "longitude": -67.902657, "addresstype": "city",
                   "osm_type": "administrative", "extent_km": 279.16}
        with mock.patch.object(osint, "forward_geocode_candidates", return_value=[commune]):
            hyps = osint._hypotheses_for_candidate(name_only, 0.0)
        self.assertEqual([(h["latitude"], h["longitude"]) for h in hyps],
                         [(commune["latitude"], commune["longitude"])])

    def test_lake_beach_does_not_expect_a_sea_coast(self):
        lake_beach = {"scene": {"beach": True, "lake": True, "forest": True}}
        sea_beach = {"scene": {"beach": True, "ocean_or_sea": True}}
        self.assertEqual(detect_expected_features(lake_beach), ["water", "forest"])
        self.assertIn("coast", detect_expected_features(sea_beach))


if __name__ == "__main__":
    unittest.main()
