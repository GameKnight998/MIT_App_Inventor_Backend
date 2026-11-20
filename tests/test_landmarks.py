"""Named-place check: an ambiguous name is judged by its nearest match.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils import landmarks  # noqa: E402

VILLAGE = (36.0544, -112.1401)  # Grand Canyon Village, South Rim
# Real geocoder answers for "Grand Canyon South Rim": a cliff ~152 km west
# ranks first; the village viewpoints come second and third.
SOUTH_RIM_HITS = [
    {"name": "Grand Canyon South Rim", "latitude": 36.025454, "longitude": -113.832114, "addresstype": "cliff"},
    {"name": "South Rim", "latitude": 36.061378, "longitude": -112.116285, "addresstype": "locality"},
    {"name": "Grand Canyon South Rim", "latitude": 36.065764, "longitude": -112.11798, "addresstype": "tourism"},
]


def check(lat, lon, hits, name="Grand Canyon South Rim"):
    with mock.patch.object(landmarks, "forward_geocode_candidates", return_value=hits), \
            mock.patch.object(landmarks, "search_nearby", return_value=[]):
        return landmarks.verify_named_and_water(lat, lon, None, [], place_name=name)


class NamedPlaceCheck(unittest.TestCase):
    def test_correct_point_kept_when_first_hit_is_elsewhere(self):
        report = check(*VILLAGE, SOUTH_RIM_HITS)
        self.assertEqual(report["place_status"], "matched")
        self.assertLess(report["place_distance_km"], 5)

    def test_point_far_from_every_match_is_still_rejected(self):
        report = check(47.6101, -122.2015, SOUTH_RIM_HITS)  # Bellevue, WA
        self.assertEqual(report["place_status"], "missed")

    def test_single_far_match_is_rejected(self):
        report = check(*VILLAGE, SOUTH_RIM_HITS[:1])
        self.assertEqual(report["place_status"], "missed")


if __name__ == "__main__":
    unittest.main()
