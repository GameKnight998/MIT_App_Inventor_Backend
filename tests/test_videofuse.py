"""Tests for combining several video frames into one answer.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.videofuse import merge_frame_visions  # noqa: E402


def frame(name, lat, lon, votes, conf=0.6):
    return {"available": True, "scene": {}, "candidates": [
        {"name": name, "latitude": lat, "longitude": lon, "confidence": conf, "top_votes": votes}]}


class VideoFuseTests(unittest.TestCase):
    def test_nearby_frames_outvote_a_distant_pair(self):
        # Three frames in one region (names differ, all within 150 km) and two
        # weakly agreed frames far away with a higher self-reported confidence.
        frames = [
            frame("Chloride Cliff", 36.79, -116.82, 0.67),
            frame("Cerro Gordo", 36.537, -117.79, 1.0),
            frame("Death Valley", 36.24, -117.08, 1.0),
            frame("Big Bend National Park", 29.25, -103.25, 0.33, conf=0.9),
            frame("Big Bend National Park", 29.25, -103.25, 0.5, conf=0.9),
        ]
        merged = merge_frame_visions(frames)
        self.assertNotEqual(merged["candidates"][0]["name"], "Big Bend National Park")
        self.assertGreater(merged["consensus"]["top_agreement"], 0.5)

    def test_support_is_weighted_by_agreement_within_each_frame(self):
        frames = [frame("A", 10.0, 10.0, 1.0), frame("B", 40.0, 40.0, 0.2)]
        merged = merge_frame_visions(frames)
        self.assertEqual(merged["candidates"][0]["name"], "A")
        self.assertAlmostEqual(merged["candidates"][0]["top_votes"], 0.5)


if __name__ == "__main__":
    unittest.main()
