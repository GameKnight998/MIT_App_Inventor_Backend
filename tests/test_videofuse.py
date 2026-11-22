"""Tests for combining several video frames into one answer.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.videofuse import merge_frame_visions  # noqa: E402


def frame(name, lat, lon, votes, conf=0.6, strength=None):
    return {"available": True, "scene": {}, "clue_strength": strength, "candidates": [
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
        top = merged["candidates"][0]
        big_bend = next(c for c in merged["candidates"] if c["name"] == "Big Bend National Park")
        self.assertNotEqual(top["name"], "Big Bend National Park")
        self.assertGreater(top["top_votes"], big_bend["top_votes"])

    def test_support_is_weighted_by_agreement_within_each_frame(self):
        frames = [frame("A", 10.0, 10.0, 1.0), frame("B", 40.0, 40.0, 0.2)]
        merged = merge_frame_visions(frames)
        self.assertEqual(merged["candidates"][0]["name"], "A")
        self.assertAlmostEqual(merged["candidates"][0]["top_votes"], 0.833, places=3)  # 1.0 of 1.2 total weight

    def test_distinctive_frames_count_for_more(self):
        # Two generic ridge frames vs one frame with a narrow-range species.
        frames = [
            frame("Generic Range", 40.0, -117.0, 1.0, strength=0.1),
            frame("Generic Range", 40.0, -117.0, 1.0, strength=0.1),
            frame("Tortoise Slope", 36.1, -115.4, 1.0, strength=0.7),
        ]
        merged = merge_frame_visions(frames)
        self.assertEqual(merged["candidates"][0]["name"], "Tortoise Slope")

    def test_votes_fade_with_distance(self):
        frames = [frame("A", 36.0, -115.0, 1.0), frame("B", 36.5, -115.0, 1.0)]  # ~56 km apart
        merged = merge_frame_visions(frames)
        a = next(c for c in merged["candidates"] if c["name"] == "A")
        self.assertGreater(a["top_votes"], 0.5)   # its own vote plus part of B's
        self.assertLess(a["top_votes"], 1.0)      # but not all of B's

    def test_full_agreement_is_full_support_whatever_the_clue_strength(self):
        frames = [frame("Ridge", 36.27, -115.69, 1.0, strength=0.3) for _ in range(4)]
        merged = merge_frame_visions(frames)
        self.assertAlmostEqual(merged["candidates"][0]["top_votes"], 1.0)


if __name__ == "__main__":
    unittest.main()
