"""Unit tests for consensus sampling, landmark filtering and confidence fusion.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.consensus import merge_samples  # noqa: E402
from utils.landmarks import _looks_generic, extract_named_features  # noqa: E402
from utils.osint import _DEFAULT_WEIGHTS, _fuse_confidence, fusion_features  # noqa: E402


def sample(cands, scene=None, **extra):
    return {
        "available": True,
        "candidates": [
            {"name": n, "latitude": lat, "longitude": lon, "confidence": c}
            for n, lat, lon, c in cands
        ],
        "scene": scene or {},
        "confidence": cands[0][3] if cands else 0.0,
        **extra,
    }


class ConsensusTests(unittest.TestCase):
    def test_agreeing_samples_give_full_support(self):
        merged = merge_samples(
            [
                sample([("Chelan", 47.84, -120.02, 0.6)]),
                sample([("Lake Chelan", 47.85, -120.05, 0.5)]),  # 3 km away
                sample([("Chelan, WA", 47.83, -120.01, 0.7)]),
            ]
        )
        self.assertEqual(merged["consensus"]["samples"], 3)
        self.assertEqual(merged["consensus"]["top_agreement"], 1.0)
        self.assertEqual(merged["candidates"][0]["top_votes"], 1.0)
        self.assertAlmostEqual(merged["candidates"][0]["confidence"], 0.6, places=2)

    def test_scattered_samples_give_low_support(self):
        merged = merge_samples(
            [
                sample([("Oregon", 44.0, -120.5, 0.5)]),
                sample([("Spain", 40.4, -3.7, 0.5)]),
                sample([("Chile", -33.4, -70.6, 0.5)]),
            ]
        )
        self.assertAlmostEqual(merged["consensus"]["top_agreement"], 0.333, places=2)
        # A candidate only one sample named is averaged down, not taken at face value.
        self.assertAlmostEqual(merged["candidates"][0]["confidence"], 0.167, places=2)

    def test_scene_feature_needs_majority(self):
        merged = merge_samples(
            [
                sample([("A", 1, 1, 0.5)], scene={"lake": True, "forest": True}),
                sample([("A", 1, 1, 0.5)], scene={"lake": False, "forest": True}),
                sample([("A", 1, 1, 0.5)], scene={"lake": False, "forest": True}),
            ]
        )
        self.assertFalse(merged["scene"]["lake"])
        self.assertTrue(merged["scene"]["forest"])

    def test_single_sample_has_no_agreement_signal(self):
        merged = merge_samples([sample([("A", 1, 1, 0.5)])])
        self.assertIsNone(merged["consensus"]["top_agreement"])
        self.assertNotIn("top_votes", merged["candidates"][0])


class LandmarkFilterTests(unittest.TestCase):
    def test_generic_descriptions_are_detected(self):
        for text in ("a gothic clock tower", "the old bridge", "suspension bridge", "Two towers"):
            self.assertTrue(_looks_generic(text), text)
        for text in ("Space Needle", "Golden Gate Bridge", "Lake Chelan"):
            self.assertFalse(_looks_generic(text), text)

    def test_named_landmarks_field_is_preferred(self):
        vision = {"landmarks": ["a clock tower", "Big Ben"], "named_landmarks": ["Big Ben"]}
        self.assertEqual(extract_named_features(vision), ["Big Ben"])

    def test_city_names_are_not_natural_features(self):
        from utils.landmarks import _looks_like_named_feature

        for name in ("Seattle", "Parkland", "Baytown", "Seaside Heights Mall"):
            self.assertFalse(_looks_like_named_feature(name), name)
        for name in ("Lake Chelan", "Mt. Rainier", "Golden Gate Bridge"):
            self.assertTrue(_looks_like_named_feature(name), name)

    def test_candidate_names_are_not_checked_against_themselves(self):
        vision = {"named_landmarks": [], "candidates": [{"name": "Lake Chelan"}]}
        self.assertEqual(extract_named_features(vision), [])

    def test_legacy_landmarks_drop_generic_entries(self):
        vision = {"landmarks": ["a clock tower", "Big Ben"]}
        self.assertEqual(extract_named_features(vision), ["Big Ben"])


class FusionTests(unittest.TestCase):
    base = {"ai_conf": 0.55, "clue_bonus": 0.14, "ambiguity_gap": 0.2, "verify_status": "skipped"}

    def test_agreement_raises_and_disagreement_lowers_confidence(self):
        none = _fuse_confidence(self.base)
        agree = _fuse_confidence({**self.base, "agreement": 1.0})
        split = _fuse_confidence({**self.base, "agreement": 1 / 3})
        self.assertGreater(agree, none + 0.2)
        self.assertLess(split, none)

    def test_features_match_default_weight_names(self):
        feats = fusion_features(
            {
                **self.base,
                "verify_status": "verified",
                "solar_status": "consistent",
                "landmark_named": "missed",
                "agreement": 0.5,
                "precision_m": 500,
                "landmark_corroborated": True,
            }
        )
        for name in feats:
            self.assertIn(name, _DEFAULT_WEIGHTS, name)

    def test_calibrated_weights_override_defaults(self):
        w = dict(_DEFAULT_WEIGHTS, bias=2.0)
        self.assertGreater(_fuse_confidence(self.base, w), _fuse_confidence(self.base))


if __name__ == "__main__":
    unittest.main()
