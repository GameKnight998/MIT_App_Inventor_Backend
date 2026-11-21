"""Tests for the map check: mirror list, query size, disk cache, elevation fallback.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import utils.verify as V  # noqa: E402


class VerifyTests(unittest.TestCase):
    def test_regional_or_restricted_mirrors_are_not_used(self):
        hosts = " ".join(V.OVERPASS_ENDPOINTS)
        for bad in ("osm.ch", "openstreetmap.fr", "mail.ru"):
            self.assertNotIn(bad, hosts)

    def test_query_asks_only_for_needed_features(self):
        q = V._build_query(47.6, -122.1, 3000, ["mountain"])
        self.assertIn("peak", q)
        self.assertNotIn('"natural"="wood"', q)
        self.assertNotIn("out tags center 200", q)
        self.assertIn("coastline", V._build_query(47.6, -122.1, 3000, ["water"]))  # coast counts as water

    def test_elevation_fallback_when_map_service_is_down(self):
        steep = [2000.0] + [2600.0, 1800.0] * 4
        flat = [600.0] + [610.0, 595.0] * 4
        with mock.patch("utils.geoenv.elevations", return_value=steep):
            r = V._elevation_fallback(36.27, -115.69, ["mountain", "forest"], 3000, "down")
        self.assertEqual(r["confirmed"], ["mountain"])
        self.assertEqual(r["unchecked"], ["forest"])
        self.assertEqual(r["status"], "partial")
        with mock.patch("utils.geoenv.elevations", return_value=flat):
            r = V._elevation_fallback(36.17, -115.14, ["mountain"], 3000, "down")
        self.assertEqual(r["status"], "mismatch")
        self.assertIsNone(V._elevation_fallback(47.6, -122.1, ["water"], 3000, "down"))

    def test_successful_answers_survive_a_restart_via_disk(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(V, "VERIFY_DISK_CACHE", str(Path(d) / "c.json")), \
                    mock.patch.object(V, "_disk", None):
                V._disk_set(("verify", 1.0, 2.0, ("water",), 3000), {"status": "verified"})
                V._disk = None  # simulate a fresh process
                self.assertEqual(V._disk_get(("verify", 1.0, 2.0, ("water",), 3000)), {"status": "verified"})

    def test_outage_falls_back_instead_of_unavailable(self):
        with mock.patch.object(V, "_query_overpass", return_value=(None, "down")), \
                mock.patch.object(V, "VERIFY_DISK_CACHE", ""), \
                mock.patch("utils.geoenv.elevations", return_value=[2000.0] + [2600.0, 1800.0] * 4):
            r = V.verify_location(36.2701, -115.6901, ["mountain"])
        self.assertEqual(r["status"], "verified")
        self.assertEqual(r["source"], "elevation")


if __name__ == "__main__":
    unittest.main()
