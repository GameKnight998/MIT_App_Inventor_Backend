"""Benchmark bookkeeping: cost, results.csv upgrades, refusal stop.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import benchmark  # noqa: E402


class Cost(unittest.TestCase):
    def test_priced_model(self):
        # Sonnet 5.5: $2 in / $10 out per 1M tokens.
        c = benchmark.cost_usd("claude-sonnet-5-5", {"input_tokens": 1_000_000, "output_tokens": 100_000})
        self.assertAlmostEqual(c, 3.0)

    def test_label_suffix_uses_base_price(self):
        a = benchmark.cost_usd("claude-haiku-5-5 (quick)", {"input_tokens": 2000, "output_tokens": 500})
        b = benchmark.cost_usd("claude-haiku-5-5", {"input_tokens": 2000, "output_tokens": 500})
        self.assertEqual(a, b)

    def test_unpriced_model_is_blank_not_zero(self):
        self.assertIsNone(benchmark.cost_usd("no-such-model", {"input_tokens": 10, "output_tokens": 10}))


class Run(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "m.csv").write_text(
            "file,latitude,longitude,country,place,category\n"
            + "".join(f"p{i}.jpg,1,2,X,Y,z\n" for i in range(5)), encoding="utf-8")

    def _run(self, fake, **kw):
        with mock.patch.object(benchmark, "run_one", side_effect=fake), \
                mock.patch.object(benchmark, "report"):
            benchmark.run(self.tmp / "m.csv", ["m"], self.tmp / "out", **kw)
        with (self.tmp / "out" / "results.csv").open(newline="") as f:
            return list(csv.DictReader(f))

    def test_stops_calling_a_model_that_only_refuses(self):
        calls = []

        def refuse(photo, model):
            calls.append(photo["file"])
            return {"file": photo["file"], "model": model, "error": "declined the request (refusal)."}

        rows = self._run(refuse, stop_after_refusals=2)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(rows), 5)  # skipped photos still get a (failed) row
        self.assertTrue(all(r["error"] for r in rows))

    def test_a_model_that_answered_is_never_stopped(self):
        calls = []

        def mixed(photo, model):
            calls.append(photo["file"])
            err = "" if photo["file"] == "p0.jpg" else "declined the request (refusal)."
            return {"file": photo["file"], "model": model, "error": err}

        self._run(mixed, stop_after_refusals=2)
        self.assertEqual(len(calls), 5)

    def test_old_results_file_is_upgraded_before_appending(self):
        out = self.tmp / "out"
        out.mkdir()
        old = [f for f in benchmark.FIELDS if f not in benchmark.USAGE_FIELDS
               and f not in ("cost_usd", "vision_samples", "claude_effort")]
        with (out / "results.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=old)
            w.writeheader()
            w.writerow({"file": "p0.jpg", "model": "m", "error_km": "3.5", "error": ""})

        def ok(photo, model):
            return {"file": photo["file"], "model": model, "error": "", "input_tokens": 100, "cost_usd": 0.01}

        rows = self._run(ok)
        self.assertEqual(len(rows), 5)  # p0 kept, p1..p4 added
        self.assertEqual(rows[0]["error_km"], "3.5")
        self.assertEqual(rows[1]["input_tokens"], "100")


if __name__ == "__main__":
    unittest.main()
