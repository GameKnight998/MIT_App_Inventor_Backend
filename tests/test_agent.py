"""Unit tests for agent mode (utils/agent.py), with a fake OpenAI client.

Run: python -m unittest discover -s tests
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image  # noqa: E402

import utils.agent as agent  # noqa: E402


def _call(i, name, args=None):
    return SimpleNamespace(type="function_call", call_id=f"c{i}", name=name, arguments=json.dumps(args or {}))


def _reply(calls=None, content=""):
    output = list(calls or []) or [SimpleNamespace(type="message")]
    return SimpleNamespace(output=output, output_text=content, usage=None)


class FakeClient:
    """Replays a scripted list of model replies and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.sent = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.sent.append(kwargs)
        return self.replies.pop(0)


def _look(name, lat, lon):
    return {"available": True, "model": "gpt-test",
            "candidates": [{"name": name, "country": "United States", "latitude": lat,
                            "longitude": lon, "confidence": 0.7}],
            "scene": {}, "confidence": 0.7}


class AgentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        Image.new("RGB", (64, 48), "white").save(self.tmp.name)
        self.vision_calls = []

        def fake_samples(path, samples=None, model=None, focus=None):
            self.vision_calls.append(focus)
            return [_look("Bellevue Downtown Park", 47.6125, -122.2045)] * 3

        def fake_determine(metadata, vision, forensics=None):
            top = (vision.get("candidates") or [{}])[0]
            return {"location_name": top.get("name"), "latitude": top.get("latitude"),
                    "longitude": top.get("longitude"), "confidence": 0.8, "evidence": [],
                    "reasoning_trace": [{"stage": "Conclusion", "details": ["ok"]}]}

        patches = [
            mock.patch("utils.vision.analyze_image_samples", fake_samples),
            mock.patch("utils.osint.determine_location", fake_determine),
            mock.patch("utils.verify.verify_location", lambda *a, **k: {"status": "match", "note": ""}),
            mock.patch("utils.landmarks.verify_named_and_water",
                       lambda *a, **k: {"place_status": "match", "landmark_status": "unknown",
                                        "water_status": "unknown", "note": ""}),
            mock.patch("utils.geocode.forward_geocode",
                       lambda q: {"latitude": 47.61, "longitude": -122.20, "display_name": q}),
        ]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)

    def tearDown(self):
        Path(self.tmp.name).unlink(missing_ok=True)

    def test_agent_chooses_tools_and_scores_last(self):
        client = FakeClient([
            _reply([_call(1, "check_photo")]),
            _reply([_call(2, "read_scene")]),
            _reply([_call(3, "list_places")]),
            _reply([_call(4, "read_scene", {"focus": "the sign on the left"})]),
            _reply([_call(5, "verify_place", {"name": "Bellevue Downtown Park"})]),
            _reply([_call(6, "rank_and_score")]),
            _reply(content="Search Bellevue Downtown Park first."),
        ])
        out = agent.run_agent(self.tmp.name, "JPEG", ignore_gps=True, model="gpt-test", client=client)
        tools = [s["tool"] for s in out["agent"]["steps"]]
        self.assertEqual(tools, ["check_photo", "read_scene", "list_places", "read_scene",
                                 "verify_place", "rank_and_score"])
        self.assertEqual(self.vision_calls, [None, "the sign on the left"])
        self.assertEqual(out["vision"]["consensus"]["samples"], 6)  # both looks merged
        self.assertEqual(out["location"]["location_name"], "Bellevue Downtown Park")
        self.assertEqual(out["agent"]["brief"], "Search Bellevue Downtown Park first.")
        self.assertEqual(out["location"]["reasoning_trace"][0]["stage"], "Agent tool calls")
        # Tool results were fed back to the model before its next decision.
        tool_msgs = [m for m in client.sent[-1]["input"]
                     if isinstance(m, dict) and m.get("type") == "function_call_output"]
        self.assertEqual(len(tool_msgs), 6)

    def test_pipeline_finishes_when_agent_stops_early(self):
        client = FakeClient([_reply([_call(1, "check_photo")]), _reply(content="Done.")])
        out = agent.run_agent(self.tmp.name, "JPEG", ignore_gps=True, model="gpt-test", client=client)
        tools = [s["tool"] for s in out["agent"]["steps"]]
        self.assertEqual(tools, ["check_photo", "read_scene", "rank_and_score"])
        self.assertEqual(out["location"]["location_name"], "Bellevue Downtown Park")

    def test_tool_limits(self):
        client = FakeClient([
            _reply([_call(1, "read_scene"), _call(2, "read_scene"), _call(3, "read_scene")]),
            _reply([_call(4, "rank_and_score"), _call(5, "rank_and_score")]),
            _reply(content="ok"),
        ])
        out = agent.run_agent(self.tmp.name, "JPEG", ignore_gps=True, model="gpt-test", client=client)
        self.assertEqual(len(self.vision_calls), 2)
        self.assertIn("limit", out["agent"]["steps"][2]["summary"])
        self.assertIn("limit", out["agent"]["steps"][4]["summary"])

    def test_non_openai_model_uses_fixed_order(self):
        out = agent.run_agent(self.tmp.name, "JPEG", ignore_gps=True, model="claude-sonnet-5-5")
        tools = [s["tool"] for s in out["agent"]["steps"]]
        self.assertEqual(tools, ["check_photo", "read_scene", "rank_and_score"])
        self.assertIn("OpenAI models only", out["agent"]["note"])


if __name__ == "__main__":
    unittest.main()
