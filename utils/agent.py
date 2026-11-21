"""Agent mode: the vision model runs the five pipeline stages as tool calls.

In the fixed pipeline (main._run_pipeline) Python calls every stage in the same
order for every photo. Here the AI model gets the five stages as tools and
decides which one to call next, after reading the previous tool's result:

  1. check_photo     - EXIF metadata and file provenance (utils/exif, forensics)
  2. read_scene      - several independent looks at the photo (utils/vision),
                       optionally focused on a detail the agent wants re-read
  3. list_places     - the candidate places and how many looks agree on each
  4. verify_place    - map checks for one place (utils/verify, utils/landmarks)
  5. rank_and_score  - the calibrated fusion step (utils/osint.determine_location)

So the agent can, for example, skip the scene reading when the photo carries
real GPS, take a second look focused on a half-read sign when the first looks
disagree, or check a runner-up place on the map before scoring.

The final location and confidence always come from rank_and_score, the same
tested, calibrated scoring the fixed pipeline uses: the agent chooses what
evidence to gather, not the confidence number.

Only OpenAI models are supported as the agent for now; for other models
run_agent() falls back to the fixed pipeline order (and says so).
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Optional

AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "10"))
# Caps per tool, so a confused agent cannot run up the bill.
_TOOL_LIMITS = {"read_scene": 2, "verify_place": 4}

_SYSTEM = """You are the planning agent of Whereabouts, a tool that helps searchers \
find where a photo of a missing person was taken. You cannot see GPS or maps \
yourself: you work only through five tools, one stage each.

1. check_photo: the photo's hidden data (time, camera, GPS) and whether it looks \
like an original camera photo, a screenshot or a stripped export.
2. read_scene: several independent expert looks at the photo, returning the clues \
(text, signs, landmarks, terrain, plants). Pass `focus` to ask for a targeted \
second look at a detail, e.g. a sign that was only partly read.
3. list_places: the candidate places from all looks so far, with `support` = the \
share of looks whose first choice was that place.
4. verify_place: checks one place against maps (named landmarks, water, terrain \
features the photo shows).
5. rank_and_score: fuses all evidence into the final ranked place and a \
calibrated confidence. Call it exactly once, last.

Plan like an OSINT analyst. Always start with check_photo. If it finds real GPS, \
you may go straight to rank_and_score. Otherwise read the scene and list places. \
If the looks disagree (top support below 0.67) or a clue is half-read, take one \
focused second look. Verify a runner-up when it is close to the leader. \
Do not repeat a tool call with the same arguments. After rank_and_score, reply \
with a two-sentence search brief for the family in plain text (no markdown) and stop."""

_TOOL_SPECS: list[tuple[str, str, dict[str, Any]]] = [
    ("check_photo",
     "Stage 1. Read the photo's metadata: capture time, camera, GPS, and file provenance.",
     {"type": "object", "properties": {}, "additionalProperties": False}),
    ("read_scene",
     "Stage 2. Independent expert looks at the photo (with zoomed-in crops). "
     "Returns the geolocation clues found.",
     {"type": "object", "properties": {
         "focus": {"type": "string",
                   "description": "Optional detail for a targeted second look, e.g. "
                                  "'the partly hidden shop sign on the left'."}},
      "additionalProperties": False}),
    ("list_places",
     "Stage 3. Candidate places from all looks so far, with how many looks agree.",
     {"type": "object", "properties": {}, "additionalProperties": False}),
    ("verify_place",
     "Stage 4. Check one candidate place against map data.",
     {"type": "object", "properties": {
         "name": {"type": "string", "description": "Place name as listed by list_places."},
         "latitude": {"type": "number"},
         "longitude": {"type": "number"}},
      "required": ["name"], "additionalProperties": False}),
    ("rank_and_score",
     "Stage 5. Fuse all evidence into the final ranked place and calibrated "
     "confidence. Call once, last.",
     {"type": "object", "properties": {}, "additionalProperties": False}),
]
# OpenAI Responses API format (GPT-6.x models take function tools only there).
_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "name": n, "description": d, "parameters": p} for n, d, p in _TOOL_SPECS
]


def _short(value: Any, limit: int = 12) -> Any:
    """Trim lists so tool results stay small for the agent."""
    if isinstance(value, list):
        return [v for v in value if v not in (None, "")][:limit]
    return value


class _Run:
    """The evidence gathered so far for one photo, and the tool implementations."""

    def __init__(self, image_path: str, fmt: Optional[str], ignore_gps: bool, model: str):
        self.image_path, self.fmt, self.ignore_gps, self.model = image_path, fmt, ignore_gps, model
        self.metadata: Optional[dict[str, Any]] = None
        self.forensics: Optional[dict[str, Any]] = None
        self.samples: list[dict[str, Any]] = []
        self.vision: dict[str, Any] = {}
        self.location: Optional[dict[str, Any]] = None
        self.verified: list[dict[str, Any]] = []

    # -- helpers -----------------------------------------------------------
    def _ensure_metadata(self) -> None:
        if self.metadata is None:
            from utils.exif import extract_exif
            from utils.forensics import assess_forensics

            metadata = extract_exif(self.image_path)
            if self.ignore_gps:
                metadata = {**metadata, "gps": None}
            self.metadata = metadata
            self.forensics = assess_forensics(self.image_path, self.fmt, metadata)

    def _vision_or_empty(self) -> dict[str, Any]:
        if self.vision:
            return self.vision
        from utils.vision import _empty_analysis

        return _empty_analysis("The agent did not read the scene.")

    # -- tools ---------------------------------------------------------------
    def check_photo(self) -> dict[str, Any]:
        self._ensure_metadata()
        md, fx = self.metadata or {}, self.forensics or {}
        return {
            "gps_found": md.get("gps") is not None,
            "gps": md.get("gps"),
            "taken_at": md.get("timestamp"),
            "camera": md.get("camera"),
            "provenance_notes": _short(fx.get("notes"), 4),
        }

    def read_scene(self, focus: Optional[str] = None) -> dict[str, Any]:
        from utils.consensus import merge_samples
        from utils.vision import analyze_image_samples

        new = analyze_image_samples(self.image_path, model=self.model, focus=focus)
        usable = [s for s in new if s.get("available")]
        if not usable:
            return {"error": (new[0].get("note") if new else "no answer")}
        self.samples.extend(usable)
        self.vision = merge_samples(self.samples)
        v = self.vision
        return {
            "looks_so_far": len(self.samples),
            "text_read": _short(v.get("ocr_text")),
            "signage": _short(v.get("signage")),
            "named_landmarks": _short(v.get("named_landmarks")),
            "terrain": v.get("terrain"),
            "vegetation": v.get("vegetation"),
            "architecture": v.get("architecture"),
            "scene": [k for k, on in (v.get("scene") or {}).items() if on],
            "reasoning": (v.get("reasoning") or "")[:600],
        }

    def list_places(self) -> dict[str, Any]:
        if not self.vision:
            return {"error": "No looks yet: call read_scene first."}
        places = [
            {
                "name": c.get("name"), "country": c.get("country"),
                "latitude": c.get("latitude"), "longitude": c.get("longitude"),
                "support": c.get("top_votes"), "model_confidence": c.get("confidence"),
            }
            for c in (self.vision.get("candidates") or [])[:6]
        ]
        consensus = self.vision.get("consensus") or {}
        return {"places": places, "looks": consensus.get("samples"),
                "country_agreement": consensus.get("country_agreement")}

    def verify_place(self, name: str, latitude: Optional[float] = None,
                     longitude: Optional[float] = None) -> dict[str, Any]:
        from utils.geocode import forward_geocode
        from utils.landmarks import verify_named_and_water
        from utils.verify import detect_expected_features, verify_location

        found = None
        if latitude is None or longitude is None:
            found = forward_geocode(name)
            if not found:
                return {"name": name, "error": "Place not found on the map."}
            latitude, longitude = found["latitude"], found["longitude"]
        vision = self._vision_or_empty()
        expected = detect_expected_features(vision)
        features = verify_location(latitude, longitude, expected)
        named = verify_named_and_water(latitude, longitude, vision, expected, place_name=name)
        out = {
            "name": name, "latitude": latitude, "longitude": longitude,
            "map_name": (found or {}).get("display_name"),
            "features_expected": expected,
            "features_status": features.get("status"),
            "features_note": features.get("note"),
            "place_name_status": named.get("place_status"),
            "landmark_status": named.get("landmark_status"),
            "water_status": named.get("water_status"),
            "note": named.get("note"),
        }
        self.verified.append(out)
        return out

    def rank_and_score(self) -> dict[str, Any]:
        from utils.osint import determine_location

        self._ensure_metadata()
        self.location = determine_location(self.metadata or {}, self._vision_or_empty(), self.forensics)
        loc = self.location
        return {
            "location": loc.get("location_name"), "country": loc.get("country"),
            "latitude": loc.get("latitude"), "longitude": loc.get("longitude"),
            "confidence": loc.get("confidence"),
            "confidence_by_distance": loc.get("confidence_by_distance"),
            "map_check": (loc.get("verification") or {}).get("status"),
            "alternatives": [a.get("name") for a in (loc.get("alternatives") or [])[:3]],
        }

    def call(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        tools: dict[str, Callable[..., dict[str, Any]]] = {
            "check_photo": self.check_photo, "read_scene": self.read_scene,
            "list_places": self.list_places, "verify_place": self.verify_place,
            "rank_and_score": self.rank_and_score,
        }
        fn = tools.get(name)
        if fn is None:
            return {"error": f"Unknown tool {name}."}
        try:
            return fn(**args)
        except TypeError as exc:
            return {"error": f"Bad arguments for {name}: {exc}"}
        except Exception as exc:  # a failing stage must not kill the run
            return {"error": f"{name} failed: {exc}"}


def _summary(name: str, args: dict[str, Any], result: dict[str, Any]) -> str:
    """One readable line per tool call, for the reasoning trace."""
    if result.get("error"):
        return f"{name}: {result['error']}"
    if name == "check_photo":
        return "check_photo: " + ("GPS found." if result.get("gps_found") else "no GPS in the photo.")
    if name == "read_scene":
        focus = f" focused on “{args['focus']}”" if args.get("focus") else ""
        return f"read_scene{focus}: {result.get('looks_so_far')} looks so far."
    if name == "list_places":
        places = result.get("places") or []
        top = ", ".join(f"{p['name']} ({p.get('support')})" for p in places[:3])
        return f"list_places: {top or 'none'}."
    if name == "verify_place":
        return (f"verify_place {args.get('name')}: features {result.get('features_status')}, "
                f"name {result.get('place_name_status')}.")
    if name == "rank_and_score":
        return f"rank_and_score: {result.get('location')} at {result.get('confidence')} confidence."
    return name


def _responses_usage(resp: Any) -> None:
    """Add a Responses API call's tokens to the shared usage counter (for cost)."""
    from utils.providers import _add_usage

    u = getattr(resp, "usage", None)
    if u is None:
        return
    cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
    _add_usage(int(getattr(u, "input_tokens", 0) or 0) - cached, getattr(u, "output_tokens", 0), cached, 0)


def _openai_client() -> Any:
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def _thumbnail(image_path: str) -> Optional[str]:
    try:
        from PIL import Image

        from utils.vision import _to_jpeg_b64

        with Image.open(image_path) as img:
            return _to_jpeg_b64(img.convert("RGB"), 768)
    except Exception:
        return None


def run_agent(image_path: str, fmt: Optional[str], ignore_gps: bool = False,
              model: Optional[str] = None, client: Any = None) -> dict[str, Any]:
    """Run the agent and return (metadata, vision, location, agent_report) as a dict."""
    from utils.providers import provider_for
    from utils.vision import VISION_MODEL

    model = model or os.getenv("AGENT_MODEL") or VISION_MODEL
    run = _Run(image_path, fmt, ignore_gps, model)
    steps: list[dict[str, Any]] = []
    brief = ""
    note = "ok"

    if provider_for(model) != "openai" and client is None:
        note = f"Agent mode supports OpenAI models only; ran the fixed stage order for {model}."
        for name in ("check_photo", "read_scene", "rank_and_score"):
            result = run.call(name, {})
            steps.append({"tool": name, "args": {}, "summary": _summary(name, {}, result)})
    else:
        client = client or _openai_client()
        user: list[dict[str, Any]] = [{"type": "input_text", "text":
            "Find where this photo was taken. Use your tools; a small preview is attached "
            "so you can plan, but rely on read_scene for the clues."}]
        thumb = _thumbnail(image_path)
        if thumb:
            user.append({"type": "input_image", "detail": "low",
                         "image_url": f"data:image/jpeg;base64,{thumb}"})
        items: list[Any] = [{"role": "user", "content": user}]
        used: dict[str, int] = {}
        for _ in range(AGENT_MAX_STEPS):
            resp = client.responses.create(model=model, instructions=_SYSTEM, input=items, tools=_TOOLS)
            _responses_usage(resp)
            output = list(getattr(resp, "output", None) or [])
            items.extend(output)  # keep reasoning and calls in the conversation
            calls = [o for o in output if getattr(o, "type", None) == "function_call"]
            if not calls:
                brief = (getattr(resp, "output_text", "") or "").strip()
                break
            for c in calls:
                name = c.name
                try:
                    args = json.loads(c.arguments or "{}") or {}
                except ValueError:
                    args = {}
                used[name] = used.get(name, 0) + 1
                if used[name] > _TOOL_LIMITS.get(name, 1 if name == "rank_and_score" else 3):
                    result = {"error": f"{name} call limit reached; move on."}
                else:
                    result = run.call(name, args)
                steps.append({"tool": name, "args": args, "summary": _summary(name, args, result)})
                items.append({"type": "function_call_output", "call_id": c.call_id,
                              "output": json.dumps(result, default=str)[:6000]})
        else:
            note = f"Agent stopped after {AGENT_MAX_STEPS} steps."

    if run.location is None:
        # The agent never scored: finish the job so the app still gets an answer.
        if not run.vision and not (run.metadata or {}).get("gps"):
            run.call("read_scene", {})
            steps.append({"tool": "read_scene", "args": {}, "summary": "read_scene (added by the pipeline)."})
        run.call("rank_and_score", {})
        steps.append({"tool": "rank_and_score", "args": {}, "summary": "rank_and_score (added by the pipeline)."})

    run._ensure_metadata()
    location = run.location or {}
    location.setdefault("reasoning_trace", [])
    location["reasoning_trace"].insert(0, {
        "stage": "Agent tool calls",
        "details": [f"{i}. {s['summary']}" for i, s in enumerate(steps, 1)],
    })
    return {
        "metadata": run.metadata or {},
        "forensics": run.forensics or {},
        "vision": run._vision_or_empty(),
        "location": location,
        "agent": {"model": model, "steps": steps, "brief": brief, "note": note,
                  "places_verified": len(run.verified)},
    }
