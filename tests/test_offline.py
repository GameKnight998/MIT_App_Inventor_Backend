"""Offline tests for the Raven-parity feature set.

Runs with plain Python (no pytest needed, no network, no API keys) so it can be
used as a pre-deploy smoke check:

    venv\\Scripts\\python.exe tests\\test_offline.py

Every external call is stubbed. What is actually being verified is the logic we
own: that a hallucinated street name cannot move the answer, that a synthetic
image caps confidence, that agreeing images combine correctly, and that the
refactored candidate-selection plan is equivalent to the old nested loop.
"""

from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Keep tests deterministic and independent of whatever is in .env.
os.environ.setdefault("GEOCODING_ENABLED", "0")
os.environ.setdefault("VERIFY_ENABLED", "0")
os.environ.setdefault("ENRICH_ENABLED", "0")
os.environ["CASE_DB_PATH"] = str(Path(tempfile.gettempdir()) / "il_test_cases.db")

import numpy as np
from PIL import Image, PngImagePlugin

_results: list[tuple[str, bool, str]] = []


def check(name: str):
    """Decorator that records pass/fail instead of aborting the whole run."""

    def wrap(fn):
        try:
            fn()
            _results.append((name, True, ""))
        except AssertionError as exc:
            _results.append((name, False, str(exc) or "assertion failed"))
        except Exception:
            _results.append((name, False, traceback.format_exc(limit=3)))
        return fn

    return wrap


TMP = Path(tempfile.mkdtemp(prefix="il_tests_"))


# --------------------------------------------------------------------------
# 1. Vision client: JSON extraction and media typing
# --------------------------------------------------------------------------
@check("visionclient.extract_json handles bare, fenced and prose-wrapped JSON")
def _t1():
    from utils.visionclient import extract_json

    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": 2}\n```') == {"a": 2}
    assert extract_json('Sure! Here it is:\n{"a": 3}\nHope that helps.') == {"a": 3}
    # Braces inside strings must not confuse the balance scan.
    assert extract_json('prefix {"a": "}{", "b": 4} suffix') == {"a": "}{", "b": 4}
    assert extract_json("not json at all") is None
    assert extract_json("") is None
    assert extract_json("[1,2,3]") is None, "a bare list is not a usable object"


@check("visionclient detects media type from magic bytes")
def _t2():
    from utils.visionclient import _media_type

    assert _media_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert _media_type(b"\x89PNG\r\n\x1a\nrest") == "image/png"
    assert _media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert _media_type(b"GIF89a...") == "image/gif"


# --------------------------------------------------------------------------
# 2. Image enhancement
# --------------------------------------------------------------------------
def _write_image(path: Path, array: np.ndarray) -> str:
    Image.fromarray(array.astype("uint8")).save(path)
    return str(path)


@check("preprocess upscales, brightens and sharpens a small dark soft image")
def _t3():
    from utils.preprocess import assess_quality, enhance_image

    # A smooth, dark gradient: low resolution, low brightness, low Laplacian.
    grad = np.linspace(0, 45, 300, dtype="float32")
    small_dark = np.repeat(grad[None, :], 200, axis=0)
    rgb = np.stack([small_dark] * 3, axis=-1)
    path = _write_image(TMP / "dark_small.png", rgb)

    before = assess_quality(path)
    assert before["available"], "OpenCV must be available for enhancement tests"
    assert "low_resolution" in before["issues"], before
    assert "dark" in before["issues"], before

    out = enhance_image(path)
    assert out["enhanced"], out
    applied = " ".join(out["applied"])
    assert "upscaled" in applied, applied
    assert "brightened" in applied, applied
    assert out["quality_after"]["brightness"] > before["brightness"], out
    assert max(out["quality_after"]["width"], out["quality_after"]["height"]) >= 800
    assert Path(out["path"]).exists()
    os.unlink(out["path"])


@check("preprocess downscales an oversized image and leaves good images alone")
def _t4():
    from utils.preprocess import ENHANCE_MAX_DIM, enhance_image

    big = np.random.default_rng(7).integers(60, 200, size=(1200, 3000, 3))
    path = _write_image(TMP / "big.png", big)
    out = enhance_image(path)
    assert out["enhanced"], out
    assert "downscaled" in " ".join(out["applied"]), out["applied"]
    assert max(out["quality_after"]["width"], out["quality_after"]["height"]) <= ENHANCE_MAX_DIM
    os.unlink(out["path"])

    # Well-exposed, detailed, reasonably sized -> nothing to do.
    good = np.random.default_rng(3).integers(70, 190, size=(1000, 1400, 3))
    good_path = _write_image(TMP / "good.png", good)
    out2 = enhance_image(good_path)
    assert not out2["enhanced"], out2
    assert out2["path"] == good_path


# --------------------------------------------------------------------------
# 3. Synthetic-media detection
# --------------------------------------------------------------------------
@check("synthetic detection reads Stable Diffusion parameters from PNG chunks")
def _t5():
    from utils.synthetic import assess_synthetic

    img = Image.fromarray(
        np.random.default_rng(1).integers(0, 255, (512, 512, 3)).astype("uint8")
    )
    info = PngImagePlugin.PngInfo()
    info.add_text(
        "parameters",
        "a photo of a lake\nNegative prompt: blurry\n"
        "Steps: 30, Sampler: DPM++ 2M Karras, CFG scale: 7, Seed: 12345",
    )
    path = str(TMP / "sd.png")
    img.save(path, pnginfo=info)

    report = assess_synthetic(path, {}, "PNG", 512, 512)
    assert report["status"] == "synthetic", report
    assert (report["ai_generated_probability"] or 0) >= 0.9, report
    assert any("generation parameters" in s for s in report["signals"]), report


@check("synthetic detection flags the IPTC trainedAlgorithmicMedia declaration")
def _t6():
    from utils.synthetic import assess_synthetic

    path = str(TMP / "iptc.png")
    Image.new("RGB", (700, 500), (120, 120, 120)).save(path)
    metadata = {
        "camera": {"make": None, "model": None, "software": None},
        "raw": {"DigitalSourceType": "trainedAlgorithmicMedia"},
    }
    report = assess_synthetic(path, metadata, "PNG", 700, 500)
    assert report["status"] == "synthetic", report
    assert any("DigitalSourceType" in s for s in report["signals"]), report


@check("synthetic detection accepts a camera photo and only warns on bare canvases")
def _t7():
    from utils.synthetic import assess_synthetic

    photo = str(TMP / "photo.jpg")
    Image.fromarray(
        np.random.default_rng(5).integers(0, 255, (600, 800, 3)).astype("uint8")
    ).save(photo)
    camera_meta = {"camera": {"make": "NIKON", "model": "D810"}, "raw": {}}
    ok = assess_synthetic(photo, camera_meta, "JPEG", 800, 600)
    assert ok["status"] == "authentic", ok

    # 1024x1024 PNG with no camera identity: suspicious, but not asserted.
    bare = str(TMP / "bare.png")
    Image.new("RGB", (1024, 1024), (10, 20, 30)).save(bare)
    weak = assess_synthetic(bare, {"camera": {}, "raw": {}}, "PNG", 1024, 1024)
    assert weak["status"] == "likely_synthetic", weak
    assert (weak["ai_generated_probability"] or 0) < 0.8, "must stay below assertion"


# --------------------------------------------------------------------------
# 4. Geocoding helpers
# --------------------------------------------------------------------------
@check("geocode parses Photon features and identifies street-level hits")
def _t8():
    from utils.geocode import _parse_photon, is_street_level

    feature = {
        "geometry": {"coordinates": [-122.3321, 47.6062]},
        "properties": {
            "name": "Pike Place Market",
            "street": "Pike Street",
            "housenumber": "85",
            "city": "Seattle",
            "state": "Washington",
            "country": "United States",
            "countrycode": "us",
            "osm_key": "amenity",
            "osm_value": "marketplace",
            "type": "house",
            "postcode": "98101",
        },
    }
    hit = _parse_photon(feature)
    assert hit["latitude"] == 47.6062 and hit["longitude"] == -122.3321, hit
    assert hit["country_code"] == "US"
    assert "85 Pike Street" in hit["display_name"], hit["display_name"]
    assert hit["provider"] == "photon"
    assert is_street_level(hit)

    assert _parse_photon({"geometry": {}, "properties": {}}) is None
    assert not is_street_level(
        {"addresstype": "state", "category": "boundary", "osm_type": "state"}
    )


# --------------------------------------------------------------------------
# 5. Street-level refinement
# --------------------------------------------------------------------------
@check("street refinement is skipped for scenes with nothing to read")
def _t9():
    from utils.streetlevel import refine_street_level, worth_refining

    wilderness = {"scene": {"forest": True, "mountains": True}, "ocr_text": []}
    assert not worth_refining(wilderness)
    out = refine_street_level(
        47.0, -120.0, wilderness, image_path=str(TMP / "photo.jpg")
    )
    assert out["attempted"] is False, out
    assert "no street-level identifiers" in out["note"].lower(), out

    assert worth_refining({"scene": {"urban": True}})
    assert worth_refining({"scene": {}, "signage": ["Bäckerei"]})


@check("street refinement builds prioritised queries and de-duplicates them")
def _t10():
    from utils.streetlevel import _clue_queries

    clues = {
        "street_names": ["Pike Street", "1st Avenue"],
        "house_numbers": ["85"],
        "business_names": ["Pike Place Market"],
        "transit_stops": ["Westlake Station"],
        "intersection": "Pike Street & 1st Avenue",
        "refined_place": "Pike Place Market, Pike Street",
    }
    queries = _clue_queries(clues, "Seattle")
    methods = [m for m, _ in queries]
    assert methods[0] == "address", methods
    assert "85 Pike Street" in queries[0][1], queries[0]
    assert methods == sorted(
        methods, key=lambda m: ["address", "business", "intersection", "transit",
                                "street", "place"].index(m)
    ), methods
    assert len(queries) == len({q for _, q in queries}), "queries must be unique"


@check("street refinement only moves the point when the map confirms a clue")
def _t11():
    import utils.streetlevel as S

    image = str(TMP / "photo.jpg")
    vision = {"scene": {"urban": True}, "scene_type": "urban", "ocr_text": ["Pike St"]}

    clues = {
        "available": True,
        "note": "ok",
        "street_names": ["Pike Street"],
        "house_numbers": ["85"],
        "business_names": [],
        "transit_stops": [],
        "postal_codes": [],
        "phone_numbers": [],
        "intersection": None,
        "architectural_details": ["brick facade"],
        "street_furniture": [],
        "refined_place": "85 Pike Street",
        "refined_latitude": None,
        "refined_longitude": None,
        "precision_estimate_m": 80,
        "reasoning": "Street plate legible.",
        "confidence": 0.7,
    }
    original_vision_fn = S.analyze_street_level
    original_geo = S.geocode_street
    try:
        S.analyze_street_level = lambda *a, **k: dict(clues)

        # (a) The map confirms the address -> the point moves and is precise.
        S.geocode_street = lambda q, lat, lon, radius, limit=5: {
            "latitude": 47.6088,
            "longitude": -122.3403,
            "display_name": "85 Pike Street, Seattle, Washington",
            "name": "85 Pike Street",
            "housenumber": "85",
            "street": "Pike Street",
            "addresstype": "house",
            "category": "place",
            "distance_km": 1.1,
            "provider": "photon",
        }
        got = S.refine_street_level(
            47.6062, -122.3321, vision, image_path=image, city_hint="Seattle"
        )
        assert got["refined"], got
        assert got["latitude"] == 47.6088, got
        assert got["method"] == "address_geocode", got
        assert got["precision_m"] <= 120, got
        assert got["distance_moved_km"] is not None

        # (b) The same clue with no map match -> the region must stand.
        S.geocode_street = lambda *a, **k: None
        blocked = S.refine_street_level(
            47.6062, -122.3321, vision, image_path=image, city_hint="Seattle"
        )
        assert not blocked["refined"], blocked
        assert "none could be matched" in blocked["note"], blocked
        assert blocked["latitude"] is None, "a hallucinated name must not move the point"

        # (c) A model coordinate far outside the radius must be rejected.
        far = dict(clues, refined_latitude=10.0, refined_longitude=10.0)
        S.analyze_street_level = lambda *a, **k: far
        rejected = S.refine_street_level(
            47.6062, -122.3321, vision, image_path=image, radius_km=25
        )
        assert not rejected["refined"], rejected
    finally:
        S.analyze_street_level = original_vision_fn
        S.geocode_street = original_geo


# --------------------------------------------------------------------------
# 6. Vehicle identification
# --------------------------------------------------------------------------
@check("vehicle report normalises models and folds plate regions into implications")
def _t12():
    import utils.vehicle as V

    payload = {
        "vehicles_present": True,
        "vehicles": [
            {
                "make": "Toyota",
                "model": "Hilux",
                "generation": "AN120",
                "year_range": "2016-2020",
                "body_style": "pickup",
                "color": "white",
                "steering_side": "right",
                "market_hint": "Australia/NZ",
                "distinguishing_features": ["snorkel", "bull bar"],
                "confidence": 0.72,
            },
            {"make": None, "model": None, "body_style": None, "confidence": 0.9},
        ],
        "plates": [
            {
                "text": "1ABC 234",
                "format_description": "white plate, black text, state slogan",
                "region_implied": "New South Wales, Australia",
                "confidence": 0.6,
            }
        ],
        "regional_indicators": ["right-hand drive"],
        "implied_regions": ["Australia"],
        "confidence": 0.68,
    }
    original = V.call_vision_json
    original_detect = V._roboflow_detect
    try:
        V.call_vision_json = lambda *a, **k: (payload, "ok")
        V._roboflow_detect = lambda p: []
        report = V.identify_vehicles(str(TMP / "photo.jpg"))
        assert report["available"], report
        assert len(report["vehicles"]) == 1, "entries with no identity are dropped"
        assert report["vehicles"][0]["model"] == "Hilux"
        assert "Australia" in report["implied_regions"]
        assert any(
            "New South Wales" in r for r in report["implied_regions"]
        ), report["implied_regions"]
        lines = V.vehicle_evidence(report)
        assert any("Toyota Hilux" in line for line in lines), lines
        assert any("License plate" in line for line in lines), lines
    finally:
        V.call_vision_json = original
        V._roboflow_detect = original_detect


@check("vehicle gating avoids a wasted call on scenes with no vehicles")
def _t13():
    from utils.vehicle import should_identify

    assert not should_identify({"scene": {"forest": True}, "road_side": "unknown"})
    assert should_identify({"scene": {}, "road_side": "left"})
    assert should_identify({"scene": {}, "vehicles_plates": ["ABC-123"]})
    assert should_identify({"scene": {"urban": True}})
    assert not should_identify(None)


# --------------------------------------------------------------------------
# 7. Fusion signals in the OSINT engine
# --------------------------------------------------------------------------
@check("vehicle conflict lowers confidence, agreement raises it, GPS is immune")
def _t14():
    from utils.osint import apply_vehicle_signal

    vehicle = {
        "available": True,
        "vehicles_present": True,
        "vehicles": [],
        "plates": [],
        "implied_regions": ["France"],
    }
    base_signals = {
        "ai_conf": 0.6,
        "clue_bonus": 0.1,
        "verify_status": "verified",
        "ambiguity_gap": 0.3,
    }

    conflict = {
        "country": "Canada",
        "region": "Ontario",
        "location_name": "Toronto",
        "address": "Toronto, Ontario, Canada",
        "confidence": 0.8,
        "evidence": [],
        "source": "vision_geocoded",
        "fusion_signals": dict(base_signals),
    }
    apply_vehicle_signal(conflict, vehicle)
    assert conflict["vehicle_check"]["status"] == "conflict", conflict["vehicle_check"]
    assert conflict["confidence"] < 0.8, conflict["confidence"]
    assert conflict["warning"], "a contradiction must be surfaced"

    agree = {
        "country": "France",
        "region": "Occitanie",
        "location_name": "Toulouse",
        "address": "Toulouse, France",
        "confidence": 0.5,
        "evidence": [],
        "source": "vision_geocoded",
        "fusion_signals": dict(base_signals),
    }
    apply_vehicle_signal(agree, vehicle)
    assert agree["vehicle_check"]["status"] == "match", agree["vehicle_check"]
    assert agree["confidence"] > conflict["confidence"], (
        agree["confidence"], conflict["confidence"]
    )

    gps = {
        "country": "Canada",
        "location_name": "Toronto",
        "region": None,
        "address": "Toronto",
        "confidence": 0.99,
        "evidence": [],
        "source": "exif_gps",
    }
    apply_vehicle_signal(gps, vehicle)
    assert gps["confidence"] == 0.99, "EXIF GPS must never be penalised"


@check("a synthetic image caps confidence and warns")
def _t15():
    from utils.osint import _warn_if_synthetic

    result = {"confidence": 0.9, "evidence": [], "warning": "some other warning"}
    forensics = {
        "authenticity": "synthetic",
        "synthetic": {
            "status": "synthetic",
            "note": "Declared AI-generated (Midjourney).",
        },
    }
    _warn_if_synthetic(result, forensics)
    assert result["confidence"] <= 0.15, result["confidence"]
    assert "Midjourney" in result["warning"], result["warning"]
    assert result["authenticity"] == "synthetic"

    clean = {"confidence": 0.8, "evidence": []}
    _warn_if_synthetic(clean, {"authenticity": "authentic", "synthetic": {"status": "authentic"}})
    assert clean["confidence"] == 0.8
    assert "warning" not in clean


@check("synthetic and street signals move fused confidence in the right direction")
def _t16():
    from utils.osint import _fuse_confidence

    base = {"ai_conf": 0.7, "verify_status": "verified", "clue_bonus": 0.1}
    plain = _fuse_confidence(dict(base))
    with_street = _fuse_confidence(dict(base, street_status="geocoded"))
    with_model_street = _fuse_confidence(dict(base, street_status="model"))
    synthetic = _fuse_confidence(dict(base, synthetic_status="synthetic"))

    assert with_street > plain, (with_street, plain)
    assert plain <= with_model_street < with_street, (plain, with_model_street)
    assert synthetic < plain * 0.6, (synthetic, plain)
    assert 0.0 <= synthetic <= 1.0


@check("script/plate region matching tolerates abbreviations but not near-misses")
def _t16b():
    from utils.osint import _implied_region_status

    uk = {"country": "United Kingdom", "region": "England", "location_name": "London"}
    # Abbreviations and endonyms must read as agreement, not contradiction.
    assert _implied_region_status(["UK"], uk) == "match"
    assert _implied_region_status(["Great Britain"], uk) == "match"
    assert _implied_region_status(["United Kingdom"], uk) == "match"
    assert _implied_region_status(["France"], uk) == "conflict"

    usa = {"country": "United States", "region": "Washington", "location_name": "Seattle"}
    assert _implied_region_status(["USA"], usa) == "match"
    assert _implied_region_status(["US"], usa) == "match"
    # "us" is a substring of "Russia"; a short token must not match inside a word.
    russia = {"country": "Russia", "region": "Moscow", "location_name": "Moscow"}
    assert _implied_region_status(["US"], russia) == "conflict"

    # Nothing to compare must stay silent rather than inventing a conflict.
    assert _implied_region_status([], uk) is None
    assert _implied_region_status(["UK"], {}) is None


@check("visible-text country evidence is fused and contradictions warn")
def _t16c():
    from utils.osint import _apply_text_signal, _fuse_confidence

    vision = {
        "text_analysis": {"primary_script": "Devanagari", "implied_countries": ["India"]}
    }
    agree = {"country": "India", "region": "Maharashtra", "evidence": []}
    signals: dict = {}
    _apply_text_signal(agree, vision, signals)
    assert signals["text_status"] == "match", signals
    assert any("Devanagari" in line for line in agree["evidence"]), agree["evidence"]
    assert not agree.get("warning")

    clash = {"country": "Brazil", "region": "Bahia", "evidence": []}
    clash_signals: dict = {}
    _apply_text_signal(clash, vision, clash_signals)
    assert clash_signals["text_status"] == "conflict", clash_signals
    assert "India" in (clash.get("warning") or ""), clash

    base = {"ai_conf": 0.7, "verify_status": "skipped", "clue_bonus": 0.05}
    plain = _fuse_confidence(dict(base))
    assert _fuse_confidence(dict(base, text_status="match")) > plain
    assert _fuse_confidence(dict(base, text_status="conflict")) < plain


@check("the weight tuner only reports knobs the labelled cases can identify")
def _t16d():
    sys.path.insert(0, str(ROOT / "eval"))
    try:
        import tune_weights as T
    finally:
        sys.path.pop(0)

    import utils.osint as O

    # A stand-in eval: only the AI weight changes the "confidence" it reports, so
    # every other knob must be screened out as unidentifiable.
    def fake_run():
        return {
            "cases": [{"id": "a", "confidence": round(O._AI_WEIGHT, 3)}],
            "mean_brier": abs(O._AI_WEIGHT - 1.0),
            "n": 1,
        }

    before = O._AI_WEIGHT
    try:
        knobs = T._available_knobs(O, fake_run)
    finally:
        O._AI_WEIGHT = before
    assert set(knobs) == {"FUSION_AI"}, sorted(knobs)
    # Probing must leave the module exactly as it was found.
    assert O._AI_WEIGHT == before, O._AI_WEIGHT

    assert "No weight change improved" in T._env_block({})
    block = T._env_block({"FUSION_AI": 1.5})
    assert "FUSION_AI=1.5" in block, block


@check("hypothesis planning de-duplicates coordinates and preserves ranking")
def _t17():
    import utils.osint as O

    original = O.forward_geocode_candidates
    try:
        # Same real-world point returned for two names: must be tried once.
        O.forward_geocode_candidates = lambda q, limit=3: [
            {
                "latitude": 40.0,
                "longitude": -100.0,
                "name": q,
                "country": "United States",
                "region": None,
                "display_name": q,
            }
        ]
        candidates = [
            {"name": "Alpha", "confidence": 0.8, "latitude": 10.0, "longitude": 10.0},
            {"name": "Beta", "confidence": 0.5},
            {"name": "Gamma", "confidence": 0.3},
        ]
        plan, name_only = O._plan_hypotheses(candidates, 0.05)
        coords = [(h["latitude"], h["longitude"]) for _, h in plan]
        assert len(coords) == len(set(coords)), coords
        assert coords[0] == (10.0, 10.0), "the top candidate is tried first"
        assert name_only["name"] == "Alpha", name_only
        # Confidence must decrease down the plan (best-first ordering).
        bases = [h["base"] for _, h in plan]
        assert bases == sorted(bases, reverse=True), bases
    finally:
        O.forward_geocode_candidates = original


@check("street refinement rewrites the location and relabels the source")
def _t18():
    import utils.osint as O

    result = {
        "latitude": 47.60,
        "longitude": -122.33,
        "location_name": "Seattle",
        "region": "Washington",
        "country": "United States",
        "address": "Seattle, Washington",
        "source": "vision_geocoded",
        "evidence": [],
    }
    original = O.refine_street_level
    try:
        O.refine_street_level = lambda *a, **k: {
            "attempted": True,
            "refined": True,
            "latitude": 47.6088,
            "longitude": -122.3403,
            "address": "85 Pike Street, Seattle, Washington",
            "precision_m": 60,
            "method": "address_geocode",
            "matched_query": "85 Pike Street, Seattle",
            "distance_moved_km": 1.1,
            "note": "Refined to 85 Pike Street.",
            "evidence": ["Street name(s) read: Pike Street."],
            "candidates": [],
            "street_clues": {},
        }
        O._apply_street_refinement(result, {"scene": {"urban": True}}, "img.jpg", "verified")
        assert result["latitude"] == 47.6088, result
        assert result["source"] == "street_level", result
        assert result["precision_m"] == 60, result
        assert result["location_name"] == "85 Pike Street", result
        assert any("Pike Street" in e for e in result["evidence"]), result["evidence"]

        # A map-contradicted region must not be refined into a precise wrong spot.
        blocked = dict(result, source="vision_geocoded", evidence=[])
        O._apply_street_refinement(blocked, {"scene": {"urban": True}}, "img.jpg", "mismatch")
        assert blocked["source"] == "vision_geocoded", blocked
    finally:
        O.refine_street_level = original


@check("verified landmarks snap the pin inside the 2 km defined radius")
def _t18b():
    import utils.osint as O

    original = O.reverse_geocode
    O.reverse_geocode = lambda lat, lon: {
        "display_name": "Eiffel Tower, Paris, France",
        "country": "France",
        "region": "Île-de-France",
        "name": "Eiffel Tower",
    }
    try:
        result = {
            "latitude": 48.8566,
            "longitude": 2.3522,
            "location_name": "Paris",
            "country": "France",
            "region": "Île-de-France",
            "source": "vision_geocoded",
            "evidence": [],
            "verification": {
                "named": {
                    "landmark_status": "matched",
                    "matched_landmarks": [
                        {
                            "name": "Eiffel Tower",
                            "distance_km": 4.3,
                            "latitude": 48.8584,
                            "longitude": 2.2945,
                            "osm_type": "attraction",
                        }
                    ],
                }
            },
        }
        O._snap_to_verified_feature(result)
        assert abs(result["latitude"] - 48.8584) < 1e-6, result
        assert result["location_name"] == "Eiffel Tower", result
        assert result["snapped_to"]["kind"] == "landmark", result
        O._attach_defined_radius(result)
        assert result["defined_radius_km"] == 2, result
        assert result["defined_radius_m"] == 2000, result
        assert result["meets_defined_radius"] is True, result
        assert result["precision_m"] <= 2000, result

        gps = {
            "latitude": 47.606,
            "longitude": -122.332,
            "source": "exif_gps",
            "evidence": [],
            "verification": {
                "named": {
                    "matched_landmarks": [
                        {"name": "Space Needle", "latitude": 47.62, "longitude": -122.35}
                    ]
                }
            },
        }
        O._snap_to_verified_feature(gps)
        assert gps["latitude"] == 47.606, "EXIF GPS must never be snapped"
        O._attach_defined_radius(gps)
        assert gps["meets_defined_radius"] is True, gps

        peak = {
            "latitude": 46.0207,
            "longitude": 7.7491,
            "location_name": "Zermatt",
            "source": "vision_geocoded",
            "evidence": [],
            "verification": {
                "named": {
                    "landmark_status": "matched",
                    "matched_landmarks": [
                        {
                            "name": "Matterhorn",
                            "distance_km": 8.6,
                            "latitude": 45.9763,
                            "longitude": 7.6586,
                            "osm_type": "peak",
                        }
                    ],
                }
            },
        }
        O._snap_to_verified_feature(peak)
        assert peak["latitude"] == 46.0207, "a distant peak is the subject, not the camera"

        coarse = {
            "latitude": 44.0,
            "longitude": -120.5,
            "source": "vision_geocoded",
            "evidence": [],
            "verification": {"named": {}},
        }
        O._snap_to_verified_feature(coarse)
        O._attach_defined_radius(coarse)
        assert coarse["meets_defined_radius"] is False, coarse
        assert any("defined radius" in e.lower() for e in coarse["evidence"]), coarse
    finally:
        O.reverse_geocode = original


# --------------------------------------------------------------------------
# 8. Multi-image clustering
# --------------------------------------------------------------------------
@check("clustering finds a consensus, combines confidence and isolates outliers")
def _t19():
    from utils.clustering import cluster_multi_image_candidates

    items = [
        {"filename": "a.jpg", "latitude": 47.60, "longitude": -122.33, "confidence": 0.6,
         "location_name": "Seattle", "country": "United States"},
        {"filename": "b.jpg", "latitude": 47.62, "longitude": -122.35, "confidence": 0.5,
         "location_name": "Seattle", "country": "United States"},
        {"filename": "c.jpg", "latitude": 47.61, "longitude": -122.30, "confidence": 0.55,
         "location_name": "Seattle", "country": "United States"},
        {"filename": "d.jpg", "latitude": 34.05, "longitude": -118.24, "confidence": 0.4,
         "location_name": "Los Angeles", "country": "United States"},
        {"filename": "e.jpg", "latitude": None, "longitude": None, "confidence": 0.0,
         "location_name": None},
    ]
    out = cluster_multi_image_candidates(items, radius_km=10)

    assert out["images_located"] == 4, out
    assert out["images_unlocated"] == ["e.jpg"], out
    assert len(out["clusters"]) == 2, out["clusters"]

    top = out["clusters"][0]
    assert top["images_agreeing"] == 3, top
    # Noisy-OR of 0.6/0.55/0.5: higher than any single guess, still below 1.
    assert 0.9 < top["combined_confidence"] < 1.0, top["combined_confidence"]
    assert top["spread_km"] < 10, top

    assert out["consensus"] is not None
    assert out["consensus"]["images_agreeing"] == 3
    assert "3 of 4" in out["consensus"]["note"], out["consensus"]["note"]

    # Two disagreeing images cannot form a consensus.
    split = cluster_multi_image_candidates(items[:1] + items[3:4], radius_km=10)
    assert split["consensus"] is None, split["consensus"]


@check("clustering heatmap includes alternatives at reduced weight")
def _t20():
    from utils.clustering import cluster_multi_image_candidates

    items = [
        {
            "filename": "a.jpg",
            "latitude": 10.0,
            "longitude": 20.0,
            "confidence": 0.8,
            "alternatives": [
                {"name": "alt", "latitude": 11.0, "longitude": 21.0, "confidence": 0.5}
            ],
        }
    ]
    out = cluster_multi_image_candidates(items)
    assert [10.0, 20.0, 0.8] in out["heatmap"], out["heatmap"]
    assert [11.0, 21.0, 0.2] in out["heatmap"], out["heatmap"]


# --------------------------------------------------------------------------
# 9. Case persistence
# --------------------------------------------------------------------------
@check("case store creates, appends to, reads and deletes a case")
def _t21():
    from utils import casestore

    db = Path(casestore.CASE_DB_PATH)
    if db.exists():
        db.unlink()

    casestore.init_db()
    case_id = casestore.create_case("Test case")
    assert casestore.case_exists(case_id)
    assert casestore.item_count(case_id) == 0

    casestore.add_item(
        case_id,
        "one.jpg",
        {
            "filename": "one.jpg",
            "latitude": 47.6,
            "longitude": -122.3,
            "confidence": 0.7,
            "location_name": "Seattle",
            "country": "United States",
            "region": "Washington",
            "alternatives": [{"name": "Tacoma", "latitude": 47.2, "longitude": -122.4,
                              "confidence": 0.3}],
        },
    )
    casestore.add_item(
        case_id, "two.jpg", {"filename": "two.jpg", "confidence": 0.0}
    )

    assert casestore.item_count(case_id) == 2
    items = casestore.list_items(case_id)
    assert [i["filename"] for i in items] == ["one.jpg", "two.jpg"], items
    assert items[0]["latitude"] == 47.6
    assert items[0]["alternatives"][0]["name"] == "Tacoma"
    assert items[1]["latitude"] is None

    case = casestore.get_case(case_id)
    assert case["title"] == "Test case"
    assert len(case["items"]) == 2
    assert any(c["case_id"] == case_id for c in casestore.list_cases())

    assert casestore.delete_case(case_id)
    assert not casestore.case_exists(case_id)
    assert casestore.get_case(case_id) is None
    assert not casestore.delete_case(case_id), "deleting twice must report failure"


# --------------------------------------------------------------------------
# 10. End-to-end wiring through the API layer
# --------------------------------------------------------------------------
@check("response builder exposes the new fields and stays App-Inventor flat")
def _t22():
    from utils.response import build_response, error_response

    payload = build_response(
        filename="x.jpg",
        metadata={"has_exif": False, "gps": None},
        vision={"available": True},
        location={
            "location_name": "85 Pike Street",
            "country": "United States",
            "latitude": 47.6088,
            "longitude": -122.3403,
            "confidence": 0.81,
            "source": "street_level",
            "precision_m": 60,
            "street_level": {"refined": True, "method": "address_geocode"},
            "vehicle": {"available": True},
            "vehicle_check": {"status": "match"},
            "forensics": {"authenticity": "authentic"},
            "fusion_signals": {"verify_status": "verified"},
            "evidence": [],
        },
        image_quality={"enhanced": True, "applied": ["upscaled 2.0x"]},
    )
    assert payload["source"] == "Street-level match on the map", payload["source"]
    assert payload["coordinates"] == "Latitude 47.6088 Longitude -122.3403"
    assert payload["authenticity"] == "authentic"
    assert payload["street_level"]["refined"] is True
    assert payload["vehicle_check"]["status"] == "match"
    assert payload["image_quality"]["enhanced"] is True
    assert payload["details"]["fusion_signals"]["verify_status"] == "verified"
    assert payload["defined_radius_km"] == 2, payload["defined_radius_km"]
    assert payload["defined_radius"] == "2 km", payload["defined_radius"]
    assert "Defined radius" in payload["summary"], payload["summary"]

    err = error_response("bad input")
    for key in (
        "street_level",
        "authenticity",
        "vehicle",
        "vehicle_check",
        "image_quality",
        "defined_radius_km",
        "defined_radius",
        "meets_defined_radius",
    ):
        assert key in err, f"error envelope must define {key}"
    assert err["defined_radius_km"] == 2


@check("the API exposes every new route and reports capabilities")
def _t23():
    import main

    paths = {getattr(r, "path", None) for r in main.app.routes}
    for expected in ("/analyze", "/analyze-video", "/case", "/case/{case_id}", "/cases"):
        assert expected in paths, f"missing route {expected}"

    health = main.health()
    for key in (
        "vision_provider",
        "enhancement_enabled",
        "cases_enabled",
        "video_enabled",
        "web_ui",
        "street_refine_enabled",
        "vehicle_id_enabled",
        "synthetic_detection_enabled",
        "scene_routing_enabled",
        "photon_enabled",
        "defined_radius_km",
    ):
        assert key in health, f"health must report {key}"
    assert health["defined_radius_km"] == 2, health["defined_radius_km"]
    assert health["enhancement_enabled"] is True, "OpenCV should be installed"
    # App Inventor cannot read nested objects out of a response.
    assert all(
        not isinstance(v, (dict, list)) for v in health.values()
    ), "health must stay flat"


@check("the single-image pipeline rejects non-images and oversized uploads")
def _t24():
    from main import _analyze_image_bytes

    status, payload = _analyze_image_bytes(b"this is not an image", "x.txt")
    assert status == 400, status
    assert payload["success"] is False
    assert "not a valid image" in payload["error"], payload

    status, payload = _analyze_image_bytes(None, None)
    assert status == 400 and "No image found" in payload["error"], payload

    status, payload = _analyze_image_bytes(b"x" * (26 * 1024 * 1024), "big.jpg")
    assert status == 413, status


@check("the single-image pipeline runs end to end with vision stubbed out")
def _t25():
    import main
    import utils.osint as O

    photo = str(TMP / "pipeline.jpg")
    Image.fromarray(
        np.random.default_rng(11).integers(40, 220, (900, 1200, 3)).astype("uint8")
    ).save(photo)
    data = Path(photo).read_bytes()

    fake_vision = {
        "available": True,
        "note": "ok",
        "ocr_text": [],
        "languages": [],
        "landmarks": [],
        "flags": [],
        "signage": [],
        "vehicles_plates": [],
        "architecture": None,
        "architecture_style": None,
        "architecture_regions": [],
        "vegetation": "conifers",
        "climate": "temperate",
        "terrain": "hills",
        "scene": {"forest": True},
        "scene_type": "landscape",
        "text_analysis": {"primary_script": None, "regional_spelling": [],
                          "implied_countries": []},
        "sun": {"sun_visible": False, "shadows_visible": False,
                "shadow_direction": None, "shadow_length": None,
                "approx_solar_elevation": None},
        "road_side": "unknown",
        "time_of_day": None,
        "hemisphere_hint": "Northern",
        "environment": "rural",
        "candidates": [
            {"name": "Cascade Range", "region": "Washington",
             "country": "United States", "latitude": 47.5, "longitude": -121.0,
             "confidence": 0.6, "why": "conifers"}
        ],
        "best_guess_location": {"name": "Cascade Range", "country": "United States",
                                "latitude": 47.5, "longitude": -121.0},
        "reasoning": "Coniferous forest.",
        "confidence": 0.6,
    }

    saved = (
        main.analyze_image,
        O.verify_location,
        O.reverse_geocode,
        O.enrich_location,
        O.refine_street_level,
    )
    try:
        main.analyze_image = lambda path, scene_type=None: dict(fake_vision)
        O.verify_location = lambda lat, lon, expected, radius=None: {
            "status": "verified", "note": "ok (stub)", "expected": expected,
            "confirmed": list(expected), "missing": [], "context": [],
            "nearest_m": {}, "anchors": {}, "match_score": 1.0,
            "radius_m": radius or 3000,
        }
        O.reverse_geocode = lambda lat, lon: {
            "display_name": "King County, Washington, United States",
            "name": "King County", "country": "United States",
            "region": "Washington",
        }
        O.enrich_location = lambda lat, lon: {"nearby_places": []}
        O.refine_street_level = lambda *a, **k: {
            "attempted": False, "refined": False, "note": "skipped",
            "evidence": [], "candidates": [], "street_clues": {},
        }

        status, payload = main._analyze_image_bytes(data, "pipeline.jpg")
        assert status == 200, payload
        assert payload["success"] is True
        assert payload["location_name"] == "Cascade Range", payload["location_name"]
        assert payload["verified"] == "verified", payload["verified"]
        assert 0.0 < payload["confidence"] <= 1.0, payload["confidence"]
        assert payload["authenticity"] in ("authentic", "unknown"), payload["authenticity"]
        assert payload["reasoning_trace"], "a trace must always be produced"
        assert payload["image_quality"] is not None
        assert "path" not in (payload["image_quality"] or {}), "no server paths leak"
    finally:
        (
            main.analyze_image,
            O.verify_location,
            O.reverse_geocode,
            O.enrich_location,
            O.refine_street_level,
        ) = saved


# --------------------------------------------------------------------------
def main_() -> int:
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = [(n, m) for n, ok, m in _results if not ok]

    for name, ok, msg in _results:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
        if not ok:
            for line in msg.strip().splitlines():
                print(f"         {line}")

    print()
    print(f"{passed}/{len(_results)} checks passed")
    if failed:
        print(f"{len(failed)} FAILED")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main_())
