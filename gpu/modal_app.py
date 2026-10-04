"""Serverless geolocation models on Modal (scale-to-zero, per-second billing).

Open models that give location opinions next to GPT-4o on Render. The default
set is the trio that best complemented GPT-4o on the remote-terrain benchmark
in eval/model_bench (see score.py):

  geoclip   geoclip (pip), CLIP ViT-L/14          image-to-GPS retrieval, runs on CPU
  gaea      ucf-crcv/GAEA-7B                      Qwen2.5-VL-7B LoRA, merged into full weights here
  globe     globe-project/GLOBE-Qwen2.5VL-7B      Qwen2.5-VL-7B RL fine-tune, country/city answers

Also runnable for comparison:
  geoagent  ghost233lism/GeoAgent                 Qwen2.5-VL-7B fine-tune, chain-of-thought JSON
  qwen      Qwen/Qwen2.5-VL-7B-Instruct           general model
  gemma     RedHatAI/gemma-3-12b-it-FP8-dynamic   Google Gemma 3 12B, FP8 so it fits an L4
  gemma27   RedHatAI/gemma-3-27b-it-FP8-dynamic   Google Gemma 3 27B, FP8 on an L40S
  pixtral   RedHatAI/pixtral-12b-FP8-dynamic      Mistral Pixtral 12B, FP8

Each language model gets its own L4 (24 GB) container and GeoCLIP a CPU one.
A container starts on the first request, stays warm for GEO_SCALEDOWN_S
seconds after the last one, then shuts down, so idle time costs nothing.
GeoAgent and GAEA weights are CC BY-NC 4.0 and Gemma derivatives follow
Google's Gemma terms (non-commercial / restricted use).

One-time setup (from the repo root, with `pip install modal`):
  modal setup                                      # log in via the browser
  modal run gpu/modal_app.py::download_models      # weights into the geo-models volume
  modal run gpu/modal_app.py::main --image photo.jpg   # smoke test the default models
  modal run gpu/modal_app.py::benchmark --models ...   # score models on eval/model_bench
  modal deploy gpu/modal_app.py                    # publish the HTTPS endpoint

The deployed `locate` endpoint requires Modal proxy-auth headers (Modal-Key /
Modal-Secret; create a token under Settings > Proxy Auth Tokens). A cold start
can exceed Modal's 150 s web timeout, in which case Modal answers with a 303
redirect to a result URL; clients must follow redirects with a long timeout.
"""

import os

import modal

APP_NAME = "geo-ensemble"
MODELS_DIR = "/models"
GPU = os.getenv("GEO_GPU", "L4")
SCALEDOWN_S = int(os.getenv("GEO_SCALEDOWN_S", "300"))

QWEN_REPO = "Qwen/Qwen2.5-VL-7B-Instruct"
GEOAGENT_REPO = "ghost233lism/GeoAgent"
GAEA_REPO = "ucf-crcv/GAEA-7B"
GEMMA_REPO = "RedHatAI/gemma-3-12b-it-FP8-dynamic"
GEMMA27_REPO = "RedHatAI/gemma-3-27b-it-FP8-dynamic"
PIXTRAL_REPO = "RedHatAI/pixtral-12b-FP8-dynamic"
GLOBE_REPO = "globe-project/GLOBE-Qwen2.5VL-7B"
CLIP_REPO = "openai/clip-vit-large-patch14"

DOWNLOADS = {
    "qwen": QWEN_REPO,
    "geoagent": GEOAGENT_REPO,
    "gemma": GEMMA_REPO,
    "gemma27": GEMMA27_REPO,
    "pixtral": PIXTRAL_REPO,
    "globe": GLOBE_REPO,
}

HF_HOME = f"{MODELS_DIR}/hf"

# Qwen2.5-VL turns every 28x28 patch into a token; capping pixels keeps the
# image around 1.3k tokens so weights + KV cache fit on a 24 GB L4. Gemma 3
# always encodes an image as 256 tokens, so it needs no cap.
MAX_PIXELS = 1280 * 28 * 28
MAX_IMAGE_SIDE = 1536
_QWEN_ENGINE = {"mm_processor_kwargs": {"max_pixels": MAX_PIXELS}}

MODEL_PATHS = {
    "geoagent": f"{MODELS_DIR}/geoagent",
    "gemma": f"{MODELS_DIR}/gemma-3-12b-it-fp8",
    "gemma27": f"{MODELS_DIR}/gemma-3-27b-it-fp8",
    "pixtral": f"{MODELS_DIR}/pixtral-12b-fp8",
    "globe": f"{MODELS_DIR}/globe-qwen2.5vl-7b",
    "gaea": f"{MODELS_DIR}/gaea-7b-merged",
    "qwen": f"{MODELS_DIR}/qwen2.5-vl-7b-instruct",
}
ENGINE_KWARGS = {
    "geoagent": _QWEN_ENGINE,
    "gemma": {},
    "gemma27": {},
    "pixtral": {},
    "globe": _QWEN_ENGINE,
    "gaea": _QWEN_ENGINE,
    "qwen": _QWEN_ENGINE,
}
# Models too big for an L4 even at FP8.
MODEL_GPU = {"gemma27": "L40S"}
# Pixtral's chat template cannot mix a system turn with image content.
SYSTEM_IN_USER = {"pixtral"}
ALL_MODELS = (*MODEL_PATHS, "geoclip")
DEFAULT_MODELS = ("geoclip", "gaea", "globe")
GEOCLIP_TOP_K = 5

GEOAGENT_SYSTEM = """You are an expert with rich experience in the field of geolocation, skilled at accurately locating the geographic location of images through various clues in the images, such as traffic signs, architectural styles, natural landscapes, etc. At the same time, you are also a mentor in building the chain of thought, able to organize complex ideas into clear and standardized patterns. You possess knowledge in multiple disciplines such as geography, cartography, transportation, and architecture, and are able to identify the characteristics of different countries, regions, and locations. At the same time, you have the ability to analyze logic and construct a chain of thought. Task: Output the thought chain and final answer based on the image input by the user. The thought chain includes:
        Country Identification/Regional Guess/Precise Localization.
        Possible clues include: National clues: (Example: traffic sign shape/color, language and text, driving direction, architectural style, vegetation and climate characteristics, etc.)
        Regional clues: (logo/enterprise, topography, vegetation type, regional traffic signs, dialect/spelling, license plate style, area code/postal code, infrastructure features, etc.)
        Accurate positioning: (road sign text, street name, house number, landmark building, river and lake water system, place attributes such as park/city/commercial district, shop name and storefront, etc.)
        Do not output objects that do not exist in the image.
        Output strictly in JSON format:
        {
        "ChainOfThought": {
            "CountryIdentification": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            },
            "RegionalGuess": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            },
            "PreciseLocalization": {
            "Clues": [],
            "Reasoning": "",
            "Conclusion": "",
            "Uncertainty": ""
            }
        },
        "FinalAnswer": "Country; Region; Specific Location"
        }"""

GAEA_PREAMBLE = (
    "You are an expert in geography and tourism. You possess extensive knowledge of "
    "geography, terrain, landscapes, flora, fauna, infrastructure, and other natural or "
    "man-made features that help determine a location from images or descriptions. "
    "Additionally, you are well-versed in tourism-related information, including amenities "
    "such as hotels, restaurants, attractions, and services available in various locations."
)

JSON_SYSTEM = (
    "You are an expert image geolocator. Use only clues visible in the image: text and "
    "signs, languages, landmarks, architecture, road markings, driving side, vehicles, "
    "vegetation, terrain, water bodies, climate. Always commit to your single best guess, "
    "even when unsure; express doubt through confidence, never by leaving fields blank. "
    "Reply with one JSON object and nothing else, with these keys:\n"
    "country, region, city (nearest town), specific_location (the most precise place you "
    "can name: landmark, street, lake, neighbourhood), latitude, longitude (decimal "
    "degrees of your guess), confidence (0-1), clues (list of the visible evidence).\n"
    'Example: {"country": "Italy", "region": "Lombardy", "city": "Bellagio", '
    '"specific_location": "Lake Como, near Bellagio", "latitude": 45.98, '
    '"longitude": 9.26, "confidence": 0.35, "clues": ["large alpine lake", '
    '"steep forested slopes", "Italian-style villas"]}'
)

GLOBE_PROMPT = (
    "You are a geolocation expert. You are participating in a geolocation challenge. "
    "Based on the provided image:\n1. Carefully analyze the image for clues about its "
    "location (architecture, signage, vegetation, terrain, etc.)\n2. Think step-by-step "
    "about what country, and city this is likely to be in and why\n\nYour final answer "
    "include these two lines somewhere in your response:\ncountry: [country name]\n"
    "city: [city name]\n\nYou MUST output the thinking process in <think> </think> and "
    "give answer in <answer> </answer> tags."
)

PROMPTS = {
    "geoagent": {
        "system": GEOAGENT_SYSTEM,
        "user": "Based on the image, tell me the specific location and your thinking process",
        "max_tokens": 2048,
    },
    "gaea": {
        "system": None,
        "user": GAEA_PREAMBLE
        + "\nQuestion: Where is this image taken? Respond with only the city and country.",
        "max_tokens": 64,
    },
    "globe": {
        "system": "You are a helpful assistant.",
        "user": GLOBE_PROMPT,
        "max_tokens": 1024,
    },
    "qwen": {
        "system": JSON_SYSTEM,
        "user": "Where was this photo taken?",
        "max_tokens": 600,
    },
    "gemma": {
        "system": JSON_SYSTEM,
        "user": "Where was this photo taken?",
        "max_tokens": 600,
    },
    "gemma27": {
        "system": JSON_SYSTEM,
        "user": "Where was this photo taken?",
        "max_tokens": 600,
    },
    "pixtral": {
        "system": JSON_SYSTEM,
        "user": "Where was this photo taken?",
        "max_tokens": 600,
    },
}

volume = modal.Volume.from_name("geo-models", create_if_missing=True)

download_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.6.0", index_url="https://download.pytorch.org/whl/cpu")
    # transformers 4.52 renamed Qwen2.5-VL's layers; GAEA's adapter uses the old
    # `model.layers.*` names, so the merge must run on an earlier release.
    .pip_install(
        "transformers==4.51.3",
        "peft==0.15.2",
        "accelerate",
        "safetensors",
        "huggingface_hub[hf_transfer]<1.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vllm_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "vllm==0.10.1.1", "transformers==4.55.4", "pillow"
)

geoclip_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.6.0",
        "torchvision==0.21.0",
        index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install("geoclip==1.2.1", "transformers==4.55.4")
    .env({"HF_HOME": HF_HOME})
)

web_image = modal.Image.debian_slim(python_version="3.12").pip_install(
    "fastapi[standard]"
)

app = modal.App(APP_NAME)


@app.function(
    image=download_image,
    volumes={MODELS_DIR: volume},
    cpu=4,
    memory=49152,
    timeout=3 * 3600,
)
def download_models(force: bool = False) -> None:
    """Fetch every model's weights, merging GAEA's LoRA into Qwen weights."""
    import shutil
    from pathlib import Path

    from huggingface_hub import snapshot_download

    def done(path: str) -> bool:
        return not force and (Path(path) / "config.json").exists()

    for name, repo in DOWNLOADS.items():
        if not done(MODEL_PATHS[name]):
            snapshot_download(repo, local_dir=MODEL_PATHS[name])
            volume.commit()

    # GeoCLIP loads CLIP by repo id at start-up; caching it in the volume's
    # HF_HOME avoids a 1.7 GB download on every cold start.
    snapshot_download(
        CLIP_REPO,
        cache_dir=f"{HF_HOME}/hub",
        allow_patterns=["*.json", "*.txt", "model.safetensors"],
    )
    volume.commit()

    if not done(MODEL_PATHS["gaea"]):
        import torch
        from peft import PeftModel
        from transformers import Qwen2_5_VLForConditionalGeneration

        adapter_dir = snapshot_download(
            GAEA_REPO, allow_patterns=["adapter_*", "*.json"]
        )
        base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_PATHS["qwen"], torch_dtype=torch.bfloat16
        )
        probe = "model.layers.0.self_attn.q_proj.weight"
        before = base.state_dict()[probe].clone()
        merged = PeftModel.from_pretrained(base, adapter_dir).merge_and_unload()
        if torch.equal(before, merged.state_dict()[probe]):
            raise RuntimeError("GAEA adapter did not change the base weights; merge failed.")

        out = Path(MODEL_PATHS["gaea"])
        merged.save_pretrained(out, safe_serialization=True)
        for name in (
            "preprocessor_config.json",
            "chat_template.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt",
            "generation_config.json",
        ):
            src = Path(MODEL_PATHS["qwen"]) / name
            if src.exists():
                shutil.copy(src, out / name)
        volume.commit()

    for name, path in MODEL_PATHS.items():
        print(f"{name}: {'ready' if (Path(path) / 'config.json').exists() else 'MISSING'}")


def _load_image(image_b64: str):
    """Decode an upload to an RGB PIL image capped at MAX_IMAGE_SIDE."""
    import base64
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(image_b64)))
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    return img


def _to_data_url(image_b64: str) -> str:
    """Normalise any uploaded image to an RGB JPEG (drops EXIF) as a data URL."""
    import base64
    import io

    img = _load_image(image_b64)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _parse_json(text: str) -> dict | None:
    import json
    import re

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    body = fenced.group(1) if fenced else text
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(body[start : end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _answer(model: str, text: str, parsed: dict | None) -> str:
    """A one-line location answer for display and later geocoding on Render."""
    if model == "geoagent" and parsed and parsed.get("FinalAnswer"):
        return str(parsed["FinalAnswer"]).strip()
    if model == "globe":
        import re

        found = {
            k.lower(): v.strip()
            for k, v in re.findall(r"(?im)^\s*(country|city)\s*:\s*(.+?)\s*$", text)
        }
        if found:
            return ", ".join(found[k] for k in ("city", "country") if found.get(k))
    if parsed and any(parsed.get(k) for k in ("specific_location", "city", "country")):
        parts = [
            parsed.get(k)
            for k in ("specific_location", "city", "region", "country")
            if parsed.get(k)
        ]
        if parts:
            return ", ".join(str(p) for p in parts)
    return text.strip().splitlines()[0][:300] if text.strip() else ""


@app.cls(
    image=vllm_image,
    gpu=GPU,
    volumes={MODELS_DIR: volume},
    scaledown_window=SCALEDOWN_S,
    timeout=900,
    max_containers=1,
)
class Geolocator:
    model: str = modal.parameter()

    @modal.enter()
    def load(self) -> None:
        from vllm import LLM

        if self.model not in MODEL_PATHS:
            raise ValueError(f"Unknown model {self.model!r}")
        self.llm = LLM(
            model=MODEL_PATHS[self.model],
            max_model_len=8192,
            gpu_memory_utilization=0.92,
            limit_mm_per_prompt={"image": 1},
            seed=0,
            **ENGINE_KWARGS[self.model],
        )

    @modal.method()
    def locate(self, image_b64: str) -> dict:
        import time

        from vllm import SamplingParams

        spec = PROMPTS[self.model]
        user_text = spec["user"]
        messages = []
        if spec["system"] and self.model in SYSTEM_IN_USER:
            user_text = f"{spec['system']}\n\n{user_text}"
        elif spec["system"]:
            messages.append({"role": "system", "content": spec["system"]})
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _to_data_url(image_b64)}},
                    {"type": "text", "text": user_text},
                ],
            }
        )
        started = time.monotonic()
        out = self.llm.chat(
            messages, SamplingParams(temperature=0, max_tokens=spec["max_tokens"])
        )
        text = out[0].outputs[0].text
        parsed = _parse_json(text) if self.model not in ("gaea", "globe") else None
        return {
            "model": self.model,
            "answer": _answer(self.model, text, parsed),
            "parsed": parsed,
            "text": text,
            "latency_s": round(time.monotonic() - started, 2),
        }


@app.cls(
    image=geoclip_image,
    cpu=4,
    memory=8192,
    volumes={MODELS_DIR: volume},
    scaledown_window=SCALEDOWN_S,
    timeout=600,
    max_containers=1,
)
class GeoClip:
    @modal.enter()
    def load(self) -> None:
        import torch
        import torch.nn.functional as F
        from geoclip import GeoCLIP

        torch.set_num_threads(4)
        self.model = GeoCLIP().eval()
        # The 100k-point GPS gallery never changes, so encode it once instead
        # of on every predict() call as the library does.
        with torch.no_grad():
            self.gallery = self.model.gps_gallery
            self.gallery_feats = F.normalize(
                self.model.location_encoder(self.gallery), dim=1
            )

    @modal.method()
    def locate(self, image_b64: str) -> dict:
        import time

        import torch
        import torch.nn.functional as F

        started = time.monotonic()
        with torch.no_grad():
            pixels = self.model.image_encoder.preprocess_image(_load_image(image_b64))
            feats = F.normalize(self.model.image_encoder(pixels), dim=1)
            logits = self.model.logit_scale.exp() * feats @ self.gallery_feats.t()
            top = logits.softmax(dim=-1)[0].topk(GEOCLIP_TOP_K)
        predictions = [
            {
                "latitude": round(float(self.gallery[i][0]), 5),
                "longitude": round(float(self.gallery[i][1]), 5),
                "probability": round(float(p), 4),
            }
            for p, i in zip(top.values, top.indices)
        ]
        best = predictions[0]
        return {
            "model": "geoclip",
            "answer": f"{best['latitude']}, {best['longitude']}",
            "parsed": {"predictions": predictions},
            "text": "",
            "latency_s": round(time.monotonic() - started, 2),
        }


def _worker(model: str):
    if model == "geoclip":
        return GeoClip()
    cls = Geolocator.with_options(gpu=MODEL_GPU[model]) if model in MODEL_GPU else Geolocator
    return cls(model=model)


def _spawn(model: str, image_b64: str):
    return _worker(model).locate.spawn(image_b64)


def _fan_out(image_b64: str, models: list[str]) -> dict:
    calls = {m: _spawn(m, image_b64) for m in models}
    results = {}
    for name, call in calls.items():
        try:
            results[name] = call.get(timeout=900)
        except Exception as exc:
            results[name] = {"model": name, "error": f"{type(exc).__name__}: {exc}"}
    return results


@app.function(image=web_image, timeout=900)
@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
def locate(body: dict) -> dict:
    """POST {"image_b64": "...", "models": ["geoagent", ...]} -> {model: result}."""
    image_b64 = body.get("image_b64")
    if not image_b64:
        return {"error": "image_b64 is required"}
    models = [m for m in (body.get("models") or DEFAULT_MODELS) if m in ALL_MODELS]
    return {"results": _fan_out(image_b64, models)}


@app.local_entrypoint()
def main(image: str, models: str = ",".join(DEFAULT_MODELS)) -> None:
    import base64
    import json

    with open(image, "rb") as fh:
        image_b64 = base64.b64encode(fh.read()).decode()
    wanted = [m.strip() for m in models.split(",") if m.strip() in ALL_MODELS]
    for name, result in _fan_out(image_b64, wanted).items():
        if "error" in result:
            print(f"== {name}: ERROR {result['error']}")
            continue
        print(f"== {name} ({result['latency_s']} s): {result['answer']}")
        if result.get("parsed") is None:
            print(result["text"][:1500])
        else:
            print(json.dumps(result["parsed"], ensure_ascii=False, indent=2)[:1500])


@app.local_entrypoint()
def benchmark(
    manifest: str = "eval/model_bench/manifest.json",
    models: str = ",".join(DEFAULT_MODELS),
    out: str = "eval/model_bench/results_local.json",
) -> None:
    """Run each model over every manifest image; results merge into `out`."""
    import base64
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from pathlib import Path

    rows = json.loads(Path(manifest).read_text(encoding="utf-8"))
    image_dir = Path(manifest).parent / "images"
    images = [base64.b64encode((image_dir / r["file"]).read_bytes()).decode() for r in rows]
    out_path = Path(out)
    results = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    wanted = [m.strip() for m in models.split(",") if m.strip() in ALL_MODELS]

    def run(model: str) -> tuple[str, dict]:
        outputs = _worker(model).locate.map(images, return_exceptions=True)
        per_image = {}
        for row, output in zip(rows, outputs):
            if not isinstance(output, dict):
                output = {"model": model, "error": repr(output)}
            per_image[row["id"]] = output
        return model, per_image

    with ThreadPoolExecutor(max_workers=len(wanted)) as pool:
        for future in as_completed([pool.submit(run, m) for m in wanted]):
            model, per_image = future.result()
            results[model] = per_image
            out_path.write_text(
                json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            errors = sum("error" in v for v in per_image.values())
            print(f"== {model}: {len(per_image) - errors} ok, {errors} errors")
