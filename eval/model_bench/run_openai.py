"""GPT-4o baselines for the remote-terrain benchmark.

  gpt4o         the raw model, same JSON prompt the open models get
  api_pipeline  the full /analyze pipeline (GPT-4o + map verification + fusion)
  gpt-5.5, gpt-5.4, o3, gpt-4.1
                newer closed OpenAI models, same JSON prompt

Results merge into results_openai.json; existing entries are kept unless
--refresh is passed, so an interrupted run resumes where it stopped.

  python eval/model_bench/run_openai.py [--only gpt4o|api_pipeline] [--refresh]
"""

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "gpu")]

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from modal_app import JSON_SYSTEM  # noqa: E402
from utils.visionclient import call_vision_json  # noqa: E402

OUT = HERE / "results_openai.json"


def _gpt4o(path: Path) -> dict:
    started = time.monotonic()
    parsed, note = call_vision_json(
        str(path), JSON_SYSTEM, "Where was this photo taken?", temperature=0.0
    )
    return {
        "model": "gpt4o",
        "parsed": parsed,
        "error": None if parsed else note,
        "latency_s": round(time.monotonic() - started, 2),
    }


def _reasoning_model(model: str):
    """GPT-5 / o-series: no temperature, and max_completion_tokens covers reasoning."""

    def run(path: Path) -> dict:
        import base64

        from openai import OpenAI

        from utils.visionclient import extract_json

        b64 = base64.b64encode(path.read_bytes()).decode()
        started = time.monotonic()
        resp = OpenAI(timeout=300).chat.completions.create(
            model=model,
            max_completion_tokens=16000,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": JSON_SYSTEM},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Where was this photo taken?"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}},
                    ],
                },
            ],
        )
        parsed = extract_json(resp.choices[0].message.content or "")
        return {
            "model": model,
            "parsed": parsed,
            "error": None if parsed else "no JSON in reply",
            "latency_s": round(time.monotonic() - started, 2),
            "usage": {
                "input": resp.usage.prompt_tokens,
                "output": resp.usage.completion_tokens,
            },
        }

    return run


def _pipeline(path: Path) -> dict:
    from main import _analyze_image_bytes

    started = time.monotonic()
    status, payload = _analyze_image_bytes(path.read_bytes(), path.name, "image/jpeg")
    return {
        "model": "api_pipeline",
        "parsed": {
            k: payload.get(k)
            for k in ("latitude", "longitude", "location_name", "confidence", "country")
        },
        "error": None if status == 200 and payload.get("latitude") is not None else f"status {status}",
        "latency_s": round(time.monotonic() - started, 2),
    }


def _pipeline_with(model: str):
    """The full pipeline with VISION_MODEL swapped (run one per process)."""

    def run(path: Path) -> dict:
        import os

        os.environ["VISION_MODEL"] = model
        result = _pipeline(path)
        result["model"] = f"pipeline_{model}"
        return result

    return run


RUNNERS = {
    "gpt4o": _gpt4o,
    "api_pipeline": _pipeline,
    **{f"pipeline_{m}": _pipeline_with(m) for m in ("gpt-5.5", "o3")},
    **{m: _reasoning_model(m) for m in ("gpt-5.5", "gpt-5.4", "o3", "gpt-4.1")},
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=sorted(RUNNERS), action="append")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default=str(OUT), help="results file (one per process when running in parallel)")
    ap.add_argument("--limit", type=int, help="only the first N images (smoke test)")
    args = ap.parse_args()

    out = Path(args.out)
    rows = json.loads((HERE / "manifest.json").read_text(encoding="utf-8"))[: args.limit]
    results = {} if args.refresh or not out.exists() else json.loads(out.read_text(encoding="utf-8"))
    for name, runner in RUNNERS.items():
        if args.only and name not in args.only:
            continue
        per_image = results.setdefault(name, {})
        for row in rows:
            if row["id"] in per_image and not per_image[row["id"]].get("error"):
                continue
            try:
                per_image[row["id"]] = runner(HERE / "images" / row["file"])
            except Exception as exc:
                per_image[row["id"]] = {"model": name, "error": f"{type(exc).__name__}: {exc}"}
            out.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"{name} {row['id']}: {per_image[row['id']].get('error') or 'ok'}")


if __name__ == "__main__":
    main()
