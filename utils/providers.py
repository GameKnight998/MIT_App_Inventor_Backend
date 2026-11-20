"""Send the same vision prompt to OpenAI, Anthropic (Claude) or Google (Gemini).

The provider is picked from the model name:
  claude-...  -> Anthropic Messages API  (needs ANTHROPIC_API_KEY)
  gemini-...  -> Google Gemini API       (needs GEMINI_API_KEY or GOOGLE_API_KEY)
  anything else -> OpenAI Chat Completions (needs OPENAI_API_KEY)

Every provider gets the identical system prompt, user text and images, and
returns raw JSON strings, so model comparisons measure the model and nothing
else.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Optional

# Claude models that support the server-side refusal fallback. A safety
# classifier occasionally declines a benign request; with fallbacks enabled the
# API re-runs it on a fallback model instead of returning nothing.
_CLAUDE_FALLBACK_MODELS = (
    "claude-fable-5-2",
    "claude-fable-5-1",
    "claude-opus-5-5",
    "claude-opus-5",
    "claude-sonnet-5-5",
)
CLAUDE_EFFORT = os.getenv("CLAUDE_EFFORT", "high")


class ProviderError(RuntimeError):
    pass


def provider_for(model: str) -> str:
    m = (model or "").lower()
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("gemini-"):
        return "google"
    return "openai"


def missing_key(model: str) -> Optional[str]:
    """Name of the environment variable this model needs, if it is unset."""
    provider = provider_for(model)
    if provider == "anthropic":
        return None if os.getenv("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY"
    if provider == "google":
        return None if (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) else "GEMINI_API_KEY"
    return None if os.getenv("OPENAI_API_KEY") else "OPENAI_API_KEY"


def extract_json(text: str) -> str:
    """Pull the JSON object out of a reply that may wrap it in prose or fences."""
    text = (text or "").strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.S)
    if fenced:
        return fenced.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        return text[start : end + 1]
    return text


def _parallel(fn: Any, n: int) -> list[Any]:
    from utils.cache import parallel_map

    return parallel_map(fn, list(range(n)))


# --- OpenAI -----------------------------------------------------------------


def _openai(model: str, system: str, text: str, images: list[str], n: int, temperature: float) -> list[str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    content: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for b64 in images:
        content.append(
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "high"}}
        )
    base = {
        "model": model,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": content}],
    }
    try:
        # One request with n choices: input tokens are billed once.
        resp = client.chat.completions.create(**base, temperature=temperature, n=n)
        return [c.message.content or "{}" for c in resp.choices]
    except Exception as exc:
        msg = str(exc).lower()
        if "temperature" not in msg and "'n'" not in msg and "parameter" not in msg:
            raise
    # Some reasoning models reject n / temperature: separate default requests.
    return _parallel(
        lambda _: client.chat.completions.create(**base).choices[0].message.content or "{}", n
    )


# --- Anthropic --------------------------------------------------------------


def _anthropic(model: str, system: str, text: str, images: list[str], n: int) -> list[str]:
    import anthropic

    client = anthropic.Anthropic()
    content: list[dict[str, Any]] = [
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
        for b64 in images
    ]
    content.append({"type": "text", "text": text + "\nRespond with the JSON object only."})
    params: dict[str, Any] = {
        "model": model,
        "max_tokens": 16000,
        "system": system,
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": CLAUDE_EFFORT},
    }

    def one(_: int) -> str:
        if model in _CLAUDE_FALLBACK_MODELS:
            resp = client.beta.messages.create(
                **params, betas=["server-side-fallback-2026-07-01"], fallbacks="default"
            )
        else:
            resp = client.messages.create(**params)
        if resp.stop_reason == "refusal":
            raise ProviderError(f"{model} declined the request (refusal).")
        reply = "".join(b.text for b in resp.content if b.type == "text")
        return extract_json(reply)

    # Current Claude models take no sampling parameters, and the API has no
    # `n`: independent samples are separate requests.
    return _parallel(one, n)


# --- Google Gemini ----------------------------------------------------------


def _gemini(model: str, system: str, text: str, images: list[str], n: int, temperature: float) -> list[str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
    parts: list[Any] = [
        types.Part.from_bytes(data=base64.b64decode(b64), mime_type="image/jpeg") for b64 in images
    ]
    parts.append(text)
    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        temperature=temperature,
    )

    def one(_: int) -> str:
        resp = client.models.generate_content(model=model, contents=parts, config=config)
        return extract_json(resp.text or "{}")

    return _parallel(one, n)


def request_samples(
    model: str,
    system: str,
    text: str,
    images: list[str],
    n: int,
    temperature: float,
) -> list[str]:
    """Return n raw JSON answers from `model` for the given prompt and images."""
    provider = provider_for(model)
    if provider == "anthropic":
        return _anthropic(model, system, text, images, n)
    if provider == "google":
        return _gemini(model, system, text, images, n, temperature)
    return _openai(model, system, text, images, n, temperature)


def is_json(text: str) -> bool:
    try:
        return isinstance(json.loads(text), dict)
    except (TypeError, ValueError):
        return False
