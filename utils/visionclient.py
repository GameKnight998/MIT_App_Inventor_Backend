"""One place that talks to a vision model, whichever provider is configured.

Every vision-powered feature in this app (scene analysis, street-level
refinement, vehicle identification) needs the same thing: hand a model an image
plus a prompt and get structured JSON back. Keeping that in one module means a
provider swap is an env-var change rather than an edit in three files, and it
lets the slower/cheaper trade-off be made per deployment.

Supported providers (set `VISION_PROVIDER`):
  openai     - GPT-4o family. Native JSON mode, most accurate OCR today.
  anthropic  - Claude Sonnet family. Noticeably faster first-token latency.
  ollama     - Any local vision model (llava, qwen2-vl, ...). No API cost.

Only the SDK for the selected provider needs to be installed; imports are lazy
and a missing package degrades to a clear note instead of a crash. Every call
returns `(parsed_json_or_None, note)` so callers never have to catch exceptions.
"""

from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Optional

# Default model per provider, used when VISION_MODEL is unset. Kept as
# "-latest"/family aliases so a provider's newer snapshot is picked up without a
# code change.
_DEFAULT_MODELS = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-5",
    "ollama": "llava:13b",
}

# Magic-byte prefixes -> media type. Anthropic and Ollama need an accurate media
# type; guessing "jpeg" for a PNG makes Anthropic reject the request.
_MAGIC = (
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)

VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT", "60"))
VISION_MAX_TOKENS = int(os.getenv("VISION_MAX_TOKENS", "2000"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")


def provider() -> str:
    """The configured provider name, normalised."""
    return (os.getenv("VISION_PROVIDER") or "openai").strip().lower()


def model_name() -> str:
    """The model to use: explicit VISION_MODEL, else the provider default."""
    explicit = (os.getenv("VISION_MODEL") or "").strip()
    if explicit:
        return explicit
    return _DEFAULT_MODELS.get(provider(), _DEFAULT_MODELS["openai"])


def _media_type(data: bytes) -> str:
    for magic, mime in _MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


def _read(image_path: str) -> tuple[bytes, str]:
    with open(image_path, "rb") as fh:
        data = fh.read()
    return data, _media_type(data)


def vision_available() -> tuple[bool, str]:
    """Whether the configured provider has the credentials it needs."""
    prov = provider()
    if prov == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            return False, "OPENAI_API_KEY not set; vision analysis skipped."
        return True, "ok"
    if prov == "anthropic":
        if not os.getenv("ANTHROPIC_API_KEY"):
            return False, "ANTHROPIC_API_KEY not set; vision analysis skipped."
        return True, "ok"
    if prov == "ollama":
        return True, "ok"  # local server; failure surfaces at call time
    return False, f"Unknown VISION_PROVIDER {prov!r}."


def extract_json(text: str) -> Optional[dict[str, Any]]:
    """Parse JSON from a model reply that may be wrapped in prose or fences.

    Only OpenAI guarantees a bare JSON body, so we strip ```json fences and, as
    a last resort, scan for the outermost balanced {...} block.
    """
    if not text:
        return None
    text = text.strip()

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except json.JSONDecodeError:
                    return None
    return None


def _call_openai(
    image_path: str,
    system: str,
    user: str,
    temperature: float,
    detail: str,
    max_tokens: int,
) -> tuple[Optional[dict[str, Any]], str]:
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed; vision skipped."

    data, mime = _read(image_path)
    b64 = base64.b64encode(data).decode("utf-8")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=VISION_TIMEOUT)
    resp = client.chat.completions.create(
        model=model_name(),
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{mime};base64,{b64}",
                            "detail": detail,
                        },
                    },
                ],
            },
        ],
    )
    return extract_json(resp.choices[0].message.content or ""), "ok"


def _call_anthropic(
    image_path: str,
    system: str,
    user: str,
    temperature: float,
    detail: str,
    max_tokens: int,
) -> tuple[Optional[dict[str, Any]], str]:
    try:
        import anthropic
    except ImportError:
        return None, "anthropic package not installed; vision skipped."

    data, mime = _read(image_path)
    b64 = base64.b64encode(data).decode("utf-8")
    client = anthropic.Anthropic(
        api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=VISION_TIMEOUT
    )
    # Claude has no JSON mode, so the instruction carries the contract and
    # `extract_json` tolerates any preamble that slips through anyway.
    resp = client.messages.create(
        model=model_name(),
        max_tokens=max_tokens,
        temperature=temperature,
        system=system + "\n\nReply with the JSON object only. No prose, no markdown.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": mime,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": user},
                ],
            }
        ],
    )
    text = "".join(
        block.text for block in resp.content if getattr(block, "type", "") == "text"
    )
    return extract_json(text), "ok"


def _call_ollama(
    image_path: str,
    system: str,
    user: str,
    temperature: float,
    detail: str,
    max_tokens: int,
) -> tuple[Optional[dict[str, Any]], str]:
    import requests

    data, _ = _read(image_path)
    b64 = base64.b64encode(data).decode("utf-8")
    resp = requests.post(
        f"{OLLAMA_URL.rstrip('/')}/api/chat",
        json={
            "model": model_name(),
            "stream": False,
            "format": "json",
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user, "images": [b64]},
            ],
        },
        timeout=VISION_TIMEOUT,
    )
    resp.raise_for_status()
    return extract_json(resp.json().get("message", {}).get("content", "")), "ok"


_DISPATCH = {
    "openai": _call_openai,
    "anthropic": _call_anthropic,
    "ollama": _call_ollama,
}


def call_vision_json(
    image_path: str,
    system: str,
    user: str,
    *,
    temperature: float = 0.2,
    detail: str = "high",
    max_tokens: Optional[int] = None,
) -> tuple[Optional[dict[str, Any]], str]:
    """Ask the configured vision model for structured JSON about an image.

    Returns `(parsed, note)`. `parsed` is None on any failure and `note`
    explains why, so callers can degrade gracefully rather than raise.
    """
    ok, note = vision_available()
    if not ok:
        return None, note

    fn = _DISPATCH.get(provider())
    if fn is None:
        return None, f"Unknown VISION_PROVIDER {provider()!r}."

    try:
        parsed, call_note = fn(
            image_path,
            system,
            user,
            temperature,
            detail,
            int(max_tokens or VISION_MAX_TOKENS),
        )
    except Exception as exc:
        return None, f"Vision call failed ({provider()}): {exc}"

    if parsed is None:
        return None, "Vision model did not return usable JSON."
    return parsed, call_note
