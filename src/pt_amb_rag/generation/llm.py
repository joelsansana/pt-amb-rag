from __future__ import annotations

import os
import re
from typing import Any

from openai import OpenAI

MODEL = "MiniMax-M3"


def get_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.minimax.io/v1",
        api_key=os.environ["MINIMAX_API_KEY"],
    )


def generate(prompt: str, *, temperature: float | None = None) -> str:
    """Generate a completion from MiniMax-M3.

    MiniMax-M3 always reasons internally — there is no way to disable
    thinking for this model family, so no reasoning-mode toggle exists
    here. Thinking arrives inline as <think>...</think> and is stripped
    before returning.

    temperature: optional sampling temperature, range [0, 2] per MiniMax's
    API (defaults to 1 server-side if omitted). Pass a low value (e.g. 0.2)
    for deterministic/structured tasks like golden-set generation.
    """
    client = get_client()

    # Typed as dict[str, Any] rather than dict[str, float]: a float-valued
    # **kwargs makes the call unfalsifiable, since only temperature is a
    # float parameter and every other overload parameter must reject it.
    kwargs: dict[str, Any] = {}
    if temperature is not None:
        kwargs["temperature"] = temperature

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        **kwargs,
    )
    raw = response.choices[0].message.content or ""

    return _strip_thinking(raw)


_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove any <think>...</think> reasoning block a model may emit
    inline in its response content (safety net even with thinking disabled)."""
    return _THINK_TAG_RE.sub("", text).strip()
