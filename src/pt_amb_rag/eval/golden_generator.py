from __future__ import annotations

import json

from pt_amb_rag.generation.llm import generate

# Keep this prompt byte-stable: rewrapping it changes the bytes sent to
# the model and silently moves the golden-set numbers.
GOLDEN_PROMPT = """Estás a criar perguntas de avaliação para um sistema de RAG sobre legislação portuguesa.

Dado o seguinte fragmento de texto legal, gera exatamente 3 perguntas distintas em linguagem natural que este fragmento — e só este fragmento — responde. Varia a formulação e a especificidade para que as perguntas não sejam quase duplicadas umas das outras.

Cabeçalho do fragmento: {header}
Texto do fragmento:
{text}

Responde APENAS com um array JSON, sem outro texto, sem blocos de código markdown, neste formato exato:
[
  {{"question": "...", "difficulty": "easy" | "medium" | "hard", "answerable": true}},
  {{"question": "...", "difficulty": "easy" | "medium" | "hard", "answerable": true}},
  {{"question": "...", "difficulty": "easy" | "medium" | "hard", "answerable": true}}
]
"""

_RETRY_SUFFIX = (
    "\n\nLembrete: devolve APENAS o array JSON. "
    "Sem texto, sem blocos de código markdown, sem explicações."
)


class GoldenGenerationError(Exception):
    """Raised when the model fails to produce parseable JSON after a retry."""


def generate_candidates(chunk: dict, *, temperature: float = 0.2) -> list[dict]:
    """Generate ~3 candidate golden questions for a single chunk.

    Each candidate is annotated with the chunk's content_hash, diploma
    and ELI so manual vetting can trace it back to a specific corpus
    chunk (see the eval-drift pitfall — this linkage is load-bearing).
    """
    prompt = GOLDEN_PROMPT.format(
        header=chunk.get("header") or "",
        text=chunk["text"],
    )

    raw = generate(prompt, temperature=temperature)
    candidates = _parse_json_array(raw)

    if candidates is None:
        raw_retry = generate(prompt + _RETRY_SUFFIX, temperature=temperature)
        candidates = _parse_json_array(raw_retry)

    if candidates is None:
        raise GoldenGenerationError(
            f"could not parse JSON for chunk {chunk.get('content_hash')!r}"
        )

    for c in candidates:
        c["content_hash"] = chunk.get("content_hash")
        c["regulation"] = chunk.get("regulation")
        c["diploma"] = chunk.get("diploma")
        c["eli"] = chunk.get("eli")

    return candidates


def _parse_json_array(text: str) -> list[dict] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None

    return data if isinstance(data, list) else None
