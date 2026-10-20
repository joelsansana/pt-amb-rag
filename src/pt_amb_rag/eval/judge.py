from __future__ import annotations

import json
import logging

from pt_amb_rag.generation.llm import generate

logger = logging.getLogger(__name__)

# The rubric is in the corpus language on purpose: an English rubric
# applied to Portuguese legal text is measurably less consistent
# (cross-language judge agreement collapses), and the answer being
# judged is in Portuguese.

_ANSWERABLE_CRITERIA = """\
- faithfulness: "yes" | "no" | "partial" — Cada afirmação da resposta é
  suportada pelo contexto recuperado? Se "no" ou "partial", transcreve
  cada trecho não suportado em "evidence".
- citation_accuracy: "yes" | "no" — Cada artigo/citação referida na
  resposta contém de facto a afirmação que lhe é atribuída? Compara o
  texto da fonte citada com a afirmação."""

_ABSTENTION_CRITERIA = """\
- abstention: "yes" | "no" — A pergunta não é respondível a partir do
  contexto recuperado. A resposta recusou-se corretamente a responder
  (por exemplo, "O corpus não responde a esta pergunta.") em vez de
  fabricar uma resposta?"""

_VERDICT_SHAPE_ANSWERABLE = """\
{"faithfulness": "yes" | "no" | "partial", "citation_accuracy": "yes" | "no", "evidence": "..."}"""

_VERDICT_SHAPE_ABSTENTION = """\
{"abstention": "yes" | "no", "evidence": "..."}"""

_ALLOWED = {
    "faithfulness": {"yes", "no", "partial"},
    "citation_accuracy": {"yes", "no"},
    "abstention": {"yes", "no"},
}

_RETRY_SUFFIX = (
    "\n\nLembrete: devolve APENAS o objeto JSON. "
    "Sem texto, sem blocos de código markdown, sem explicações."
)


class JudgeError(Exception):
    """Raised when the judge fails to produce a valid verdict after a retry."""


def judge(
    question: str,
    sources: list[dict],
    answer: str,
    *,
    expect_abstention: bool = False,
    temperature: float = 0.2,
) -> dict:
    """Judge one RAG answer against its retrieved context.

    expect_abstention=False  -> verdict: faithfulness, citation_accuracy
    expect_abstention=True   -> verdict: abstention

    Raises JudgeError after one stricter retry if the output is unparseable
    or uses a value outside the allowed set. The runner should catch this
    and mark the question unjudged (same failure-tolerant pattern as
    run_eval.py's n_failed).
    """
    criteria = (
        _ABSTENTION_CRITERIA if expect_abstention else _ANSWERABLE_CRITERIA
    )
    shape = (
        _VERDICT_SHAPE_ABSTENTION
        if expect_abstention
        else _VERDICT_SHAPE_ANSWERABLE
    )

    prompt = f"""És um avaliador de um sistema de RAG sobre legislação portuguesa.

Pergunta: {question}

Contexto recuperado (fontes numeradas com os respetivos cabeçalhos):
{_format_sources(sources)}

Resposta: {answer}

Para cada critério abaixo, responde apenas com JSON:
{criteria}

O campo "evidence" é uma transcrição curta ou a justificação dos veredictos.

Responde APENAS com o objeto JSON, sem outro texto, sem blocos de código markdown, neste formato exato:
{shape}"""

    verdict = _parse_and_validate(
        generate(prompt, temperature=temperature), expect_abstention
    )
    if verdict is None:
        verdict = _parse_and_validate(
            generate(prompt + _RETRY_SUFFIX, temperature=temperature),
            expect_abstention,
        )
    if verdict is None:
        raise JudgeError(
            f"judge produced no valid verdict for question: {question[:80]!r}"
        )
    return verdict


def _format_sources(sources: list[dict]) -> str:
    parts = []
    for i, s in enumerate(sources, 1):
        header = s.get("header") or "(no header)"
        text = s.get("text") or ""
        parts.append(f"[{i}] {header}\n{text}")
    return "\n\n".join(parts)


def _parse_and_validate(raw: str, expect_abstention: bool) -> dict | None:
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    required = (
        ["abstention"]
        if expect_abstention
        else ["faithfulness", "citation_accuracy"]
    )
    for key in required:
        if data.get(key) not in _ALLOWED[key]:
            return None
    if not isinstance(data.get("evidence", ""), str):
        return None
    return data
