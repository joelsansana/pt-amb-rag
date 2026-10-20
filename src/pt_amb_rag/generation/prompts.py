from __future__ import annotations

from collections.abc import Sequence
from typing import Any

# The abstention sentinel is a deterministic string: the answer must be
# exactly this when the corpus does not address the question. It is
# matched by the judge and by the unanswerable tier, so it must change
# atomically with `eval/judge.py` and `eval/golden_generator.py`.
#
# Keep these long lines as they are. Rewrapping a prompt changes the
# bytes sent to the model, which silently moves the judge and
# golden-set numbers this repo reports.
RAG_PROMPT = """Assistente de conformidade regulatória sobre legislação portuguesa de ambiente e segurança industrial.

Responde à pergunta do utilizador usando APENAS o contexto recuperado abaixo. Cita cada afirmação com o diploma, o artigo (e o número/alínea ou anexo, se aplicável), e transcreve o texto de apoio exato.

Se o contexto não contiver informação suficiente para responder, responde exatamente:
"O corpus não responde a esta pergunta."

Cita assim: [Decreto-Lei n.º 150/2015, art. 8.º, n.º 2, al. b)] ou [Portaria n.º 38/2025, anexo I]. Nunca cites só o diploma sem o artigo.

Contexto recuperado:
{context}

Pergunta: {question}

Resposta (com citações no formato [Diploma n.º X, art. Y]):"""

MAX_CHUNK_CHARS = 1500


def _truncate(text: str, max_chars: int = MAX_CHUNK_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _format_source_header(payload: dict[str, Any]) -> str:
    """Build the '[Source N | ...]' header fields from a chunk's payload.

    Only includes fields that are present, since a chunk may be tied to
    an Article, a numbered paragraph, an annex or a preamble rather
    than all of them at once."""
    parts = [f"Diploma: {payload.get('diploma')}"]

    article_num = payload.get("article_num")
    if article_num:
        parts.append(f"Artigo: {article_num}.º")

    numero = payload.get("numero")
    if numero:
        parts.append(f"N.º: {numero}")

    alinea = payload.get("alinea")
    if alinea:
        parts.append(f"Alínea: {alinea})")

    chapter = payload.get("chapter")
    if chapter:
        parts.append(f"Capítulo: {chapter}")

    annex_id = payload.get("annex_id")
    if annex_id:
        parts.append(f"Anexo: {annex_id}")

    return " | ".join(parts)


def format_context(
    results: Sequence[Any], *, max_chunk_chars: int = MAX_CHUNK_CHARS
) -> str:
    """Build the {context} block for RAG_PROMPT from search results.

    `results` are expected to be Qdrant ScoredPoint objects (as returned by
    vector_store.search), each with a `.score` and a `.payload` dict
    containing at least 'text' and 'diploma'. Sorted by score, best
    first; each chunk's text is truncated to `max_chunk_chars`.
    """
    sorted_results = sorted(results, key=lambda r: r.score, reverse=True)

    blocks: list[str] = []
    for i, result in enumerate(sorted_results, start=1):
        payload = result.payload or {}
        header = _format_source_header(payload)
        text = _truncate(payload.get("text", ""), max_chunk_chars)
        blocks.append(f"[Fonte {i} | {header}]\n{text}")

    return "\n\n".join(blocks)


def build_prompt(
    question: str,
    results: Sequence[Any],
    *,
    max_chunk_chars: int = MAX_CHUNK_CHARS,
) -> str:
    """Assemble the full RAG_PROMPT with retrieved context and the question."""
    context = format_context(results, max_chunk_chars=max_chunk_chars)
    return RAG_PROMPT.format(context=context, question=question)
