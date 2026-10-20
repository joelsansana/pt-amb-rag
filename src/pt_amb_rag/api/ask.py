from __future__ import annotations

from typing import Any

from pt_amb_rag.generation.llm import generate
from pt_amb_rag.generation.prompts import build_prompt
from pt_amb_rag.retrieval.embedder import embed
from pt_amb_rag.retrieval.vector_store import search


def ask(
    question: str, *, k: int = 5, answer: bool = True, **filters: Any
) -> dict[str, Any]:
    """End-to-end RAG: embed the question, retrieve the top-k matching
    chunks, generate an answer grounded in that context, and return both.

    `filters` are forwarded to vector_store.search (currently `regulation`
    and/or `article_num`), e.g. ask("...", regulation="dl_150_2015").

    answer=False skips generation and returns answer=None — retrieval-only
    mode for the keyless CI eval (run_eval.py --no-llm), which discards
    the generated text anyway. The embedder is local, so this path needs
    no MINIMAX_API_KEY.
    """
    q_emb = embed([question])[0]
    results = search(q_emb, k=k, **filters)

    if not answer:
        return {
            "answer": None,
            "sources": [
                {"score": r.score, **(r.payload or {})} for r in results
            ],
        }

    prompt = build_prompt(question, results)
    answer_text = generate(prompt)

    return {
        "answer": answer_text,
        "sources": [{"score": r.score, **(r.payload or {})} for r in results],
    }
