from typing import Any

import pytest
from pydantic import ValidationError

from pt_amb_rag.api.schemas import AskRequest, SearchRequest, SourceChunk


def make_chunk(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = dict(
        score=0.9,
        regulation="dl_150_2015",
        part="article",
        diploma="Decreto-Lei n.º 150/2015",
        eli="https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html",
        header="Artigo 26.º — Decreto-Lei n.º 150/2015",
        text="O operador é responsável por...",
        content_hash="abc123",
    )
    base.update(overrides)
    return base


# --- AskRequest ---


def test_ask_defaults():
    req = AskRequest(question="hello")
    assert req.k == 5
    assert req.regulation is None
    assert req.article_num is None
    assert req.part is None


def test_ask_accepts_filters():
    req = AskRequest(
        question="q",
        k=10,
        regulation="dl_127_2013",
        article_num=21,
        part="article",
    )
    assert (req.k, req.regulation, req.article_num, req.part) == (
        10,
        "dl_127_2013",
        21,
        "article",
    )


@pytest.mark.parametrize("question", ["", "x" * 2001])
def test_ask_rejects_bad_question_length(question):
    with pytest.raises(ValidationError):
        AskRequest(question=question)


@pytest.mark.parametrize("k", [0, -1, 21, 1000])
def test_ask_rejects_k_out_of_range(k):
    with pytest.raises(ValidationError):
        AskRequest(question="q", k=k)


@pytest.mark.parametrize("k", [1, 20])
def test_ask_accepts_k_boundaries(k):
    assert AskRequest(question="q", k=k).k == k


def test_ask_accepts_max_length_question():
    assert len(AskRequest(question="x" * 2000).question) == 2000


def test_ask_requires_question():
    with pytest.raises(ValidationError):
        AskRequest.model_validate({})


def test_ask_rejects_non_int_article_num():
    with pytest.raises(ValidationError):
        AskRequest.model_validate(
            {"question": "q", "article_num": "not a number"}
        )


# --- SearchRequest ---


def test_search_defaults():
    req = SearchRequest(question="hello")
    assert req.k == 5


@pytest.mark.parametrize("k", [0, 21])
def test_search_rejects_k_out_of_range(k):
    with pytest.raises(ValidationError):
        SearchRequest(question="q", k=k)


def test_search_rejects_empty_question():
    with pytest.raises(ValidationError):
        SearchRequest(question="")


# --- SourceChunk ---


def test_source_chunk_optional_fields_default_to_none():
    chunk = SourceChunk(**make_chunk())
    assert chunk.article_num is None
    assert chunk.numero is None
    assert chunk.alinea is None
    assert chunk.annex_id is None
    assert chunk.chapter is None


def test_source_chunk_accepts_optional_fields():
    chunk = SourceChunk(**make_chunk(article_num=26, chapter="CAPÍTULO III"))
    assert chunk.article_num == 26
    assert chunk.chapter == "CAPÍTULO III"


@pytest.mark.parametrize(
    "missing",
    [
        "score",
        "regulation",
        "part",
        "diploma",
        "eli",
        "header",
        "text",
        "content_hash",
    ],
)
def test_source_chunk_requires_core_fields(missing):
    data = make_chunk()
    del data[missing]
    with pytest.raises(ValidationError):
        SourceChunk(**data)


def test_source_chunk_ignores_extra_payload_keys():
    # Qdrant payloads may carry keys the schema doesn't model
    chunk = SourceChunk(**make_chunk(some_extra_key="ignored"))
    assert not hasattr(chunk, "some_extra_key")


def test_source_chunk_dump_roundtrip():
    chunk = SourceChunk(**make_chunk(article_num=26))
    assert SourceChunk(**chunk.model_dump()) == chunk
