from types import SimpleNamespace
from unittest.mock import patch

import pytest

from pt_amb_rag.api.ask import ask


def make_result(payload: dict | None, score: float) -> SimpleNamespace:
    # Stand-in for a qdrant_client ScoredPoint: just needs .score and .payload.
    return SimpleNamespace(score=score, payload=payload)


# ---------------------------------------------------------------------------
# ask — mocked embed/search/generate, no network, no real DB
# ---------------------------------------------------------------------------


class TestAsk:
    def test_returns_answer_and_sources(self):
        result = make_result(
            {
                "diploma": "Decreto-Lei n.º 150/2015",
                "article_num": "26",
                "chapter": "4 — Deployer obligations",
                "header": "Article 26",
                "text": "Deployers of high-risk AI systems shall ensure...",
            },
            score=0.87,
        )
        with (
            patch(
                "pt_amb_rag.api.ask.embed", return_value=[[0.1, 0.2, 0.3]]
            ) as mock_embed,
            patch(
                "pt_amb_rag.api.ask.search", return_value=[result]
            ) as mock_search,
            patch(
                "pt_amb_rag.api.ask.generate", return_value="Deployers must..."
            ) as mock_generate,
        ):
            response = ask("What are the deployer obligations?")

        assert response["answer"] == "Deployers must..."
        assert response["sources"] == [
            {
                "score": 0.87,
                "diploma": "Decreto-Lei n.º 150/2015",
                "article_num": "26",
                "chapter": "4 — Deployer obligations",
                "header": "Article 26",
                "text": "Deployers of high-risk AI systems shall ensure...",
            }
        ]
        mock_embed.assert_called_once()
        mock_search.assert_called_once()
        mock_generate.assert_called_once()

    def test_embeds_question_and_uses_first_vector(self):
        with (
            patch(
                "pt_amb_rag.api.ask.embed", return_value=[[0.5, 0.6]]
            ) as mock_embed,
            patch("pt_amb_rag.api.ask.search", return_value=[]) as mock_search,
            patch("pt_amb_rag.api.ask.generate", return_value="answer"),
        ):
            ask("some question")

        mock_embed.assert_called_once_with(["some question"])
        # the embedded vector should be forwarded to search as the query
        args, kwargs = mock_search.call_args
        assert (args[0] if args else kwargs.get("query_embedding")) == [
            0.5,
            0.6,
        ]

    def test_passes_k_and_filters_to_search(self):
        with (
            patch("pt_amb_rag.api.ask.embed", return_value=[[0.1]]),
            patch("pt_amb_rag.api.ask.search", return_value=[]) as mock_search,
            patch("pt_amb_rag.api.ask.generate", return_value="a"),
        ):
            ask("q", k=3, regulation="dl_127_2013", article_num="21")

        _, kwargs = mock_search.call_args
        assert kwargs["k"] == 3
        assert kwargs["regulation"] == "dl_127_2013"
        assert kwargs["article_num"] == "21"

    def test_default_k_is_five(self):
        with (
            patch("pt_amb_rag.api.ask.embed", return_value=[[0.1]]),
            patch("pt_amb_rag.api.ask.search", return_value=[]) as mock_search,
            patch("pt_amb_rag.api.ask.generate", return_value="a"),
        ):
            ask("q")

        _, kwargs = mock_search.call_args
        assert kwargs["k"] == 5

    def test_generate_receives_a_prompt_containing_question_and_context(self):
        result = make_result(
            {
                "diploma": "Decreto-Lei n.º 150/2015",
                "article_num": "8",
                "text": "chunk text",
            },
            score=0.9,
        )
        with (
            patch("pt_amb_rag.api.ask.embed", return_value=[[0.1]]),
            patch("pt_amb_rag.api.ask.search", return_value=[result]),
            patch(
                "pt_amb_rag.api.ask.generate", return_value="a"
            ) as mock_generate,
        ):
            ask("What are deployer obligations?")

        prompt = mock_generate.call_args[0][0]
        assert "What are deployer obligations?" in prompt
        assert "chunk text" in prompt
        assert "Diploma: Decreto-Lei n.º 150/2015" in prompt

    def test_no_results_still_returns_answer_with_empty_sources(self):
        with (
            patch("pt_amb_rag.api.ask.embed", return_value=[[0.1]]),
            patch("pt_amb_rag.api.ask.search", return_value=[]),
            patch(
                "pt_amb_rag.api.ask.generate",
                return_value="The corpus does not address this.",
            ),
        ):
            response = ask("something not in the corpus")

        assert response["sources"] == []
        assert response["answer"] == "The corpus does not address this."

    def test_source_with_none_payload_does_not_crash(self):
        result = make_result(None, score=0.5)
        with (
            patch("pt_amb_rag.api.ask.embed", return_value=[[0.1]]),
            patch("pt_amb_rag.api.ask.search", return_value=[result]),
            patch("pt_amb_rag.api.ask.generate", return_value="a"),
        ):
            response = ask("q")

        assert response["sources"] == [{"score": 0.5}]

    def test_multiple_sources_preserve_score_and_payload_fields(self):
        r1 = make_result(
            {
                "diploma": "Decreto-Lei n.º 150/2015",
                "article_num": "8",
            },
            score=0.9,
        )
        r2 = make_result(
            {"diploma": "Decreto-Lei n.º 127/2013", "article_num": "8"},
            score=0.7,
        )
        with (
            patch("pt_amb_rag.api.ask.embed", return_value=[[0.1]]),
            patch("pt_amb_rag.api.ask.search", return_value=[r1, r2]),
            patch("pt_amb_rag.api.ask.generate", return_value="a"),
        ):
            response = ask("q")

        assert len(response["sources"]) == 2
        assert response["sources"][0] == {
            "score": 0.9,
            "diploma": "Decreto-Lei n.º 150/2015",
            "article_num": "8",
        }
        assert response["sources"][1] == {
            "score": 0.7,
            "diploma": "Decreto-Lei n.º 127/2013",
            "article_num": "8",
        }


# ---------------------------------------------------------------------------
# Integration test — real embedder, real Qdrant, real MiniMax API.
# Slow, needs a running Qdrant (localhost:6333) with an already-built index
# and a valid MINIMAX_API_KEY. Skipped by default (see pyproject.toml
# addopts = "-m 'not integration'").
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAskIntegration:
    def test_ask_real_pipeline(self):
        question = (
            "What are the deployer obligations under Article 26 of the AI Act?"
        )
        r = ask(question)

        print("ANSWER:", r["answer"])
        print()
        print("SOURCES:")
        for s in r["sources"]:
            print(
                f"  [{s['score']:.3f}] {s['regulation']} "
                f"Art. {s['article_num']} — {s['header']}"
            )

        assert isinstance(r["answer"], str)
        assert r["answer"] != ""

        assert isinstance(r["sources"], list)
        assert len(r["sources"]) > 0

        for source in r["sources"]:
            assert isinstance(source["score"], float)
            assert "regulation" in source
            assert "article_num" in source
            assert "header" in source

        # this question is specifically about AI Act Art. 26, so the top
        # source should plausibly be from there -- a soft, informative
        # check rather than a hard requirement on retrieval ranking
        top_source = r["sources"][0]
        if top_source["diploma"] != "Decreto-Lei n.º 150/2015":
            print(
                f"NOTE: top source was {top_source['diploma']} "
                f"Art. {top_source['article_num']}, not ai_act — "
                "check retrieval quality if this is unexpected."
            )
