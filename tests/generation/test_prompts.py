from types import SimpleNamespace

from pt_amb_rag.generation.prompts import (
    MAX_CHUNK_CHARS,
    RAG_PROMPT,
    _format_source_header,
    _truncate,
    build_prompt,
    format_context,
)


def make_result(payload: dict | None, score: float) -> SimpleNamespace:
    # Stand-in for a qdrant_client ScoredPoint: just needs .score and .payload.
    return SimpleNamespace(score=score, payload=payload)


# ---------------------------------------------------------------------------
# _truncate
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("texto curto") == "texto curto"

    def test_text_exactly_at_limit_unchanged(self):
        text = "x" * MAX_CHUNK_CHARS
        assert _truncate(text) == text

    def test_text_over_limit_is_truncated_with_ellipsis(self):
        text = "x" * (MAX_CHUNK_CHARS + 100)
        result = _truncate(text)
        assert len(result) == MAX_CHUNK_CHARS
        assert result.endswith("…")

    def test_truncation_strips_trailing_whitespace_before_ellipsis(self):
        text = "a" * (MAX_CHUNK_CHARS - 1) + "   mais texto cortado"
        result = _truncate(text)
        assert not result[:-1].endswith(" ")
        assert result.endswith("…")

    def test_custom_max_chars(self):
        result = _truncate("olá mundo", max_chars=5)
        # the trailing space is stripped before the ellipsis is appended
        assert result == "olá…"
        assert len(result) == 4

    def test_empty_string(self):
        assert _truncate("") == ""


# ---------------------------------------------------------------------------
# _format_source_header
# ---------------------------------------------------------------------------


class TestFormatSourceHeader:
    def test_article_chunk(self):
        payload = {
            "diploma": "Decreto-Lei n.º 150/2015",
            "article_num": "8",
            "chapter": "CAPÍTULO I",
        }
        header = _format_source_header(payload)
        assert (
            header == "Diploma: Decreto-Lei n.º 150/2015 | Artigo: 8.º | "
            "Capítulo: CAPÍTULO I"
        )

    def test_numbered_paragraph_only(self):
        payload = {"diploma": "Decreto-Lei n.º 150/2015", "numero": "2"}
        header = _format_source_header(payload)
        assert header == "Diploma: Decreto-Lei n.º 150/2015 | N.º: 2"
        assert "Artigo" not in header

    def test_alinea_chunk(self):
        payload = {
            "diploma": "Decreto-Lei n.º 150/2015",
            "article_num": "8",
            "alinea": "b",
        }
        header = _format_source_header(payload)
        assert header == (
            "Diploma: Decreto-Lei n.º 150/2015 | Artigo: 8.º | Alínea: b)"
        )

    def test_annex_only_chunk(self):
        payload = {
            "diploma": "Decreto-Lei n.º 150/2015",
            "annex_id": "I",
        }
        header = _format_source_header(payload)
        assert header == "Diploma: Decreto-Lei n.º 150/2015 | Anexo: I"

    def test_diploma_only(self):
        payload = {"diploma": "Decreto-Lei n.º 127/2013"}
        assert (
            _format_source_header(payload)
            == "Diploma: Decreto-Lei n.º 127/2013"
        )

    def test_field_order_is_stable(self):
        # artigo, n.º, alínea, capítulo, anexo — in that order when present
        payload = {
            "diploma": "Decreto-Lei n.º 150/2015",
            "article_num": "9",
            "numero": "5",
            "alinea": "a",
            "chapter": "2",
            "annex_id": "I",
        }
        header = _format_source_header(payload)
        assert header == (
            "Diploma: Decreto-Lei n.º 150/2015 | Artigo: 9.º | N.º: 5 | "
            "Alínea: a) | Capítulo: 2 | Anexo: I"
        )

    def test_missing_diploma_renders_none(self):
        # defensive: shouldn't crash even with a malformed payload
        header = _format_source_header({})
        assert header == "Diploma: None"


# ---------------------------------------------------------------------------
# format_context
# ---------------------------------------------------------------------------


class TestFormatContext:
    def test_empty_results_returns_empty_string(self):
        assert format_context([]) == ""

    def test_single_result_format(self):
        result = make_result(
            {
                "diploma": "Decreto-Lei n.º 150/2015",
                "article_num": "8",
                "chapter": "CAPÍTULO I",
                "text": "O operador é responsável por...",
            },
            score=0.91,
        )
        context = format_context([result])
        expected = (
            "[Fonte 1 | Diploma: Decreto-Lei n.º 150/2015 | Artigo: 8.º | "
            "Capítulo: CAPÍTULO I]\n"
            "O operador é responsável por..."
        )
        assert context == expected

    def test_sorts_by_score_descending(self):
        low = make_result(
            {"diploma": "dl_127_2013", "text": "bloco fraco"}, score=0.3
        )
        high = make_result(
            {"diploma": "dl_150_2015", "text": "bloco forte"}, score=0.9
        )
        mid = make_result(
            {"diploma": "dl_127_2013", "text": "bloco médio"}, score=0.6
        )

        context = format_context([low, high, mid])

        # "Fonte 1" should be the highest-scoring chunk
        assert context.index("[Fonte 1") < context.index("bloco forte")
        assert context.index("bloco forte") < context.index("bloco médio")
        assert context.index("bloco médio") < context.index("bloco fraco")

    def test_source_numbering_reflects_sorted_order_not_input_order(self):
        low = make_result({"diploma": "a", "text": "chunk-low"}, score=0.1)
        high = make_result({"diploma": "b", "text": "chunk-high"}, score=0.99)

        context = format_context([low, high])

        assert "[Fonte 1 | Diploma: b]\nchunk-high" in context
        assert "[Fonte 2 | Diploma: a]\nchunk-low" in context

    def test_blocks_separated_by_blank_line(self):
        r1 = make_result({"diploma": "a", "text": "primeiro"}, score=0.9)
        r2 = make_result({"diploma": "b", "text": "segundo"}, score=0.5)
        context = format_context([r1, r2])
        assert "\n\n" in context
        blocks = context.split("\n\n")
        assert len(blocks) == 2

    def test_missing_text_field_defaults_to_empty(self):
        result = make_result({"diploma": "dl_150_2015"}, score=0.5)
        context = format_context([result])
        assert context == "[Fonte 1 | Diploma: dl_150_2015]\n"

    def test_none_payload_does_not_crash(self):
        result = make_result(None, score=0.5)
        context = format_context([result])
        assert context == "[Fonte 1 | Diploma: None]\n"

    def test_respects_custom_max_chunk_chars(self):
        result = make_result(
            {"diploma": "dl_150_2015", "text": "x" * 50}, score=0.9
        )
        context = format_context([result], max_chunk_chars=10)
        # header line + truncated 10-char text
        text_line = context.split("\n", 1)[1]
        assert len(text_line) == 10
        assert text_line.endswith("…")

    def test_each_chunk_truncated_independently(self):
        long_a = make_result({"diploma": "a", "text": "a" * 2000}, score=0.9)
        long_b = make_result({"diploma": "b", "text": "b" * 2000}, score=0.8)
        context = format_context([long_a, long_b])
        for block in context.split("\n\n"):
            text_line = block.split("\n", 1)[1]
            assert len(text_line) == MAX_CHUNK_CHARS


# ---------------------------------------------------------------------------
# build_prompt
# ---------------------------------------------------------------------------


class TestBuildPrompt:
    def test_embeds_question(self):
        result = make_result(
            {"diploma": "dl_150_2015", "text": "algum texto"}, score=0.8
        )
        prompt = build_prompt("Quais são os deveres do operador?", [result])
        assert "Pergunta: Quais são os deveres do operador?" in prompt

    def test_embeds_context(self):
        result = make_result(
            {
                "diploma": "Decreto-Lei n.º 150/2015",
                "article_num": "8",
                "text": "algum texto",
            },
            score=0.8,
        )
        prompt = build_prompt("pergunta", [result])
        assert (
            "[Fonte 1 | Diploma: Decreto-Lei n.º 150/2015 | Artigo: 8.º]"
            in prompt
        )
        assert "algum texto" in prompt

    def test_matches_rag_prompt_template_structure(self):
        result = make_result(
            {"diploma": "dl_150_2015", "text": "texto"}, score=0.8
        )
        prompt = build_prompt("q", [result])
        expected = RAG_PROMPT.format(
            context=format_context([result]), question="q"
        )
        assert prompt == expected

    def test_no_results_still_produces_valid_prompt(self):
        prompt = build_prompt("pergunta sem correspondências", [])
        assert "Pergunta: pergunta sem correspondências" in prompt

    def test_instructions_and_fallback_phrase_present(self):
        prompt = build_prompt("q", [])
        assert '"O corpus não responde a esta pergunta."' in prompt
        assert "[Decreto-Lei n.º 150/2015, art. 8.º" in prompt

    def test_context_with_curly_braces_does_not_break_formatting(self):
        # Regulatory text containing literal braces (e.g. from a formula or
        # citation) must not be mis-parsed by str.format on the template.
        result = make_result(
            {"diploma": "dl_150_2015", "text": "Ver Anexo {III}."},
            score=0.8,
        )
        prompt = build_prompt("q", [result])
        assert "Ver Anexo {III}." in prompt
