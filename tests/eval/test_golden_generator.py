from unittest.mock import patch

import pytest

from pt_amb_rag.eval.golden_generator import (
    GoldenGenerationError,
    generate_candidates,
)

CHUNK = {
    "content_hash": "abc123",
    "regulation": "dl_150_2015",
    "eli": "https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html",
    "header": "Article 26",
    "text": "Deployers shall...",
}

VALID_JSON = (
    "[\n"
    '  {"question": "What must deployers do?", "difficulty": "easy",'
    ' "answerable": true},\n'
    '  {"question": "Who assigns human oversight?", "difficulty": "medium",'
    ' "answerable": true},\n'
    '  {"question": "What happens on an incident?", '
    '"difficulty": "hard", "answerable": true}\n'
    "]"
)


@patch("pt_amb_rag.eval.golden_generator.generate")
def test_parses_valid_json_on_first_try(mock_generate):
    mock_generate.return_value = VALID_JSON
    result = generate_candidates(CHUNK)
    assert len(result) == 3
    assert mock_generate.call_count == 1


@patch("pt_amb_rag.eval.golden_generator.generate")
def test_attaches_chunk_metadata_to_each_candidate(mock_generate):
    mock_generate.return_value = VALID_JSON
    result = generate_candidates(CHUNK)
    for c in result:
        assert c["content_hash"] == "abc123"
        assert c["regulation"] == "dl_150_2015"
        assert (
            c["eli"]
            == "https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html"
        )


@patch("pt_amb_rag.eval.golden_generator.generate")
def test_strips_markdown_fences(mock_generate):
    mock_generate.return_value = f"```json\n{VALID_JSON}\n```"
    result = generate_candidates(CHUNK)
    assert len(result) == 3


@patch("pt_amb_rag.eval.golden_generator.generate")
def test_retries_once_on_invalid_json_then_succeeds(mock_generate):
    mock_generate.side_effect = ["not json at all", VALID_JSON]
    result = generate_candidates(CHUNK)
    assert len(result) == 3
    assert mock_generate.call_count == 2
    second_call_prompt = mock_generate.call_args_list[1].args[0]
    assert "Lembrete" in second_call_prompt


@patch("pt_amb_rag.eval.golden_generator.generate")
def test_raises_after_second_failure(mock_generate):
    mock_generate.side_effect = ["nope", "still nope"]
    with pytest.raises(GoldenGenerationError):
        generate_candidates(CHUNK)
    assert mock_generate.call_count == 2


@patch("pt_amb_rag.eval.golden_generator.generate")
def test_passes_temperature_through(mock_generate):
    mock_generate.return_value = VALID_JSON
    generate_candidates(CHUNK, temperature=0.5)
    for call in mock_generate.call_args_list:
        assert call.kwargs.get("temperature") == 0.5
