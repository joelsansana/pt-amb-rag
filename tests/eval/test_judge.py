from unittest.mock import patch

import pytest

from pt_amb_rag.eval.judge import JudgeError, judge

QUESTION = "What must deployers of high-risk AI systems do for their workers?"
ANSWER = (
    "Deployers must inform workers' representatives before putting a "
    "high-risk system into service at the workplace (Article 26)."
)

SOURCES = [
    {
        "header": "Article 26, Regulation (EU) 2024/1689 (AI Act)",
        "text": (
            "Deployers of high-risk AI systems shall inform workers' "
            "representatives..."
        ),
    },
    {
        "header": "Recital 133, Regulation (EU) 2024/1689 (AI Act)",
        "text": "Workers should be informed...",
    },
]

VALID_ANSWERABLE = """{
  "faithfulness": "yes",
  "citation_accuracy": "yes",
  "evidence": "Both claims appear verbatim in source [1]."
}"""

VALID_ABSTENTION = (
    "{\n"
    '  "abstention": "yes",\n'
    '  "evidence": "The answer states the corpus does not address '
    'national procedure."\n'
    "}"
)


@patch("pt_amb_rag.eval.judge.generate")
def test_parses_valid_verdict_on_first_try(mock_generate):
    mock_generate.return_value = VALID_ANSWERABLE
    verdict = judge(QUESTION, SOURCES, ANSWER)
    assert verdict["faithfulness"] == "yes"
    assert verdict["citation_accuracy"] == "yes"
    assert mock_generate.call_count == 1


@patch("pt_amb_rag.eval.judge.generate")
def test_prompt_carries_question_answer_and_numbered_sources(mock_generate):
    mock_generate.return_value = VALID_ANSWERABLE
    judge(QUESTION, SOURCES, ANSWER)
    prompt = mock_generate.call_args_list[0].args[0]
    assert QUESTION in prompt
    assert ANSWER in prompt
    assert "[1] Article 26, Regulation (EU) 2024/1689 (AI Act)" in prompt
    assert "[2] Recital 133" in prompt


@patch("pt_amb_rag.eval.judge.generate")
def test_answerable_mode_does_not_ask_for_abstention(mock_generate):
    mock_generate.return_value = VALID_ANSWERABLE
    judge(QUESTION, SOURCES, ANSWER, expect_abstention=False)
    prompt = mock_generate.call_args_list[0].args[0]
    assert "faithfulness" in prompt
    assert "abstention" not in prompt


@patch("pt_amb_rag.eval.judge.generate")
def test_abstention_mode_swaps_criteria_and_validates(mock_generate):
    mock_generate.return_value = VALID_ABSTENTION
    verdict = judge(QUESTION, SOURCES, ANSWER, expect_abstention=True)
    assert verdict["abstention"] == "yes"
    prompt = mock_generate.call_args_list[0].args[0]
    assert "abstention" in prompt
    assert "faithfulness" not in prompt


@patch("pt_amb_rag.eval.judge.generate")
def test_strips_markdown_fences(mock_generate):
    mock_generate.return_value = f"```json\n{VALID_ANSWERABLE}\n```"
    verdict = judge(QUESTION, SOURCES, ANSWER)
    assert verdict["faithfulness"] == "yes"


@patch("pt_amb_rag.eval.judge.generate")
def test_retries_once_on_invalid_json_then_succeeds(mock_generate):
    mock_generate.side_effect = ["not json at all", VALID_ANSWERABLE]
    verdict = judge(QUESTION, SOURCES, ANSWER)
    assert verdict["faithfulness"] == "yes"
    assert mock_generate.call_count == 2
    second_call_prompt = mock_generate.call_args_list[1].args[0]
    assert "Lembrete" in second_call_prompt


@patch("pt_amb_rag.eval.judge.generate")
def test_raises_after_second_failure(mock_generate):
    mock_generate.side_effect = ["nope", "still nope"]
    with pytest.raises(JudgeError):
        judge(QUESTION, SOURCES, ANSWER)
    assert mock_generate.call_count == 2


@patch("pt_amb_rag.eval.judge.generate")
def test_rejects_verdict_value_outside_allowed_set(mock_generate):
    # "maybe" is verdict drift, not a verdict — must fail validation,
    # retry, and raise rather than be accepted.
    drifted = (
        '{"faithfulness": "maybe", "citation_accuracy": "yes", '
        '"evidence": "..."}'
    )
    mock_generate.side_effect = [drifted, drifted]
    with pytest.raises(JudgeError):
        judge(QUESTION, SOURCES, ANSWER)
    assert mock_generate.call_count == 2


@patch("pt_amb_rag.eval.judge.generate")
def test_rejects_wrong_schema_for_mode(mock_generate):
    # An abstention-only verdict is invalid for answerable mode...
    mock_generate.side_effect = [VALID_ABSTENTION, VALID_ABSTENTION]
    with pytest.raises(JudgeError):
        judge(QUESTION, SOURCES, ANSWER, expect_abstention=False)
    # ...and an answerable verdict is invalid for abstention mode.
    mock_generate.reset_mock()
    mock_generate.side_effect = [VALID_ANSWERABLE, VALID_ANSWERABLE]
    with pytest.raises(JudgeError):
        judge(QUESTION, SOURCES, ANSWER, expect_abstention=True)


@patch("pt_amb_rag.eval.judge.generate")
def test_passes_temperature_through(mock_generate):
    mock_generate.return_value = VALID_ANSWERABLE
    judge(QUESTION, SOURCES, ANSWER, temperature=0.5)
    for call in mock_generate.call_args_list:
        assert call.kwargs.get("temperature") == 0.5
