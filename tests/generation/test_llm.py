import re
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from pt_amb_rag.generation.llm import (
    MODEL,
    _strip_thinking,
    generate,
    get_client,
)


@pytest.mark.integration
def test_generate_pong():
    output = generate("Reply with one word: pong.")

    assert output is not None, "generate() returned None"
    assert isinstance(output, str), f"Expected str, got {type(output)}"

    # Normalize: lowercase, strip whitespace, drop surrounding punctuation
    normalized = re.sub(r"[^\w\s]", "", output).strip().lower()

    assert normalized == "pong", f"Expected 'pong', got: {output!r}"


# ---------------------------------------------------------------------------
# _strip_thinking — pure function, no mocking needed
# ---------------------------------------------------------------------------


class TestStripThinking:
    def test_no_think_tag_passthrough(self):
        assert _strip_thinking("pong") == "pong"

    def test_strips_single_think_block(self):
        text = "<think>reasoning here</think>pong"
        assert _strip_thinking(text) == "pong"

    def test_strips_think_block_with_surrounding_whitespace(self):
        text = "  <think>reasoning</think>  pong  "
        assert _strip_thinking(text) == "pong"

    def test_strips_multiline_think_block(self):
        text = "<think>\nstep 1\nstep 2\n</think>\npong"
        assert _strip_thinking(text) == "pong"

    def test_strips_multiple_think_blocks(self):
        text = "<think>a</think>pong<think>b</think>"
        assert _strip_thinking(text) == "pong"

    def test_empty_string(self):
        assert _strip_thinking("") == ""

    def test_only_think_block_returns_empty(self):
        assert _strip_thinking("<think>only reasoning, no answer</think>") == ""

    def test_unclosed_think_tag_not_stripped(self):
        # No closing tag > regex shouldn't match, text passes through (stripped)
        text = "<think>never closes pong"
        assert _strip_thinking(text) == text.strip()

    def test_text_before_and_after_think_block(self):
        text = "prefix <think>hidden</think> suffix"
        assert _strip_thinking(text) == "prefix  suffix"


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_uses_env_api_key_and_base_url(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key-123")
        with patch("pt_amb_rag.generation.llm.OpenAI") as mock_openai:
            get_client()
            mock_openai.assert_called_once_with(
                base_url="https://api.minimax.io/v1",
                api_key="test-key-123",
            )

    def test_missing_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
        with pytest.raises(KeyError):
            get_client()


# ---------------------------------------------------------------------------
# generate — mock the OpenAI client so no network call is made
# ---------------------------------------------------------------------------


def _make_mock_client(content: str | None):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=MagicMock(content=content))]
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client


class TestGenerate:
    def test_returns_stripped_content(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        with patch(
            "pt_amb_rag.generation.llm.get_client",
            return_value=_make_mock_client("pong"),
        ):
            assert generate("Reply with one word: pong.") == "pong"

    def test_strips_think_tags_from_response(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        with patch(
            "pt_amb_rag.generation.llm.get_client",
            return_value=_make_mock_client(
                "<think>let me consider</think>pong"
            ),
        ):
            assert generate("ping") == "pong"

    def test_none_content_returns_empty_string(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        with patch(
            "pt_amb_rag.generation.llm.get_client",
            return_value=_make_mock_client(None),
        ):
            assert generate("ping") == ""

    def test_calls_model_with_expected_params(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_client = _make_mock_client("pong")
        with patch(
            "pt_amb_rag.generation.llm.get_client", return_value=mock_client
        ):
            generate("hello")
            mock_client.chat.completions.create.assert_called_once_with(
                model=MODEL,
                messages=[{"role": "user", "content": "hello"}],
            )

    def test_passes_prompt_through_unmodified(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        mock_client = _make_mock_client("pong")
        with patch(
            "pt_amb_rag.generation.llm.get_client", return_value=mock_client
        ):
            generate("a very specific prompt")
            _, kwargs = mock_client.chat.completions.create.call_args
            assert kwargs["messages"][0]["content"] == "a very specific prompt"

    def test_passes_temperature_when_given(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        captured = {}

        class FakeMessage:
            content = "ok"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices: ClassVar[list[FakeChoice]] = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        with patch(
            "pt_amb_rag.generation.llm.get_client",
            return_value=FakeClient(),
        ):
            generate("hi", temperature=0.2)

        assert captured["temperature"] == 0.2

    def test_omits_temperature_when_not_given(self, monkeypatch):
        monkeypatch.setenv("MINIMAX_API_KEY", "test-key")
        captured = {}

        class FakeMessage:
            content = "ok"

        class FakeChoice:
            message = FakeMessage()

        class FakeResponse:
            choices: ClassVar[list[FakeChoice]] = [FakeChoice()]

        class FakeCompletions:
            def create(self, **kwargs):
                captured.update(kwargs)
                return FakeResponse()

        class FakeChat:
            completions = FakeCompletions()

        class FakeClient:
            chat = FakeChat()

        with patch(
            "pt_amb_rag.generation.llm.get_client",
            return_value=FakeClient(),
        ):
            generate("hi")

        assert "temperature" not in captured
