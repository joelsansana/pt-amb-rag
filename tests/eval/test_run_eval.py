import json
import sys
from unittest.mock import patch

import pytest
import run_eval

GOLDEN_RECORD = {
    "q_id": "q001",
    "tier": "principal",
    "question": "What must deployers do under Article 26?",
    "gold_ids": ["hash_a"],
    "all_acceptable_ids": ["hash_a", "hash_b"],
    "difficulty": "medium",
}


def write_golden(tmp_path, records):
    path = tmp_path / "principal.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


@patch("run_eval.ask")
def test_perfect_hit(mock_ask, tmp_path):
    mock_ask.return_value = {
        "sources": [{"content_hash": "hash_a"}]
        + [{"content_hash": f"x{i}"} for i in range(9)]
    }
    path = write_golden(tmp_path, [GOLDEN_RECORD])

    result = run_eval.run_retrieval_eval(path)

    assert result["hit_at_5"] == 1.0
    assert result["mrr_at_10"] == 1.0
    # only hash_a of {hash_a, hash_b} retrieved
    assert result["recall_at_10"] == 0.5
    assert result["n_questions"] == 1
    assert result["n_failed"] == 0


@patch("run_eval.ask")
def test_miss(mock_ask, tmp_path):
    mock_ask.return_value = {
        "sources": [{"content_hash": f"x{i}"} for i in range(10)]
    }
    path = write_golden(tmp_path, [GOLDEN_RECORD])

    result = run_eval.run_retrieval_eval(path)

    assert result["hit_at_5"] == 0.0
    assert result["mrr_at_10"] == 0.0
    assert result["recall_at_10"] == 0.0


@patch("run_eval.ask")
def test_always_retrieves_at_k_rank_not_k_hit(mock_ask, tmp_path):
    mock_ask.return_value = {"sources": []}
    path = write_golden(tmp_path, [GOLDEN_RECORD])

    run_eval.run_retrieval_eval(path, k_hit=5, k_rank=10)

    _, kwargs = mock_ask.call_args
    assert kwargs["k"] == 10  # the k=5-vs-k=10 bug this design avoids


@patch("run_eval.ask")
def test_calls_ask_unfiltered(mock_ask, tmp_path):
    mock_ask.return_value = {"sources": []}
    path = write_golden(tmp_path, [GOLDEN_RECORD])

    run_eval.run_retrieval_eval(path)

    args, kwargs = mock_ask.call_args
    assert args[0] == GOLDEN_RECORD["question"]
    assert "regulation" not in kwargs
    assert "part" not in kwargs
    assert "article_num" not in kwargs


@patch("run_eval.ask")
def test_continues_after_one_failure(mock_ask, tmp_path):
    other = {**GOLDEN_RECORD, "q_id": "q002", "question": "second question"}
    path = write_golden(tmp_path, [GOLDEN_RECORD, other])

    mock_ask.side_effect = [
        RuntimeError("boom"),
        {"sources": [{"content_hash": "hash_a"}]},
    ]

    result = run_eval.run_retrieval_eval(path)

    assert result["n_failed"] == 1
    assert result["failed_q_ids"] == ["q001"]
    assert result["n_questions"] == 1


@patch("run_eval.ask")
def test_empty_golden_file_returns_zeros_without_calling_ask(
    mock_ask, tmp_path
):
    path = tmp_path / "empty.jsonl"
    path.write_text("")

    result = run_eval.run_retrieval_eval(path)

    assert result["hit_at_5"] == 0.0
    assert result["n_questions"] == 0
    mock_ask.assert_not_called()


@patch("run_eval.ask")
def test_per_question_results_recorded(mock_ask, tmp_path):
    mock_ask.return_value = {"sources": [{"content_hash": "hash_a"}]}
    path = write_golden(tmp_path, [GOLDEN_RECORD])

    result = run_eval.run_retrieval_eval(path)

    assert len(result["per_question"]) == 1
    assert result["per_question"][0]["q_id"] == "q001"
    assert result["per_question"][0]["hit"] == 1


def test_main_raises_for_missing_tier_file(tmp_path, monkeypatch):
    monkeypatch.setattr(run_eval, "GOLDEN_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_eval.py", "--tier", "principal"])
    with pytest.raises(FileNotFoundError):
        run_eval.main()
