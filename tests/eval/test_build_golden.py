import json
from unittest.mock import patch

import build_golden
import pytest

CHUNK_WITH_HASH = {
    "content_hash": "abc123",
    "regulation": "dl_150_2015",
    "eli": "https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html",
    "header": "Article 26",
    "text": "Deployers shall...",
}

CHUNK_NO_HASH = {
    "content_hash": None,
    "regulation": "dl_150_2015",
    "eli": "https://data.dre.pt/eli/dec-lei/150/2015/08/05/p/dre/pt/html",
    "header": "Article 1",
    "text": "Scope...",
}

FAKE_CANDIDATES = [
    {
        "question": "q1",
        "difficulty": "easy",
        "answerable": True,
        "content_hash": "abc123",
    },
    {
        "question": "q2",
        "difficulty": "medium",
        "answerable": True,
        "content_hash": "abc123",
    },
]


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data" / "processed"
    d.mkdir(parents=True)
    monkeypatch.setattr(build_golden, "DATA_DIR", d)
    return d


def write_jsonl(path, chunks):
    with path.open("w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")


def test_load_chunks_reads_jsonl(data_dir):
    write_jsonl(data_dir / "dl_150_2015.jsonl", [CHUNK_WITH_HASH])
    assert build_golden.load_chunks("dl_150_2015") == [CHUNK_WITH_HASH]


def test_load_chunks_missing_file_raises(data_dir):
    with pytest.raises(FileNotFoundError):
        build_golden.load_chunks("nonexistent")


def test_load_chunks_skips_blank_lines(data_dir):
    path = data_dir / "dl_150_2015.jsonl"
    path.write_text(json.dumps(CHUNK_WITH_HASH) + "\n\n")
    assert len(build_golden.load_chunks("dl_150_2015")) == 1


@patch("build_golden.generate_candidates")
def test_build_golden_writes_candidates(mock_gen, data_dir, tmp_path):
    write_jsonl(data_dir / "dl_150_2015.jsonl", [CHUNK_WITH_HASH])
    mock_gen.return_value = FAKE_CANDIDATES

    output = tmp_path / "out.jsonl"
    summary = build_golden.build_golden("dl_150_2015", output)

    assert len(output.read_text().splitlines()) == 2
    assert summary["candidates_written"] == 2
    assert summary["chunks_total"] == 1
    assert summary["chunks_skipped_no_hash"] == 0
    assert summary["chunks_failed"] == 0


@patch("build_golden.generate_candidates")
def test_build_golden_skips_chunks_without_content_hash(
    mock_gen, data_dir, tmp_path
):
    write_jsonl(
        data_dir / "dl_150_2015.jsonl", [CHUNK_NO_HASH, CHUNK_WITH_HASH]
    )
    mock_gen.return_value = FAKE_CANDIDATES

    output = tmp_path / "out.jsonl"
    summary = build_golden.build_golden("dl_150_2015", output)

    assert summary["chunks_skipped_no_hash"] == 1
    assert mock_gen.call_count == 1  # only called for the hashed chunk


@patch("build_golden.generate_candidates")
def test_build_golden_continues_after_generation_failure(
    mock_gen, data_dir, tmp_path
):
    from pt_amb_rag.eval.golden_generator import GoldenGenerationError

    other_chunk = {**CHUNK_WITH_HASH, "content_hash": "def456"}
    write_jsonl(data_dir / "dl_150_2015.jsonl", [CHUNK_WITH_HASH, other_chunk])
    mock_gen.side_effect = [GoldenGenerationError("boom"), FAKE_CANDIDATES]

    output = tmp_path / "out.jsonl"
    summary = build_golden.build_golden("dl_150_2015", output)

    assert summary["chunks_failed"] == 1
    assert summary["failed_hashes"] == ["abc123"]
    assert summary["candidates_written"] == 2  # only from the second chunk


@patch("build_golden.generate_candidates")
def test_build_golden_passes_temperature_through(mock_gen, data_dir, tmp_path):
    write_jsonl(data_dir / "dl_150_2015.jsonl", [CHUNK_WITH_HASH])
    mock_gen.return_value = FAKE_CANDIDATES

    output = tmp_path / "out.jsonl"
    build_golden.build_golden("dl_150_2015", output, temperature=0.7)

    _, kwargs = mock_gen.call_args
    assert kwargs["temperature"] == 0.7
