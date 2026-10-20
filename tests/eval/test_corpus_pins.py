import hashlib

from pt_amb_rag.eval.corpus_pins import (
    check_record_pins,
    count_term_hits,
    file_sha256,
)


def test_file_sha256_matches_hashlib(tmp_path):
    p = tmp_path / "f.txt"
    p.write_text("hello world")
    assert file_sha256(p) == hashlib.sha256(b"hello world").hexdigest()


def test_count_term_hits_case_insensitive_by_default(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text('{"text": "menciona o REI aqui"}\n{"text": "sem match"}\n')
    assert count_term_hits(p, "rei") == 1


def test_count_term_hits_exact_case_suffix(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text(
        '{"text": "APA existe assim"}\n{"text": "apa existe em minúsculas"}\n'
    )
    assert count_term_hits(p, "APA (exact case)") == 1
    assert count_term_hits(p, "APA") == 2


def test_count_term_hits_counts_lines_not_occurrences(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text(
        '{"text": "SEVESO SEVESO SEVESO numa linha"}\n'
        '{"text": "SEVESO outra vez"}\n'
    )
    # 2 matching lines, not 4 occurrences
    assert count_term_hits(p, "SEVESO") == 2


def test_check_record_pins_detects_stale_hash(tmp_path):
    rei = tmp_path / "dl_127_2013.jsonl"
    seveso = tmp_path / "dl_150_2015.jsonl"
    rei.write_text('{"text": "x"}\n')
    seveso.write_text('{"text": "y"}\n')

    record = {
        "q_id": "u001",
        "corpus_snapshot": {
            "dl_127_2013_sha256": "wrong",
            "dl_150_2015_sha256": "wrong",
        },
        "anchor_terms": {},
        "must_be_zero": [],
    }
    problems = check_record_pins(
        record, {"dl_127_2013": rei, "dl_150_2015": seveso}
    )
    assert any("corpus_snapshot" in p for p in problems)


def test_check_record_pins_clean_when_everything_matches(tmp_path):
    rei = tmp_path / "dl_127_2013.jsonl"
    seveso = tmp_path / "dl_150_2015.jsonl"
    rei.write_text('{"text": "no relevant terms"}\n')
    seveso.write_text('{"text": "also none"}\n')

    record = {
        "q_id": "u001",
        "corpus_snapshot": {
            "dl_127_2013_sha256": file_sha256(rei),
            "dl_150_2015_sha256": file_sha256(seveso),
        },
        "anchor_terms": {"Espanha": [0, 0]},
        "must_be_zero": ["Espanha"],
    }
    assert (
        check_record_pins(record, {"dl_127_2013": rei, "dl_150_2015": seveso})
        == []
    )


def test_check_record_pins_detects_must_be_zero_violation(tmp_path):
    rei = tmp_path / "dl_127_2013.jsonl"
    seveso = tmp_path / "dl_150_2015.jsonl"
    rei.write_text('{"text": "Espanha mentioned here now"}\n')
    seveso.write_text('{"text": "none"}\n')

    record = {
        "q_id": "u001",
        "corpus_snapshot": {
            "dl_127_2013_sha256": file_sha256(rei),
            "dl_150_2015_sha256": file_sha256(seveso),
        },
        "anchor_terms": {},
        "must_be_zero": ["Espanha"],
    }
    problems = check_record_pins(
        record, {"dl_127_2013": rei, "dl_150_2015": seveso}
    )
    assert any("must_be_zero" in p for p in problems)


def test_check_record_pins_generalises_to_any_number_of_diplomas(tmp_path):
    paths = {}
    snapshot = {}
    for i in range(5):
        path = tmp_path / f"dl_{i}.jsonl"
        path.write_text(f'{{"text": "corpo {i}"}}\n')
        paths[f"dl_{i}"] = path
        snapshot[f"dl_{i}_sha256"] = file_sha256(path)

    record = {
        "q_id": "u001",
        "corpus_snapshot": snapshot,
        "anchor_terms": {"corpo": [1, 1, 1, 1, 1]},
        "must_be_zero": [],
    }
    assert check_record_pins(record, paths) == []


def test_check_record_pins_detects_a_new_diploma_in_the_corpus(tmp_path):
    """Adding a diploma to the manifest invalidates the recorded
    snapshot — the pins must fail loudly rather than silently accept
    the new corpus."""
    rei = tmp_path / "dl_127_2013.jsonl"
    rei.write_text('{"text": "x"}\n')

    record = {
        "q_id": "u001",
        "corpus_snapshot": {"dl_127_2013_sha256": file_sha256(rei)},
        "anchor_terms": {},
        "must_be_zero": [],
    }
    problems = check_record_pins(
        record, {"dl_127_2013": rei, "dl_150_2015": rei}
    )
    assert any("dl_150_2015_sha256" in p for p in problems)
