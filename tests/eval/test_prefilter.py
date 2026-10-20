from pt_amb_rag.eval.prefilter import is_junk_candidate, prefilter


def make(q: str, **overrides) -> dict:
    base = {
        "question": q,
        "difficulty": "easy",
        "answerable": True,
        "content_hash": "h1",
    }
    base.update(overrides)
    return base


def test_too_short_is_junk():
    assert is_junk_candidate(make("Why?"), set())


def test_missing_question_mark_is_junk():
    assert is_junk_candidate(
        make("What deployers must do under Article 26"), set()
    )


def test_exact_duplicate_is_junk():
    assert is_junk_candidate(
        make("What must deployers do?"), {"what must deployers do?"}
    )


def test_good_question_is_not_junk():
    assert not is_junk_candidate(
        make("What must deployers do under Article 26?"), set()
    )


def test_prefilter_drops_junk_and_keeps_good():
    candidates = [
        make("What must deployers do under Article 26?"),
        make("Why?"),
        make("What must deployers do under Article 26?"),  # dup
        make("Who must assign human oversight under the AI Act?"),
    ]
    kept, dropped = prefilter(candidates)
    assert len(kept) == 2
    assert dropped == 2


def test_prefilter_empty_list():
    kept, dropped = prefilter([])
    assert kept == []
    assert dropped == 0
