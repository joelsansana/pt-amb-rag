from __future__ import annotations

from pydantic import BaseModel, Field

UNANSWERABLE_CATEGORIES = {
    "national_transposition",
    "interplay_absent_instrument",
    "absent_standard",
    "lex_specialis_interplay",
    "absent_instrument",
    "post_enactment_content",
    "post_enactment_fact",
    "near_miss_locator",
    "future_amendment",
}


class CorpusSnapshot(BaseModel):
    """SHA-256 of every processed corpus file at the time the record was
    verified. Keys are `<corpus_id>_sha256`, one per diploma in the
    manifest."""

    model_config = {"extra": "allow"}


class UnanswerableQuestion(BaseModel):
    q_id: str
    tier: str = "unanswerable"
    category: str
    question: str = Field(min_length=1)
    unanswerable_because: str
    anchor_terms: dict[str, list[int]] = Field(default_factory=dict)
    must_be_zero: list[str] = Field(default_factory=list)
    nearest_in_corpus: str | None = None
    corpus_snapshot: CorpusSnapshot
    verified_on: str
    verification_method: str
