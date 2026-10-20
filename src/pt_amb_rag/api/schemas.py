from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    regulation: str | None = None  # corpus id, e.g. "dl_150_2015"
    article_num: int | None = None
    part: str | None = None  # "preamble" | "article" | "annex" | "chapter"


class SourceChunk(BaseModel):
    score: float
    regulation: str
    part: str
    article_num: int | None = None
    numero: int | None = None
    alinea: str | None = None
    annex_id: str | None = None
    chapter: str | None = None
    diploma: str
    diploma_type: str | None = None
    diploma_number: str | None = None
    published: str | None = None
    effective_date: str | None = None
    eli: str
    header: str
    text: str
    content_hash: str


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: int


class SearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20)
    regulation: str | None = None
    article_num: int | None = None
    part: str | None = None
