"""Configuration for the pt-amb-rag fork.

Single source of truth for the four axes that were hardcoded in the
upstream `eu-ai-rag`: the corpus source, the language, the embedder, and
the instrument naming. Everything is read from `corpora.yaml` at the repo
root so that adding a diploma is a manifest edit, not a code change.

See `UPSTREAM.md` for the cherry-pick contract against the upstream repo.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "corpora.yaml"

# Env var used by CI and by the container to point at an alternative
# manifest (e.g. a pinned corpus snapshot in a release branch).
MANIFEST_ENV_VAR = "PT_AMB_RAG_MANIFEST"


class ConfigError(Exception):
    """Raised when the manifest is missing, malformed, or inconsistent."""


@dataclass(frozen=True)
class Diploma:
    """One legal instrument in the corpus.

    `tipo` and `key` are the two identifiers the DRE screenservice needs
    to address a document: `tipo` is the URL segment (`dec-lei`, `lei`,
    `portaria`) and `key` is the DRE content key (e.g. `150-2015-69951097`,
    the trailing number of a `/dr/detalhe/<tipo>/<key>` URL).
    """

    id: str
    title: str
    family: str
    number: str
    published: str
    tipo: str
    key: str
    eli: str = ""
    pdf_url: str = ""
    effective_date: str = ""
    tags: tuple[str, ...] = ()

    @property
    def short_ref(self) -> str:
        """Compact citation form, e.g. `DL n.º 150/2015`."""
        return f"{self.family} n.º {self.number}"

    def as_metadata(self) -> dict[str, Any]:
        """Metadata attached to every chunk derived from this diploma."""
        return {
            "regulation": self.id,
            "diploma": self.title,
            "diploma_type": self.family,
            "diploma_number": self.number,
            "published": self.published,
            "effective_date": self.effective_date or self.published,
            "eli": self.eli,
        }


@dataclass(frozen=True)
class LanguageProfile:
    """Everything that changes when the corpus language changes.

    Kept here (rather than hardcoded in `prompts.py`) so that the
    abstention sentinel, the month names used in chunk headers, and the
    judge rubric can never drift apart: they all read from this object.
    """

    code: str
    month_names: tuple[str, ...]
    abstention: str
    citation_hint: str

    def format_date(self, value: str) -> str:
        """Render an ISO date the way the corpus language writes it."""
        year, month, day = value.split("-")
        name = self.month_names[int(month) - 1]
        return f"{int(day)} de {name} de {year}"


@dataclass(frozen=True)
class EmbeddingProfile:
    """The embedder and the vector store collection it feeds.

    `dimension` is asserted against the model's own metadata at load
    time — a silent 384-vs-1024 mismatch between the model and the
    Qdrant collection is the obvious failure mode of swapping an
    English-only encoder for a multilingual one.
    """

    model: str
    dimension: int
    collection: str


@dataclass(frozen=True)
class Config:
    language: LanguageProfile
    embedding: EmbeddingProfile
    diplomas: tuple[Diploma, ...]
    tiers: tuple[str, ...] = field(default=())

    @property
    def diploma_ids(self) -> tuple[str, ...]:
        return tuple(d.id for d in self.diplomas)

    def diploma(self, diploma_id: str) -> Diploma:
        for diploma in self.diplomas:
            if diploma.id == diploma_id:
                return diploma
        raise ConfigError(
            f"unknown diploma {diploma_id!r}; "
            f"known: {', '.join(self.diploma_ids)}"
        )

    def diplomas_with_tag(self, tag: str) -> tuple[Diploma, ...]:
        return tuple(d for d in self.diplomas if tag in d.tags)


def _require(mapping: dict[str, Any], key: str, where: str) -> Any:
    if key not in mapping:
        raise ConfigError(f"missing {key!r} in {where}")
    return mapping[key]


def _parse_diploma(raw: dict[str, Any], index: int) -> Diploma:
    where = f"corpora[{index}]"
    if not isinstance(raw, dict):
        raise ConfigError(f"{where} must be a mapping")

    source = raw.get("source") or {}
    if not isinstance(source, dict):
        raise ConfigError(f"{where}.source must be a mapping")
    if source.get("type") != "dre":
        raise ConfigError(
            f"{where}.source.type must be 'dre' (the only implemented "
            f"source type), got {source.get('type')!r}"
        )

    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        raise ConfigError(f"{where}.tags must be a list")

    return Diploma(
        id=str(_require(raw, "id", where)),
        title=str(_require(raw, "title", where)),
        family=str(_require(raw, "family", where)),
        number=str(_require(raw, "number", where)),
        published=str(_require(raw, "published", where)),
        tipo=str(_require(source, "tipo", f"{where}.source")),
        key=str(_require(source, "key", f"{where}.source")),
        eli=str(source.get("eli") or ""),
        pdf_url=str(source.get("pdf_url") or ""),
        effective_date=str(raw.get("effective_date") or ""),
        tags=tuple(str(t) for t in tags),
    )


def _parse_language(raw: dict[str, Any]) -> LanguageProfile:
    where = "language"
    month_names = _require(raw, "month_names", where)
    if not isinstance(month_names, list) or len(month_names) != 12:
        raise ConfigError(
            f"{where}.month_names must be a list of exactly 12 names"
        )
    return LanguageProfile(
        code=str(_require(raw, "code", where)),
        month_names=tuple(str(m) for m in month_names),
        abstention=str(_require(raw, "abstention", where)),
        citation_hint=str(_require(raw, "citation_hint", where)),
    )


def parse_config(raw: dict[str, Any]) -> Config:
    """Build a `Config` from a parsed manifest mapping."""
    if not isinstance(raw, dict):
        raise ConfigError("manifest must be a mapping")

    language = _parse_language(raw.get("language") or {})

    embedding_raw = raw.get("embedding") or {}
    if not isinstance(embedding_raw, dict):
        raise ConfigError("embedding must be a mapping")
    embedding = EmbeddingProfile(
        model=str(_require(embedding_raw, "model", "embedding")),
        dimension=int(_require(embedding_raw, "dimension", "embedding")),
        collection=str(_require(embedding_raw, "collection", "embedding")),
    )

    corpus_raw = raw.get("corpora") or []
    if not isinstance(corpus_raw, list):
        raise ConfigError("corpora must be a list")
    diplomas = tuple(
        _parse_diploma(item, i) for i, item in enumerate(corpus_raw)
    )
    if not diplomas:
        raise ConfigError("corpora is empty — the fork needs a pilot diploma")

    seen: set[str] = set()
    for diploma in diplomas:
        if diploma.id in seen:
            raise ConfigError(f"duplicate corpus id {diploma.id!r}")
        seen.add(diploma.id)

    tiers_raw = raw.get("eval_tiers") or []
    if not isinstance(tiers_raw, list):
        raise ConfigError("eval_tiers must be a list")

    return Config(
        language=language,
        embedding=embedding,
        diplomas=diplomas,
        tiers=tuple(str(t) for t in tiers_raw),
    )


def manifest_path() -> Path:
    """Path of the manifest to load, honouring the env override."""
    override = os.environ.get(MANIFEST_ENV_VAR)
    if override:
        return Path(override)
    return DEFAULT_MANIFEST_PATH


def load_config(path: Path | None = None) -> Config:
    """Load and validate the corpus manifest."""
    manifest = path or manifest_path()
    if not manifest.is_file():
        raise ConfigError(f"corpus manifest not found: {manifest}")

    raw = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    return parse_config(raw or {})
