from __future__ import annotations

import hashlib
import re
from typing import Any

# Annexes above this approximate token count are split into smaller chunks.
# This is deliberately conservative because Annexes can contain large
# tables.
LONG_ANNEX_TOKEN_THRESHOLD = 800


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Estimate token count deterministically without a tokenizer dependency.

    This is intentionally simple. The value is useful for chunk sizing and
    metadata, but it is not intended to exactly reproduce a model tokenizer.
    """
    if not text.strip():
        return 0

    return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE))


def _make_header(unit: dict[str, Any], position: int | None = None) -> str:
    """Build the contextual header that prefixes every chunk.

    The header is the chunk's citation identity: it names the diploma,
    the date it was published, and the position inside the instrument.
    It is prepended to the body rather than stored beside it, so that
    both the dense embedder and a lexical index see it.
    """
    diploma = unit.get("diploma") or unit.get("regulation") or ""
    published = unit.get("published") or unit.get("effective_date") or ""
    date_text = _format_pt_date(published) if published else ""

    part = unit.get("part") or ""
    chapter = unit.get("chapter")
    title = unit.get("title") or ""
    annex_id = unit.get("annex_id")
    article_num = unit.get("article_num")
    numero = unit.get("numero")
    alinea = unit.get("alinea")

    if part == "article":
        locator = f"Artigo {article_num}.º"
        if numero:
            locator += f", n.º {numero}"
        if alinea:
            locator += f", al. {alinea})"
        header = f"{locator} — {diploma}"
        if date_text:
            header += f", de {date_text}"
        if chapter:
            header += f" — {chapter}"
        if title:
            header += f" — {title}"
    elif part == "annex":
        header = f"Anexo {annex_id} — {diploma}"
        if date_text:
            header += f", de {date_text}"
        if title:
            header += f" — {title}"
    elif part == "chapter":
        header = f"{chapter} — {diploma}"
        if date_text:
            header += f", de {date_text}"
        if title:
            header += f" — {title}"
    else:
        header = f"{_part_label(part)} — {diploma}"
        if date_text:
            header += f", de {date_text}"
        if title:
            header += f" — {title}"

    if position is not None:
        header += f" — Parte {position}"

    return header


def _part_label(part: str) -> str:
    labels = {
        "preamble": "Preâmbulo",
        "section": "Disposição",
        "article": "Artigo",
        "annex": "Anexo",
        "chapter": "Capítulo",
    }
    return labels.get(part, part.title())


def _format_pt_date(value: str) -> str:
    """Render an ISO date the way Portuguese legislation writes it."""
    months = (
        "janeiro",
        "fevereiro",
        "março",
        "abril",
        "maio",
        "junho",
        "julho",
        "agosto",
        "setembro",
        "outubro",
        "novembro",
        "dezembro",
    )
    try:
        year, month, day = value.split("-")
    except ValueError:
        return value
    return f"{int(day)} de {months[int(month) - 1]} de {year}"


def _make_chunk(
    unit: dict[str, Any],
    body: str,
    position: int,
) -> dict[str, Any]:
    header = _make_header(unit, position if position > 1 else None)
    text = f"{header}\n\n{body.strip()}"

    chunk_key = f"{unit.get('regulation')}{unit.get('article_num')}{position}"

    content_hash = _sha256(header + text)

    return {
        "chunk_id": _sha256(chunk_key)[:12],
        "regulation": unit.get("regulation"),
        "part": unit.get("part"),
        "article_num": unit.get("article_num"),
        "numero": unit.get("numero"),
        "alinea": unit.get("alinea"),
        "annex_id": unit.get("annex_id"),
        "chapter": unit.get("chapter"),
        "diploma": unit.get("diploma"),
        "diploma_type": unit.get("diploma_type"),
        "diploma_number": unit.get("diploma_number"),
        "published": unit.get("published"),
        "effective_date": unit.get("effective_date"),
        "eli": unit.get("eli"),
        "header": header,
        "text": text,
        "n_tokens": _estimate_tokens(text),
        "content_hash": content_hash,
    }


def _split_annex_rows(text: str) -> list[str]:
    """Split an Annex into its existing paragraph/row-like units.

    The DRE's `TextoFormatado` is made of `<p>` elements and carries no
    table markup, so the blank-line boundaries of the extracted text
    are the closest available approximation of table rows. This is the
    same shape as the upstream EUR-Lex splitter; the difference is in
    what fills it.
    """
    return [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]


def _chunk_long_annex(
    unit: dict[str, Any],
) -> list[dict[str, Any]]:
    rows = _split_annex_rows(unit["text"])

    if not rows:
        return []

    chunks: list[dict[str, Any]] = []
    current_rows: list[str] = []
    position = 1

    for row in rows:
        candidate_rows = [*current_rows, row]
        candidate_body = "\n\n".join(candidate_rows)

        # Estimate using the eventual header as well. This prevents chunks
        # from exceeding the approximate target simply because of the header.
        candidate_header = _make_header(
            unit, position if position > 1 else None
        )
        candidate_text = f"{candidate_header}\n\n{candidate_body}"

        if (
            current_rows
            and _estimate_tokens(candidate_text) > LONG_ANNEX_TOKEN_THRESHOLD
        ):
            chunks.append(
                _make_chunk(
                    unit,
                    "\n\n".join(current_rows),
                    position,
                )
            )
            position += 1
            current_rows = [row]
        else:
            current_rows.append(row)

    if current_rows:
        chunks.append(
            _make_chunk(
                unit,
                "\n\n".join(current_rows),
                position,
            )
        )

    return chunks


def chunk_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Convert parsed DRE units into retrieval chunks.

    Articles and other normal structural units remain intact. Long Annexes
    are split on their existing paragraph/row boundaries.
    """
    chunks: list[dict[str, Any]] = []

    for unit in records:
        if not unit["text"].strip():
            continue

        if unit["part"] == "annex":
            token_count = _estimate_tokens(unit["text"])

            if token_count > LONG_ANNEX_TOKEN_THRESHOLD:
                chunks.extend(_chunk_long_annex(unit))
                continue

        chunks.append(
            _make_chunk(
                unit,
                unit["text"],
                1,
            )
        )

    return chunks
