from __future__ import annotations

import os
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from pt_amb_rag.config import load_config
from pt_amb_rag.retrieval.embedder import embed

_CONFIG = load_config()

COLLECTION_NAME = _CONFIG.embedding.collection
VECTOR_SIZE = _CONFIG.embedding.dimension

PAYLOAD_FIELDS = (
    "regulation",
    "part",
    "article_num",
    "numero",
    "alinea",
    "annex_id",
    "chapter",
    "diploma",
    "diploma_type",
    "diploma_number",
    "published",
    "effective_date",
    "eli",
    "header",
    "text",
    "content_hash",
)


def get_client(url: str | None = None) -> QdrantClient:
    """Qdrant client. Defaults to QDRANT_URL env var, then localhost for dev.
    Pass url explicitly to override either."""
    url = url or os.environ.get("QDRANT_URL", "http://localhost:6333")
    return QdrantClient(url=url)


def ensure_collection(
    client: QdrantClient, collection_name: str = COLLECTION_NAME
) -> None:
    # reate the collection if it doesn't exist yet. Safe to call repeatedly.
    if not client.collection_exists(collection_name):
        client.create_collection(
            collection_name=collection_name,
            vectors_config=qmodels.VectorParams(
                size=VECTOR_SIZE, distance=qmodels.Distance.COSINE
            ),
        )


def _point_id(chunk: dict[str, Any]) -> str:
    """Deterministic point id from content_hash, so re-running upsert on the
    same chunk updates it in place instead of creating a duplicate point.
    Falls back to a random id if content_hash is missing."""
    content_hash = chunk.get("content_hash")
    if content_hash:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, content_hash))
    return str(uuid.uuid4())


def _payload(chunk: dict[str, Any]) -> dict[str, Any]:
    return {field: chunk.get(field) for field in PAYLOAD_FIELDS}


def upsert_chunks(
    chunks: list[dict[str, Any]],
    *,
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> int:
    """Embed chunk['text'] for each chunk and upsert into Qdrant.

    Each chunk is a dict expected to contain at least 'text', plus any of
    PAYLOAD_FIELDS. Returns the number of points upserted.
    """
    if not chunks:
        return 0

    client = client or get_client()
    ensure_collection(client, collection_name)

    texts = [chunk["text"] for chunk in chunks]
    vectors = embed(texts)

    points = [
        qmodels.PointStruct(
            id=_point_id(chunk), vector=vector, payload=_payload(chunk)
        )
        for chunk, vector in zip(chunks, vectors, strict=True)
    ]

    client.upsert(collection_name=collection_name, points=points)
    return len(points)


def _build_filter(
    regulation: str | None,
    article_num: int | None,
    part: str | None = None,
) -> qmodels.Filter | None:
    conditions: list[qmodels.Condition] = []

    if regulation is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="regulation", match=qmodels.MatchValue(value=regulation)
            )
        )
    if article_num is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="article_num", match=qmodels.MatchValue(value=article_num)
            )
        )
    if part is not None:
        conditions.append(
            qmodels.FieldCondition(
                key="part", match=qmodels.MatchValue(value=part)
            )
        )

    if not conditions:
        return None
    return qmodels.Filter(must=conditions)


def search(
    query_embedding: list[float],
    *,
    regulation: str | None = None,
    article_num: int | None = None,
    part: str | None = None,
    k: int = 5,
    client: QdrantClient | None = None,
    collection_name: str = COLLECTION_NAME,
) -> list[qmodels.ScoredPoint]:
    """Cosine-similarity search, optionally filtered by regulation and/or
    article_num. Returns up to k ScoredPoint results, highest score first."""
    client = client or get_client()
    query_filter = _build_filter(regulation, article_num, part)

    response = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        query_filter=query_filter,
        limit=k,
    )
    return response.points
