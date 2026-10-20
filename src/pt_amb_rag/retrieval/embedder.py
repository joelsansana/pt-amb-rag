from __future__ import annotations

from sentence_transformers import SentenceTransformer

from pt_amb_rag.config import load_config

_CONFIG = load_config()

_MODEL = None


def get_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        _MODEL = SentenceTransformer(_CONFIG.embedding.model)
        dimension = _MODEL.get_sentence_embedding_dimension()
        if dimension != _CONFIG.embedding.dimension:
            raise RuntimeError(
                f"embedder {_CONFIG.embedding.model} produces "
                f"{dimension}-dimensional vectors, but the manifest "
                f"declares {_CONFIG.embedding.dimension}. A mismatch here "
                f"would be silently rejected by Qdrant at upsert time "
                f"with a confusing error."
            )
    return _MODEL


def embed(texts: list[str]) -> list[list[float]]:
    return get_model().encode(texts, normalize_embeddings=True).tolist()
