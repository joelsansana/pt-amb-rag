from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import pt_amb_rag.retrieval.embedder as embedder_module
from pt_amb_rag.retrieval.embedder import embed, get_model


def _set_dimension(mock_st: MagicMock, dimension: int) -> None:
    mock_st.return_value.get_sentence_embedding_dimension.return_value = (
        dimension
    )


# ---------------------------------------------------------------------------
# get_model — singleton / lazy-loading behavior
# ---------------------------------------------------------------------------


class TestGetModel:
    def setup_method(self):
        # Reset the module-level singleton before each test so tests don't
        # leak state into each other.
        embedder_module._MODEL = None

    def teardown_method(self):
        embedder_module._MODEL = None

    def test_loads_model_with_expected_name(self):
        with patch(
            "pt_amb_rag.retrieval.embedder.SentenceTransformer"
        ) as mock_st:
            _set_dimension(mock_st, embedder_module._CONFIG.embedding.dimension)
            get_model()
            mock_st.assert_called_once_with(
                embedder_module._CONFIG.embedding.model
            )

    def test_returns_singleton_instance(self):
        with patch(
            "pt_amb_rag.retrieval.embedder.SentenceTransformer"
        ) as mock_st:
            _set_dimension(mock_st, embedder_module._CONFIG.embedding.dimension)
            first = get_model()
            second = get_model()
            assert first is second

    def test_only_constructs_model_once(self):
        with patch(
            "pt_amb_rag.retrieval.embedder.SentenceTransformer"
        ) as mock_st:
            _set_dimension(mock_st, embedder_module._CONFIG.embedding.dimension)
            get_model()
            get_model()
            get_model()
            mock_st.assert_called_once()

    def test_dimension_mismatch_is_rejected_loudly(self):
        """A multilingual embedder swapped in without updating the
        manifest would otherwise fail deep inside Qdrant."""
        with patch(
            "pt_amb_rag.retrieval.embedder.SentenceTransformer"
        ) as mock_st:
            _set_dimension(
                mock_st, embedder_module._CONFIG.embedding.dimension + 1
            )
            with pytest.raises(RuntimeError, match="dimensional vectors"):
                get_model()


# ---------------------------------------------------------------------------
# embed — mock the model so no real inference happens
# ---------------------------------------------------------------------------


class TestEmbed:
    def setup_method(self):
        embedder_module._MODEL = None

    def teardown_method(self):
        embedder_module._MODEL = None

    def _patch_model(self, encode_return):
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array(encode_return)
        patcher = patch(
            "pt_amb_rag.retrieval.embedder.get_model", return_value=mock_model
        )
        return patcher, mock_model

    def test_returns_list_of_lists(self):
        patcher, _ = self._patch_model([[0.1, 0.2, 0.3]])
        with patcher:
            result = embed(["hello world"])
        assert result == [[0.1, 0.2, 0.3]]
        assert isinstance(result, list)
        assert isinstance(result[0], list)

    def test_calls_encode_with_normalize_embeddings_true(self):
        patcher, mock_model = self._patch_model([[0.1, 0.2]])
        with patcher:
            embed(["hello"])
        mock_model.encode.assert_called_once_with(
            ["hello"], normalize_embeddings=True
        )

    def test_passes_texts_through_unmodified(self):
        texts = ["first sentence", "second sentence", "third"]
        patcher, mock_model = self._patch_model([[0.0]] * 3)
        with patcher:
            embed(texts)
        args, _ = mock_model.encode.call_args
        assert args[0] == texts

    def test_multiple_texts_returns_multiple_vectors(self):
        patcher, _ = self._patch_model([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        with patcher:
            result = embed(["a", "b", "c"])
        assert len(result) == 3
        assert result == [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]

    def test_empty_list_input(self):
        # sentence-transformers returns an empty array for empty input;
        # simulate that shape (0 rows).
        patcher, _ = self._patch_model(np.empty((0, 384)))
        with patcher:
            result = embed([])
        assert result == []

    def test_uses_shared_model_singleton(self):
        # embed() should go through get_model(), not construct its own model
        patcher, _ = self._patch_model([[0.1]])
        with patcher as mock_get_model:
            embed(["x"])
            mock_get_model.assert_called_once()


# ---------------------------------------------------------------------------
# Integration tests — load the real model, actually run inference.
# Slow (downloads/loads weights); skipped by default (see pyproject.toml
# addopts = "-m 'not integration'").
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestEmbedIntegration:
    def setup_method(self):
        embedder_module._MODEL = None

    def teardown_method(self):
        embedder_module._MODEL = None

    def test_embed_real_model_shape_and_normalization(self):
        result = embed(["this is a test sentence"])
        assert len(result) == 1
        vec = np.array(result[0])
        assert vec.shape == (384,)
        # normalize_embeddings=True -> unit-length vectors
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_similar_sentences_are_closer_than_dissimilar(self):
        a, b, c = embed(
            [
                "The cat sat on the mat.",
                "A feline rested on the rug.",
                "Quarterly revenue exceeded forecasts.",
            ]
        )
        sim_ab = np.dot(a, b)
        sim_ac = np.dot(a, c)
        assert sim_ab > sim_ac
