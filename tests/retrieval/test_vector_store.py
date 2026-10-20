import uuid
from unittest.mock import MagicMock, patch

from qdrant_client.http import models as qmodels

from pt_amb_rag.retrieval.vector_store import (
    COLLECTION_NAME,
    PAYLOAD_FIELDS,
    VECTOR_SIZE,
    _build_filter,
    _payload,
    _point_id,
    ensure_collection,
    get_client,
    search,
    upsert_chunks,
)


def _field_conditions(f: qmodels.Filter | None) -> list[qmodels.FieldCondition]:
    """Narrow a Filter's `.must` (typed as Condition | list[Condition] | None)
    down to a concrete list[FieldCondition], for tests that only ever build
    FieldCondition filters via _build_filter."""
    assert f is not None
    must = f.must
    assert isinstance(must, list)
    conditions: list[qmodels.FieldCondition] = []
    for condition in must:
        assert isinstance(condition, qmodels.FieldCondition)
        conditions.append(condition)
    return conditions


def _match_value(condition: qmodels.FieldCondition) -> object:
    """Narrow a FieldCondition's `.match` (a union of many Match* types) down
    to MatchValue and return its `.value`, for tests that only ever build
    MatchValue filters via _build_filter."""
    match = condition.match
    assert isinstance(match, qmodels.MatchValue)
    return match.value


# ---------------------------------------------------------------------------
# get_client
# ---------------------------------------------------------------------------


class TestGetClient:
    def test_default_local_url(self):
        with patch(
            "pt_amb_rag.retrieval.vector_store.QdrantClient"
        ) as mock_cls:
            get_client()
            mock_cls.assert_called_once_with(url="http://localhost:6333")

    def test_custom_url(self):
        with patch(
            "pt_amb_rag.retrieval.vector_store.QdrantClient"
        ) as mock_cls:
            get_client(url="http://qdrant.internal:6333")
            mock_cls.assert_called_once_with(url="http://qdrant.internal:6333")

    def test_uses_qdrant_url_env_var(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        with patch(
            "pt_amb_rag.retrieval.vector_store.QdrantClient"
        ) as mock_cls:
            get_client()
            mock_cls.assert_called_once_with(url="http://qdrant:6333")

    def test_explicit_url_overrides_env_var(self, monkeypatch):
        monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
        with patch(
            "pt_amb_rag.retrieval.vector_store.QdrantClient"
        ) as mock_cls:
            get_client(url="http://other:6333")
            mock_cls.assert_called_once_with(url="http://other:6333")


# ---------------------------------------------------------------------------
# ensure_collection
# ---------------------------------------------------------------------------


class TestEnsureCollection:
    def test_creates_collection_when_missing(self):
        client = MagicMock()
        client.collection_exists.return_value = False

        ensure_collection(client, "my_collection")

        client.collection_exists.assert_called_once_with("my_collection")
        client.create_collection.assert_called_once()
        _, kwargs = client.create_collection.call_args
        assert kwargs["collection_name"] == "my_collection"
        assert kwargs["vectors_config"].size == VECTOR_SIZE
        assert kwargs["vectors_config"].distance == qmodels.Distance.COSINE

    def test_skips_creation_when_already_exists(self):
        client = MagicMock()
        client.collection_exists.return_value = True

        ensure_collection(client, "my_collection")

        client.create_collection.assert_not_called()

    def test_default_collection_name(self):
        client = MagicMock()
        client.collection_exists.return_value = True
        ensure_collection(client)
        client.collection_exists.assert_called_once_with(COLLECTION_NAME)


# ---------------------------------------------------------------------------
# _point_id
# ---------------------------------------------------------------------------


class TestPointId:
    def test_deterministic_for_same_content_hash(self):
        chunk = {"content_hash": "abc123"}
        assert _point_id(chunk) == _point_id(chunk)

    def test_different_content_hash_gives_different_id(self):
        assert _point_id({"content_hash": "abc"}) != _point_id(
            {"content_hash": "xyz"}
        )

    def test_is_valid_uuid_string(self):
        result = _point_id({"content_hash": "abc123"})
        uuid.UUID(result)  # raises ValueError if not a valid UUID

    def test_missing_content_hash_falls_back_to_random_uuid(self):
        id_1 = _point_id({})
        id_2 = _point_id({})
        uuid.UUID(id_1)
        uuid.UUID(id_2)
        assert id_1 != id_2

    def test_uses_uuid5_namespace_url(self):
        chunk = {"content_hash": "some-hash"}
        expected = str(uuid.uuid5(uuid.NAMESPACE_URL, "some-hash"))
        assert _point_id(chunk) == expected


# ---------------------------------------------------------------------------
# _payload
# ---------------------------------------------------------------------------


class TestPayload:
    def test_extracts_only_known_fields(self):
        chunk = {
            "regulation": "dl_150_2015",
            "article_num": "26",
            "text": "some text",
            "content_hash": "hash1",
            "extra_field_not_in_schema": "should be dropped",
        }
        payload = _payload(chunk)
        assert set(payload.keys()) == set(PAYLOAD_FIELDS)
        assert "extra_field_not_in_schema" not in payload
        assert payload["regulation"] == "dl_150_2015"
        assert payload["article_num"] == "26"

    def test_missing_fields_default_to_none(self):
        payload = _payload({"regulation": "dl_150_2015", "text": "x"})
        assert payload["chapter"] is None
        assert payload["annex_id"] is None
        assert payload["numero"] is None
        assert payload["alinea"] is None
        assert payload["diploma"] is None


# ---------------------------------------------------------------------------
# upsert_chunks
# ---------------------------------------------------------------------------


class TestUpsertChunks:
    def test_empty_chunks_returns_zero_without_touching_client(self):
        client = MagicMock()
        assert upsert_chunks([], client=client) == 0
        client.upsert.assert_not_called()

    def test_embeds_chunk_texts(self):
        chunks = [
            {"text": "a", "regulation": "dl_150_2015"},
            {"text": "b", "regulation": "dl_127_2013"},
        ]
        client = MagicMock()
        client.collection_exists.return_value = True

        with patch(
            "pt_amb_rag.retrieval.vector_store.embed",
            return_value=[[0.1, 0.2], [0.3, 0.4]],
        ) as mock_embed:
            upsert_chunks(chunks, client=client)
            mock_embed.assert_called_once_with(["a", "b"])

    def test_upserts_points_with_correct_vectors_and_payload(self):
        chunks = [
            {"text": "hello", "regulation": "dl_150_2015", "article_num": "26"}
        ]
        client = MagicMock()
        client.collection_exists.return_value = True

        with patch(
            "pt_amb_rag.retrieval.vector_store.embed", return_value=[[0.5, 0.6]]
        ):
            upsert_chunks(chunks, client=client)

        client.upsert.assert_called_once()
        _, kwargs = client.upsert.call_args
        points = kwargs["points"]
        assert len(points) == 1
        assert points[0].vector == [0.5, 0.6]
        assert points[0].payload["regulation"] == "dl_150_2015"
        assert points[0].payload["article_num"] == "26"

    def test_returns_number_of_points_upserted(self):
        chunks = [{"text": "a"}, {"text": "b"}, {"text": "c"}]
        client = MagicMock()
        client.collection_exists.return_value = True
        with patch(
            "pt_amb_rag.retrieval.vector_store.embed",
            return_value=[[0.0], [0.0], [0.0]],
        ):
            result = upsert_chunks(chunks, client=client)
        assert result == 3

    def test_ensures_collection_before_upsert(self):
        client = MagicMock()
        client.collection_exists.return_value = False
        with patch(
            "pt_amb_rag.retrieval.vector_store.embed", return_value=[[0.0]]
        ):
            upsert_chunks([{"text": "x"}], client=client)
        client.create_collection.assert_called_once()

    def test_uses_default_client_when_none_provided(self):
        with (
            patch(
                "pt_amb_rag.retrieval.vector_store.get_client"
            ) as mock_get_client,
            patch(
                "pt_amb_rag.retrieval.vector_store.embed", return_value=[[0.0]]
            ),
        ):
            mock_client = MagicMock()
            mock_client.collection_exists.return_value = True
            mock_get_client.return_value = mock_client

            upsert_chunks([{"text": "x"}])

            mock_get_client.assert_called_once()
            mock_client.upsert.assert_called_once()

    def test_same_content_hash_produces_same_point_id_across_calls(self):
        chunk = {"text": "x", "content_hash": "stable-hash"}
        client = MagicMock()
        client.collection_exists.return_value = True

        with patch(
            "pt_amb_rag.retrieval.vector_store.embed", return_value=[[0.0]]
        ):
            upsert_chunks([chunk], client=client)
            first_id = client.upsert.call_args.kwargs["points"][0].id

            upsert_chunks([chunk], client=client)
            second_id = client.upsert.call_args.kwargs["points"][0].id

        assert first_id == second_id


# ---------------------------------------------------------------------------
# _build_filter
# ---------------------------------------------------------------------------


class TestBuildFilter:
    def test_no_filters_returns_none(self):
        assert _build_filter(None, None) is None

    def test_regulation_only(self):
        result = _build_filter("dl_150_2015", None)
        assert isinstance(result, qmodels.Filter)
        conditions = _field_conditions(result)
        assert len(conditions) == 1
        assert conditions[0].key == "regulation"
        assert _match_value(conditions[0]) == "dl_150_2015"

    def test_article_num_only(self):
        result = _build_filter(None, 26)
        conditions = _field_conditions(result)
        assert len(conditions) == 1
        assert conditions[0].key == "article_num"
        assert _match_value(conditions[0]) == 26

    def test_both_filters_combined_with_and(self):
        result = _build_filter("dl_150_2015", 26)
        conditions = _field_conditions(result)
        assert len(conditions) == 2
        keys = {cond.key for cond in conditions}
        assert keys == {"regulation", "article_num"}


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


class TestSearch:
    def test_calls_query_points_with_expected_args(self):
        client = MagicMock()
        mock_points = [MagicMock(), MagicMock()]
        client.query_points.return_value = MagicMock(points=mock_points)

        result = search([0.1, 0.2, 0.3], k=5, client=client)

        client.query_points.assert_called_once()
        _, kwargs = client.query_points.call_args
        assert kwargs["collection_name"] == COLLECTION_NAME
        assert kwargs["query"] == [0.1, 0.2, 0.3]
        assert kwargs["limit"] == 5
        assert kwargs["query_filter"] is None
        assert result == mock_points

    def test_applies_regulation_and_article_filter(self):
        client = MagicMock()
        client.query_points.return_value = MagicMock(points=[])

        search([0.1], regulation="dl_127_2013", article_num=21, client=client)

        _, kwargs = client.query_points.call_args
        query_filter = kwargs["query_filter"]
        assert isinstance(query_filter, qmodels.Filter)
        assert len(_field_conditions(query_filter)) == 2

    def test_default_k_is_five(self):
        client = MagicMock()
        client.query_points.return_value = MagicMock(points=[])
        search([0.1], client=client)
        _, kwargs = client.query_points.call_args
        assert kwargs["limit"] == 5

    def test_uses_default_client_when_none_provided(self):
        with patch(
            "pt_amb_rag.retrieval.vector_store.get_client"
        ) as mock_get_client:
            mock_client = MagicMock()
            mock_client.query_points.return_value = MagicMock(points=[])
            mock_get_client.return_value = mock_client

            search([0.1])

            mock_get_client.assert_called_once()
            mock_client.query_points.assert_called_once()

    def test_returns_points_from_response(self):
        client = MagicMock()
        expected_points = [MagicMock(score=0.9), MagicMock(score=0.5)]
        client.query_points.return_value = MagicMock(points=expected_points)
        result = search([0.1], client=client)
        assert result == expected_points
