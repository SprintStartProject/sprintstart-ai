import base64
from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from corpus.types import Chunk
from llm.errors import LLMUnavailableError
from store.metadata_store import IngestionMetadataStore
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_URL = "/api/v1/ingest"
_PNG = base64.b64encode(b"not really a png").decode("ascii")


class _NoVisionLLM(StubLLMClient):
    def caption_image(self, image_bytes: bytes) -> str:
        raise LLMUnavailableError("vision model offline")


class _FailingAddStore(StubVectorStore):
    def add(self, chunks: list[Chunk]) -> None:
        raise RuntimeError("disk full")


class _FailingDeleteStore(StubVectorStore):
    def delete(self, artifact_id: str, exclude_ids: list[str]) -> int:
        raise RuntimeError("index locked")


@pytest.fixture
def metadata_store(tmp_path: Path) -> Iterable[IngestionMetadataStore]:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    yield store
    store.close()


def _client(
    store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
    llm: StubLLMClient | None = None,
) -> TestClient:
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_llm] = lambda: llm or StubLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    return TestClient(app)


@pytest.fixture(autouse=True)
def _clear_overrides() -> Iterable[None]:
    yield
    app.dependency_overrides.clear()


def _post(
    client: TestClient, artifact_id: str, filename: str, content: str
) -> Response:
    return client.post(
        _URL,
        json={"artifact_id": artifact_id, "filename": filename, "content": content},
    )


def test_content_over_the_limit_is_rejected_and_the_old_chunks_are_removed(
    metadata_store: IngestionMetadataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = StubVectorStore()
    client = _client(store, metadata_store)
    assert _post(client, "a1", "notes.txt", "short text").status_code == 200
    assert store.count() == 1
    monkeypatch.setenv("INGEST_MAX_CONTENT_LENGTH", "10")

    response = _post(client, "a1", "notes.txt", "x" * 11)

    assert response.status_code == 413
    assert (
        response.json()["detail"] == "Content exceeds maximum length of 10 characters."
    )
    assert store.count() == 0
    record = metadata_store.get_artifact("a1")
    assert record is not None and record.status == "failed"
    assert record.error_message == "Content exceeds maximum length of 10 characters."


def test_content_exactly_at_the_limit_is_accepted(
    metadata_store: IngestionMetadataStore, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGEST_MAX_CONTENT_LENGTH", "10")
    client = _client(StubVectorStore(), metadata_store)

    assert _post(client, "a1", "notes.txt", "x" * 10).status_code == 200


def test_an_image_that_is_not_base64_is_rejected_and_recorded(
    metadata_store: IngestionMetadataStore,
) -> None:
    client = _client(StubVectorStore(), metadata_store)

    response = _post(client, "img", "diagram.png", "!!! not base64 !!!")

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Image content for 'diagram.png' is not valid base64."
    )
    record = metadata_store.get_artifact("img")
    assert record is not None and record.status == "failed"
    assert record.error_message == response.json()["detail"]


def test_an_image_is_ingested_as_its_caption(
    metadata_store: IngestionMetadataStore,
) -> None:
    store = StubVectorStore()
    client = _client(store, metadata_store, StubLLMClient(caption="a system diagram"))

    response = _post(client, "img", "diagram.png", _PNG)

    assert response.status_code == 200
    body = response.json()
    assert body["chunk_count"] == 1
    assert body["chunks"][0]["text"] == "a system diagram"
    assert body["chunks"][0]["kind"] == "image"


def test_an_image_is_skipped_when_the_vision_model_is_unavailable(
    metadata_store: IngestionMetadataStore,
) -> None:
    store = StubVectorStore()
    client = _client(store, metadata_store, _NoVisionLLM())

    response = _post(client, "img", "diagram.png", _PNG)

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 0
    assert store.count() == 0
    record = metadata_store.get_artifact("img")
    assert record is not None and record.status == "completed"
    assert record.chunk_count == 0


def test_content_that_parses_to_nothing_completes_with_no_chunks_and_clears_old_ones(
    metadata_store: IngestionMetadataStore,
) -> None:
    store = StubVectorStore()
    client = _client(store, metadata_store)
    assert _post(client, "a1", "notes.txt", "some text").json()["chunk_count"] == 1

    response = _post(client, "a1", "notes.txt", "   ")

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 0
    assert store.count() == 0
    record = metadata_store.get_artifact("a1")
    assert record is not None
    assert (record.status, record.chunk_count) == ("completed", 0)


def test_a_storage_failure_is_a_500_and_marks_the_artifact_failed(
    metadata_store: IngestionMetadataStore,
) -> None:
    client = _client(_FailingAddStore(), metadata_store)

    response = _post(client, "a1", "notes.txt", "some text")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to store chunks in vector database: disk full"
    )
    record = metadata_store.get_artifact("a1")
    assert record is not None and record.status == "failed"
    assert record.error_message == response.json()["detail"]


def test_failing_to_clear_old_chunks_for_empty_content_is_a_500(
    metadata_store: IngestionMetadataStore,
) -> None:
    client = _client(_FailingDeleteStore(), metadata_store)

    response = _post(client, "a1", "notes.txt", "   ")

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to delete existing vector chunks: index locked"
    )
    record = metadata_store.get_artifact("a1")
    assert record is not None and record.status == "failed"


def test_failing_to_clear_old_chunks_after_a_skipped_image_is_a_500(
    metadata_store: IngestionMetadataStore,
) -> None:
    client = _client(_FailingDeleteStore(), metadata_store, _NoVisionLLM())

    response = _post(client, "img", "diagram.png", _PNG)

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "Failed to delete existing vector chunks: index locked"
    )
