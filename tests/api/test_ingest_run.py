import time
from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from ingestion.metadata_store import IngestionMetadataStore
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


@pytest.fixture
def vector_store() -> StubVectorStore:
    return StubVectorStore()


@pytest.fixture
def metadata_store(tmp_path: Path) -> Iterable[IngestionMetadataStore]:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    yield store
    store.close()


@pytest.fixture
def client(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> Iterable[TestClient]:
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: StubLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    yield TestClient(app)

    app.dependency_overrides.clear()


Artifact = dict[str, object]


def _file_artifact(artifact_id: str = "uuid-1") -> Artifact:
    return {
        "artifactId": artifact_id,
        "sourceSystem": "GITHUB",
        "sourceId": "github:owner/repo:FILE:src/main/App.kt",
        "sourceUrl": "https://github.com/owner/repo/blob/main/src/main/App.kt",
        "artifactType": "FILE",
        "title": None,
        "bodyText": 'fun main() { println("hello") }',
        "mime": None,
        "language": "kotlin",
    }


def _issue_artifact(artifact_id: str = "uuid-2") -> Artifact:
    return {
        "artifactId": artifact_id,
        "sourceSystem": "GITHUB",
        "sourceId": "github:owner/repo:ISSUE:42",
        "sourceUrl": "https://github.com/owner/repo/issues/42",
        "artifactType": "ISSUE",
        "title": "Bug: login fails on mobile",
        "bodyText": "Steps to reproduce: open the app on iOS and tap login.",
        "mime": None,
        "language": None,
    }


def test_ingest_run_indexes_artifacts(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    response = client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [_file_artifact(), _issue_artifact()],
            "artifactsToDeindex": [],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["artifacts"]) == 2
    assert data["artifacts"][0]["artifact_id"] == "uuid-1"
    assert data["artifacts"][0]["chunk_count"] > 0
    assert data["artifacts"][0]["status"] == "completed"
    assert data["artifacts"][1]["artifact_id"] == "uuid-2"
    assert data["artifacts"][1]["chunk_count"] > 0
    assert data["artifacts"][1]["status"] == "completed"
    assert len(vector_store.chunks) == 2
    for chunk in vector_store.chunks:
        assert chunk.connector_id == "github"
        assert chunk.connector_source_id == "owner/repo"


def test_ingest_run_deindexes_before_indexing(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    # Seed an artifact that should be removed
    client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "old-artifact",
            "filename": "old.md",
            "content": "old content",
        },
    )
    assert vector_store.count_by_artifact("old-artifact") > 0

    response = client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [],
            "artifactsToDeindex": ["old-artifact"],
        },
    )

    assert response.status_code == 200
    assert vector_store.count_by_artifact("old-artifact") == 0


def test_ingest_run_empty_body_returns_empty_list(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingest/sync",
        json={"artifactsToIngest": [], "artifactsToDeindex": []},
    )

    assert response.status_code == 200
    assert response.json() == {"artifacts": []}


def test_ingest_run_artifact_with_no_content_returns_zero_chunks(
    client: TestClient,
) -> None:
    artifact: Artifact = _file_artifact()
    artifact["bodyText"] = None
    artifact["title"] = None

    response = client.post(
        "/api/v1/ingest/sync",
        json={"artifactsToIngest": [artifact], "artifactsToDeindex": []},
    )

    assert response.status_code == 200
    assert response.json()["artifacts"][0]["chunk_count"] == 0


def test_ingest_run_filename_derived_from_source_id_for_files(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [_file_artifact("uuid-kt")],
            "artifactsToDeindex": [],
        },
    )

    record = metadata_store.get_artifact("uuid-kt")
    assert record is not None
    assert record.filename == "src/main/App.kt"


def test_ingest_run_filename_derived_for_issue(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [_issue_artifact("uuid-issue")],
            "artifactsToDeindex": [],
        },
    )

    record = metadata_store.get_artifact("uuid-issue")
    assert record is not None
    assert record.filename == "issue-42.md"


def test_ingest_run_persists_issue_state_and_labels(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    artifact = _issue_artifact("uuid-state")
    artifact["state"] = "OPEN"
    artifact["labels"] = ["bug", "good first issue"]

    client.post(
        "/api/v1/ingest/sync",
        json={"artifactsToIngest": [artifact], "artifactsToDeindex": []},
    )

    record = metadata_store.get_artifact("uuid-state")
    assert record is not None
    assert record.state == "OPEN"
    assert record.labels == ["bug", "good first issue"]


def test_ingest_run_defaults_state_and_labels_when_absent(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    client.post(
        "/api/v1/ingest/sync",
        json={
            "artifactsToIngest": [_issue_artifact("uuid-no-state")],
            "artifactsToDeindex": [],
        },
    )

    record = metadata_store.get_artifact("uuid-no-state")
    assert record is not None
    assert record.state is None
    assert record.labels == []


class _FlakyEmbedLLMClient(StubLLMClient):
    """Raises LLMUnavailableError while embedding the file artifact's content,
    succeeds for everything else. Content-keyed (not a call counter) so the
    failing artifact stays deterministic under /ingest/sync's concurrent
    per-artifact processing, where call order across artifacts isn't fixed.
    """

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any("hello" in text for text in texts):
            raise LLMUnavailableError("embedding backend unavailable")
        return super().embed_batch(texts)


def test_ingest_run_llm_outage_on_one_artifact_does_not_sink_the_batch(
    metadata_store: IngestionMetadataStore,
    vector_store: StubVectorStore,
) -> None:
    """Regression test for issue #129 #6: a mid-batch LLM outage must be recorded
    per-artifact instead of 503ing the whole request and losing every result.
    """
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: _FlakyEmbedLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/ingest/sync",
            json={
                "artifactsToIngest": [
                    _file_artifact("uuid-1"),
                    _issue_artifact("uuid-2"),
                ],
                "artifactsToDeindex": [],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()["artifacts"]
    assert data[0]["artifact_id"] == "uuid-1"
    assert data[0]["status"] == "failed"
    assert data[0]["chunk_count"] == 0
    assert data[1]["artifact_id"] == "uuid-2"
    assert data[1]["status"] == "completed"
    assert data[1]["chunk_count"] > 0

    failed_record = metadata_store.get_artifact("uuid-1")
    assert failed_record is not None
    assert failed_record.status == "failed"


class _FlakyStore(StubVectorStore):
    """Raises when storing the file artifact's chunks, succeeds for everything
    else. Content-keyed (not a call counter) so the failing artifact stays
    deterministic under /ingest/sync's concurrent per-artifact processing,
    where call order across artifacts isn't fixed.
    """

    def add(self, chunks: list[Chunk]) -> None:
        if any(chunk.artifact_id == "uuid-1" for chunk in chunks):
            raise RuntimeError("storage backend unavailable")
        super().add(chunks)


def test_ingest_run_storage_error_on_one_artifact_does_not_sink_the_batch(
    metadata_store: IngestionMetadataStore,
) -> None:
    """A storage error for one artifact must not 500 the whole batch."""
    flaky_store = _FlakyStore()
    app.dependency_overrides[get_store] = lambda: flaky_store
    app.dependency_overrides[get_llm] = lambda: StubLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/ingest/sync",
            json={
                "artifactsToIngest": [
                    _file_artifact("uuid-1"),
                    _issue_artifact("uuid-2"),
                ],
                "artifactsToDeindex": [],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()["artifacts"]
    assert data[0]["artifact_id"] == "uuid-1"
    assert data[0]["status"] == "failed"
    assert data[1]["artifact_id"] == "uuid-2"
    assert data[1]["status"] == "completed"
    assert data[1]["chunk_count"] > 0


class _SlowFirstLLMClient(StubLLMClient):
    """Delays embedding the file artifact's content so it finishes after
    artifacts submitted behind it in the request -- proves the response
    stays in request order even when completion order is reversed.
    """

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if any("hello" in text for text in texts):
            time.sleep(0.2)
        return super().embed_batch(texts)


def test_ingest_run_preserves_request_order_despite_reversed_completion_order(
    metadata_store: IngestionMetadataStore,
    vector_store: StubVectorStore,
) -> None:
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: _SlowFirstLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    try:
        response = client.post(
            "/api/v1/ingest/sync",
            json={
                "artifactsToIngest": [
                    _file_artifact("uuid-1"),
                    _issue_artifact("uuid-2"),
                ],
                "artifactsToDeindex": [],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    data = response.json()["artifacts"]
    assert [item["artifact_id"] for item in data] == ["uuid-1", "uuid-2"]
    assert all(item["status"] == "completed" for item in data)


class _DelayedLLMClient(StubLLMClient):
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        time.sleep(0.15)
        return super().embed_batch(texts)


def test_ingest_run_processes_artifacts_concurrently(
    metadata_store: IngestionMetadataStore,
    vector_store: StubVectorStore,
) -> None:
    """Regression test: /ingest/sync used to process artifacts one at a time,
    so N artifacts took N times as long purely from sequential network
    round-trips to the embedding API. This asserts real overlap, not just
    that concurrent code runs without error.
    """
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: _DelayedLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    artifacts = [_file_artifact(f"uuid-{i}") for i in range(5)]

    try:
        started = time.monotonic()
        response = client.post(
            "/api/v1/ingest/sync",
            json={"artifactsToIngest": artifacts, "artifactsToDeindex": []},
        )
        elapsed = time.monotonic() - started
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert all(item["status"] == "completed" for item in response.json()["artifacts"])
    # Sequential would take >= 5 * 0.15s = 0.75s. Comfortably below that
    # (but above a single 0.15s call) proves real concurrent overlap.
    assert elapsed < 0.5
