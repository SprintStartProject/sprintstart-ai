import json
from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from ingestion.metadata_store import IngestionMetadataStore
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


class FailingEmbedLLMClient(StubLLMClient):
    def embed(self, text: str) -> list[float]:
        raise LLMUnavailableError("embedding backend unavailable")


class RecordingLLMClient(StubLLMClient):
    """Records every prompt passed to ``generate`` for assertion."""

    def __init__(self, generate_response: str = "stub answer") -> None:
        super().__init__(generate_response=generate_response)
        self.generate_calls: list[list[Message]] = []

    def generate(self, messages: list[Message]) -> str:
        self.generate_calls.append(messages)
        return super().generate(messages)


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


def test_ingest_returns_artifact_and_chunk_metadata(
    client: TestClient,
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-1",
            "filename": "notes.txt",
            "content": "SprintStart uses OLLAMA_EMBED_MODEL for embeddings.",
        },
    )

    assert response.status_code == 200

    body = response.json()

    assert body["artifact_id"] == "artifact-1"
    assert body["chunk_count"] >= 1
    assert body["artifact"]["id"] == "artifact-1"
    assert body["artifact"]["filename"] == "notes.txt"
    assert body["artifact"]["content_type"] == "text/plain"
    assert body["artifact"]["source_type"] == "file"
    assert body["artifact"]["status"] == "completed"
    assert body["artifact"]["chunk_count"] == body["chunk_count"]

    assert len(body["chunks"]) == body["chunk_count"]
    assert body["chunks"][0]["artifact_id"] == "artifact-1"
    assert body["chunks"][0]["filename"] == "notes.txt"
    assert body["chunks"][0]["chunk_index"] == 0
    assert body["chunks"][0]["vector_store_id"] == body["chunks"][0]["id"]
    assert "embedding" not in body["chunks"][0]

    artifact = metadata_store.get_artifact("artifact-1")
    assert artifact is not None
    assert artifact.status == "completed"
    assert artifact.chunk_count == body["chunk_count"]

    stored = vector_store.all_chunks()
    assert len(stored) == body["chunk_count"]
    assert stored[0].artifact_id == "artifact-1"
    assert stored[0].filename == "notes.txt"
    assert stored[0].position == 0


def test_reingest_replaces_chunks_and_preserves_created_at(
    client: TestClient,
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    first = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "doc-1",
            "filename": "notes.txt",
            "content": "first version of the document.",
        },
    )
    assert first.status_code == 200

    original = metadata_store.get_artifact("doc-1")
    assert original is not None

    second = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "doc-1",
            "filename": "notes.txt",
            "content": "second version with different content.",
        },
    )
    assert second.status_code == 200

    # Only the latest ingest's chunks remain in the vector store.
    stored = [c for c in vector_store.all_chunks() if c.artifact_id == "doc-1"]
    assert len(stored) == second.json()["chunk_count"]

    updated = metadata_store.get_artifact("doc-1")
    assert updated is not None
    assert updated.created_at == original.created_at
    assert updated.updated_at >= original.updated_at


def test_failed_embedding_marks_artifact_failed(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: FailingEmbedLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-failed",
            "filename": "notes.txt",
            "content": "This should fail during embedding.",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 503

    artifact = metadata_store.get_artifact("artifact-failed")
    assert artifact is not None
    assert artifact.status == "failed"
    assert artifact.error_message is not None
    assert "embedding backend unavailable" in artifact.error_message

    assert vector_store.all_chunks() == []


def test_ingest_auto_classifies_test_filename(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-test",
            "filename": "test_agent.py",
            "content": "def test_something(): assert True",
        },
    )

    assert response.status_code == 200
    chunks = vector_store.all_chunks()
    assert chunks
    assert all(chunk.source_role == "test" for chunk in chunks)


def test_ingest_respects_explicit_source_role(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    # A non-test filename explicitly marked as test material.
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-fixture",
            "filename": "scenario.json",
            "content": '{"sample": "data"}',
            "source_role": "test",
        },
    )

    assert response.status_code == 200
    chunks = vector_store.all_chunks()
    assert chunks
    assert all(chunk.source_role == "test" for chunk in chunks)


def test_ingest_defaults_to_primary_for_normal_files(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-doc",
            "filename": "dev-setup.md",
            "content": "# Dev setup\nInstall the toolchain.",
        },
    )

    assert response.status_code == 200
    chunks = vector_store.all_chunks()
    assert chunks
    assert all(chunk.source_role == "primary" for chunk in chunks)


def test_ingest_without_flags_skips_llm_call_by_default(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    # Defaults (semantic_boundaries=False, contextualize=False) must not
    # trigger an LLM call — the feature is opt-in.
    llm = RecordingLLMClient()
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-default-flags",
            "filename": "notes.txt",
            "content": "First paragraph.\n\nSecond paragraph.",
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert llm.generate_calls == []


def test_ingest_calls_llm_when_flags_explicitly_enabled(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    # Explicitly opting in should trigger an LLM call for text content,
    # even though the stub's non-JSON response makes the chunker fall back
    # to chunk_text.
    llm = RecordingLLMClient()
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-explicit-flags",
            "filename": "notes.txt",
            "content": "First paragraph.\n\nSecond paragraph.",
            "semantic_boundaries": True,
            "contextualize": True,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(llm.generate_calls) == 1


def test_ingest_with_both_chunking_flags_disabled_skips_llm_call(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    llm = RecordingLLMClient()
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-no-llm-chunking",
            "filename": "notes.txt",
            "content": "First paragraph.\n\nSecond paragraph.",
            "semantic_boundaries": False,
            "contextualize": False,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert llm.generate_calls == []


def test_ingest_contextualize_prepends_context_block(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    plan = json.dumps(
        {"boundaries": [], "context_blocks": {"0": "Context: onboarding notes."}}
    )
    llm = RecordingLLMClient(generate_response=plan)
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    client = TestClient(app)

    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-contextualized",
            "filename": "notes.txt",
            "content": "Just one short paragraph.",
            "semantic_boundaries": False,
            "contextualize": True,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["chunks"][0]["text"].startswith("Context: onboarding notes.")


def test_ingest_stores_project_ids_on_chunks_and_record(
    client: TestClient,
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-1",
            "filename": "notes.txt",
            "content": "SprintStart uses OLLAMA_EMBED_MODEL for embeddings.",
            "projectIds": ["project-1"],
        },
    )

    assert response.status_code == 200
    assert vector_store.chunks
    for chunk in vector_store.chunks:
        assert chunk.project_ids == ("project-1",)

    record = metadata_store.get_artifact("artifact-1")
    assert record is not None
    assert record.project_ids == ("project-1",)


class FailingEmbedBatchLLMClient(StubLLMClient):
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise LLMUnavailableError("embedding backend unavailable")


def test_ingest_revokes_lost_membership_before_embedding(
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    """Losing a project must take effect even if the re-ingest never completes.

    Chunks are only replaced after embedding succeeds, so an embedding outage
    would otherwise leave the old ``project-1`` chunks queryable from a project
    the artifact no longer belongs to.
    """
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: StubLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    try:
        client = TestClient(app)
        client.post(
            "/api/v1/ingest",
            json={
                "artifact_id": "artifact-1",
                "filename": "notes.txt",
                "content": "SprintStart uses OLLAMA_EMBED_MODEL for embeddings.",
                "projectIds": ["project-1"],
            },
        )
        assert vector_store.project_ids_for_artifact("artifact-1") == frozenset(
            {"project-1"}
        )

        app.dependency_overrides[get_llm] = lambda: FailingEmbedBatchLLMClient()
        response = TestClient(app).post(
            "/api/v1/ingest",
            json={
                "artifact_id": "artifact-1",
                "filename": "notes.txt",
                "content": "SprintStart uses OLLAMA_EMBED_MODEL for embeddings.",
                "projectIds": ["project-2"],
            },
        )

        assert response.status_code == 503
        assert vector_store.project_ids_for_artifact("artifact-1") == frozenset()
    finally:
        app.dependency_overrides.clear()


def test_ingest_rejects_project_id_containing_delimiter(client: TestClient) -> None:
    """``|`` is the metadata delimiter: ``"a|b"`` would decode as two projects."""
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-1",
            "filename": "notes.txt",
            "content": "content",
            "projectIds": ["project-1|project-2"],
        },
    )

    assert response.status_code == 422


def test_ingest_rejects_blank_project_id(client: TestClient) -> None:
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-1",
            "filename": "notes.txt",
            "content": "content",
            "projectIds": ["   "],
        },
    )

    assert response.status_code == 422


def test_ingest_normalizes_whitespace_and_duplicate_project_ids(
    client: TestClient,
    vector_store: StubVectorStore,
) -> None:
    response = client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "artifact-1",
            "filename": "notes.txt",
            "content": "SprintStart uses OLLAMA_EMBED_MODEL for embeddings.",
            "projectIds": [" project-1 ", "project-1"],
        },
    )

    assert response.status_code == 200
    assert vector_store.chunks
    for chunk in vector_store.chunks:
        assert chunk.project_ids == ("project-1",)
