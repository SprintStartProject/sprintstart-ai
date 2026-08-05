import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_URL = "/api/v1/insights/knowledge-gaps/detect"
_NOW = "2025-05-01T00:00:00+00:00"
_PROJECT = "project-1"


def _artifact(artifact_id: str, filename: str, source_id: str) -> ArtifactRecord:
    return ArtifactRecord(
        id=artifact_id,
        filename=filename,
        content_type="text/markdown",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at=_NOW,
        updated_at=_NOW,
        source_id=source_id,
        project_ids=(_PROJECT,),
    )


@pytest.fixture
def metadata_store() -> IngestionMetadataStore:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(
        _artifact("a1", "README.md", "github:acme/auth:FILE:README.md")
    )
    store.save_completed_artifact(
        _artifact("a2", "setup.md", "github:acme/auth:FILE:setup.md")
    )
    return store


def _client(
    llm: Any, metadata_store: IngestionMetadataStore
) -> Generator[TestClient, Any, None]:
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = StubVectorStore
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_detect_returns_gaps(metadata_store: IngestionMetadataStore) -> None:
    llm = StubLLMClient(generate_response=json.dumps({"present": ["readme", "setup"]}))
    client = next(_client(llm, metadata_store))

    response = client.post(_URL, json={"projectId": _PROJECT})

    assert response.status_code == 200, response.text
    gaps = response.json()["gaps"]
    assert len(gaps) == 1
    gap = gaps[0]
    assert gap["component"] == "acme/auth"
    assert gap["missingTypes"] == ["architecture", "adr", "api", "runbook"]
    assert gap["presentTypes"] == ["readme", "setup"]
    assert gap["lastUpdated"] == _NOW
    assert gap["severity"] == "high"
    # Fields the AI service deliberately does not source.
    assert "owners" not in gap
    assert "relatedQuestions" not in gap


def test_detect_returns_503_when_llm_unavailable(
    metadata_store: IngestionMetadataStore,
) -> None:
    class UnavailableLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            raise LLMUnavailableError("backend down")

    client = next(_client(UnavailableLLM(), metadata_store))

    response = client.post(_URL, json={"projectId": _PROJECT})

    assert response.status_code == 503


def test_detect_requires_a_project(metadata_store: IngestionMetadataStore) -> None:
    client = next(_client(StubLLMClient(), metadata_store))

    response = client.post(_URL, json={})

    assert response.status_code == 422


def test_detect_only_reports_the_requested_projects_components(
    metadata_store: IngestionMetadataStore,
) -> None:
    llm = StubLLMClient(generate_response=json.dumps({"present": ["readme"]}))
    client = next(_client(llm, metadata_store))

    response = client.post(_URL, json={"projectId": "project-2"})

    assert response.status_code == 200, response.text
    assert response.json()["gaps"] == []
