import json
from collections.abc import Callable, Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_PROJECT = "project-1"
_VPN = [1.0, 0.0, 0.0]
_PASSWORD = [0.0, 1.0, 0.0]


def _embed_fn(text: str) -> list[float]:
    if "VPN" in text:
        return _VPN
    if "password" in text.lower():
        return _PASSWORD
    return [0.0, 0.0, 1.0]


class _EchoLLM(StubLLMClient):
    """Scripted grouping answer, then a redaction pass-through.

    ``group_faqs`` makes exactly two ``generate`` calls: one for clustering
    and one from ``redact_pii`` for name redaction. Tests here focus on the
    HTTP contract, not clustering quality, so the grouping answer is scripted
    to match the fixture questions' ids.
    """

    def __init__(
        self,
        groups: list[list[str]] | None = None,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        super().__init__(embed_fn=embed_fn)
        self._groups = groups if groups is not None else [["q1", "q2"], ["q3"]]
        self._calls = 0

    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        self._calls += 1
        if self._calls == 1:
            return json.dumps({"groups": self._groups, "discard_ids": []})
        payload = json.loads(messages[-1]["content"])  # type: ignore[index]
        return json.dumps({"texts": payload["texts"]})


class _FailingLLM(StubLLMClient):
    def embed(self, text: str) -> list[float]:
        raise LLMUnavailableError("local LLM unavailable")


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
    app.dependency_overrides[get_llm] = lambda: _EchoLLM(embed_fn=_embed_fn)
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_group_endpoint_groups_and_returns_documents(
    client: TestClient,
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    vector_store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc_001",
                filename="vpn-setup.md",
                text="How to get VPN access set up",
                embedding=_VPN,
                project_ids=(_PROJECT,),
            )
        ]
    )
    metadata_store.save_completed_artifact(
        ArtifactRecord(
            id="doc_001",
            filename="VPN Setup Guide.md",
            content_type="text/markdown",
            source_type="confluence",
            size_bytes=100,
            chunk_count=1,
            status="completed",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    response = client.post(
        "/api/v1/insights/faq/group",
        json={
            "projectId": _PROJECT,
            "questions": [
                {"id": "q1", "text": "How do I get VPN access?"},
                {"id": "q2", "text": "Can someone enable VPN for me?"},
                {"id": "q3", "text": "How do I reset my password?"},
            ],
        },
    )

    assert response.status_code == 200
    body = response.json()

    assert len(body["groups"]) == 2
    vpn_group = body["groups"][0]
    assert vpn_group["count"] == 2
    assert vpn_group["question"] == "How do I get VPN access?"
    assert vpn_group["questions"] == [
        "How do I get VPN access?",
        "Can someone enable VPN for me?",
    ]
    assert vpn_group["documents"] == [
        {"id": "doc_001", "title": "VPN Setup Guide.md", "source": "confluence"}
    ]

    password_group = body["groups"][1]
    assert password_group["count"] == 1


def test_group_endpoint_empty_questions_returns_empty_groups(
    client: TestClient,
) -> None:
    response = client.post(
        "/api/v1/insights/faq/group", json={"projectId": _PROJECT, "questions": []}
    )

    assert response.status_code == 200
    assert response.json() == {"groups": []}


def test_group_endpoint_llm_unavailable_returns_503(client: TestClient) -> None:
    app.dependency_overrides[get_llm] = lambda: _FailingLLM()

    response = client.post(
        "/api/v1/insights/faq/group",
        json={
            "projectId": _PROJECT,
            "questions": [{"id": "q1", "text": "How do I get VPN access?"}],
        },
    )

    assert response.status_code == 503
    assert "local LLM unavailable" in response.json()["detail"]


def test_group_endpoint_rejects_missing_questions_field(client: TestClient) -> None:
    response = client.post("/api/v1/insights/faq/group", json={})

    assert response.status_code == 422
