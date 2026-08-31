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
        titles: list[str] | None = None,
    ) -> None:
        super().__init__(embed_fn=embed_fn)
        self._groups = groups if groups is not None else [["q1", "q2"], ["q3"]]
        self._titles = titles or ["Getting VPN access"] * len(self._groups)
        self._calls = 0

    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        self._calls += 1
        if self._calls == 1:
            return json.dumps(
                {
                    "groups": [
                        {"ids": ids, "title": label}
                        for ids, label in zip(self._groups, self._titles, strict=True)
                    ],
                    "discard_ids": [],
                }
            )
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
        {"ids": ["q1"], "text": "How do I get VPN access?"},
        {"ids": ["q2"], "text": "Can someone enable VPN for me?"},
    ]
    # Every member id, not just the sampled ones: the backend maps these back
    # to the messages they were asked in to rebuild recency and trend.
    assert vpn_group["questionIds"] == ["q1", "q2"]
    # The title is what a PM scans the list by, not the verbatim question.
    assert vpn_group["title"] == "Getting VPN access"
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


class _ScriptedLLM(StubLLMClient):
    """Answers every `generate` with one fixed JSON payload."""

    def __init__(self, payload: object) -> None:
        super().__init__(embed_fn=_embed_fn)
        self._payload = payload if isinstance(payload, str) else json.dumps(payload)

    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        return self._payload


def test_classify_endpoint_returns_the_matched_group_in_camel_case(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_llm] = lambda: _ScriptedLLM(
        {
            "relevant": True,
            "group_id": "g1",
            "title": "Getting VPN access",
            "redacted_question": "Can someone enable VPN for me?",
        }
    )

    response = client.post(
        "/api/v1/insights/faq/classify",
        json={
            "projectId": _PROJECT,
            "question": "Can someone enable VPN for me?",
            "groups": [
                {
                    "id": "g1",
                    "question": "How do I get VPN access?",
                    "title": "Getting VPN access",
                    "count": 3,
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "relevant": True,
        "question": "Can someone enable VPN for me?",
        "title": "Getting VPN access",
        "groupId": "g1",
        "documents": [],
    }


def test_classify_endpoint_works_without_any_existing_structure(
    client: TestClient,
) -> None:
    """The very first question of a project has no entries to match against."""
    app.dependency_overrides[get_llm] = lambda: _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "title": "Getting VPN access",
            "redacted_question": "How do I get VPN access?",
        }
    )

    response = client.post(
        "/api/v1/insights/faq/classify",
        json={"projectId": _PROJECT, "question": "How do I get VPN access?"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["relevant"] is True
    assert body["groupId"] is None


def test_classify_endpoint_llm_unavailable_returns_503(client: TestClient) -> None:
    class _FailingGenerateLLM(StubLLMClient):
        def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
            raise LLMUnavailableError("local LLM unavailable")

    app.dependency_overrides[get_llm] = lambda: _FailingGenerateLLM()

    response = client.post(
        "/api/v1/insights/faq/classify",
        json={"projectId": _PROJECT, "question": "How do I get VPN access?"},
    )

    assert response.status_code == 503
    assert "local LLM unavailable" in response.json()["detail"]


def test_classify_endpoint_rejects_a_missing_project(client: TestClient) -> None:
    response = client.post(
        "/api/v1/insights/faq/classify", json={"question": "How do I get VPN access?"}
    )

    assert response.status_code == 422


def test_merge_groups_endpoint_returns_a_merge_plan(client: TestClient) -> None:
    app.dependency_overrides[get_llm] = lambda: _ScriptedLLM(
        {"merges": [{"into": "g1", "sources": ["g2"]}]}
    )

    response = client.post(
        "/api/v1/insights/faq/groups/merge",
        json={
            "groups": [
                {"id": "g1", "question": "How do I get VPN access?", "count": 4},
                {"id": "g2", "question": "How do i get vpn access", "count": 1},
                {"id": "g3", "question": "How do I reset my password?", "count": 2},
            ],
            "targetMax": 2,
        },
    )

    assert response.status_code == 200
    assert response.json() == {"merges": [{"into": "g1", "sources": ["g2"]}]}


def test_merge_endpoint_rejects_a_zero_target(client: TestClient) -> None:
    response = client.post(
        "/api/v1/insights/faq/groups/merge",
        json={"groups": [], "targetMax": 0},
    )

    assert response.status_code == 422
