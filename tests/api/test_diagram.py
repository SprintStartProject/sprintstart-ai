import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_URL = "/api/v1/onboarding/diagram"
_SUBJECT = "how a request reaches the database"

_PAYLOAD = {
    "summary": "A request lands on the controller and ends at the repository.",
    "nodes": [
        {
            "id": "controller",
            "label": "ReportController",
            "kind": "COMPONENT",
            "summary": "Receives the HTTP request.",
            "chunk_ids": ["c1"],
        },
        {
            "id": "repo",
            "label": "ReportRepository",
            "kind": "COMPONENT",
            "summary": "Issues the query.",
            "chunk_ids": ["c2"],
        },
    ],
    "edges": [{"from_id": "controller", "to_id": "repo", "kind": "FLOWS_TO"}],
}


def _store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="ReportController.kt",
                text="ReportController receives the HTTP request and validates it",
                embedding=_EMBED,
                source_url="https://github.com/org/repo/blob/main/ReportController.kt",
            ),
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="ReportRepository.kt",
                text="ReportRepository issues the SQL query against postgres",
                embedding=_EMBED,
            ),
        ]
    )
    return store


@pytest.fixture
def client() -> Generator[TestClient, Any, None]:
    llm = StubLLMClient(generate_response=json.dumps(_PAYLOAD))
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = _store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_assembles_a_diagram_whose_citations_survive_to_the_client(
    client: TestClient,
) -> None:
    response = client.post(_URL, json={"subject": _SUBJECT})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "assembled"
    node = body["diagram"]["nodes"][0]
    assert node["label"] == "ReportController"
    # A hire has to be able to open the source, so the link travels with the box.
    assert node["citations"][0]["source_url"].endswith("ReportController.kt")
    assert body["diagram"]["edges"][0]["from_id"] == "controller"
    assert body["provenance"]["corpus_fingerprint"]


def test_an_empty_corpus_answers_skipped_with_no_diagram() -> None:
    llm = StubLLMClient(generate_response=json.dumps(_PAYLOAD))
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = StubVectorStore
    try:
        response = TestClient(app).post(_URL, json={"subject": _SUBJECT})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "skipped"
    assert response.json()["diagram"] is None


def test_a_diagram_needs_a_subject(client: TestClient) -> None:
    response = client.post(_URL, json={"subject": "   "})

    assert response.status_code == 422


def test_an_overlong_subject_is_clamped_rather_than_rejected(
    client: TestClient,
) -> None:
    """The subject is model-written; a rambling one is a bad question, not an error."""
    response = client.post(_URL, json={"subject": "how does " + "x" * 5000})

    assert response.status_code == 200, response.text


def test_an_unavailable_llm_is_a_503_not_a_fabricated_diagram() -> None:
    class _Down(StubLLMClient):
        def generate(self, messages: object, **kwargs: object) -> str:
            raise LLMUnavailableError("ollama is down")

    llm = _Down()
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = _store
    try:
        response = TestClient(app).post(_URL, json={"subject": _SUBJECT})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503


def test_stream_yields_stages_items_and_a_done_matching_the_sync_endpoint(
    client: TestClient,
) -> None:
    stream = client.post(f"{_URL}/stream", json={"subject": _SUBJECT})
    assert stream.status_code == 200, stream.text
    assert stream.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(stream.text)
    types = [e["type"] for e in events]
    assert "stage" in types
    assert "item" in types
    assert types[-1] == "done"

    plain = client.post(_URL, json={"subject": _SUBJECT}).json()
    assert events[-1]["result"]["diagram"] == plain["diagram"]
    assert events[-1]["result"]["status"] == plain["status"] == "assembled"


def test_stream_needs_a_subject(client: TestClient) -> None:
    response = client.post(f"{_URL}/stream", json={"subject": "  "})

    assert response.status_code == 422


def test_stream_turns_an_llm_outage_into_a_terminal_error_event() -> None:
    class _Down(StubLLMClient):
        def generate(self, messages: object, **kwargs: object) -> str:
            raise LLMUnavailableError("ollama is down")

    llm = _Down()
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = _store
    try:
        response = TestClient(app).post(f"{_URL}/stream", json={"subject": _SUBJECT})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "ollama is down" in events[-1]["message"]
