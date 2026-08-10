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
_URL = "/api/v1/onboarding/modules/propose"
_KEY = "deploy-runbook"
_LABEL = "Deploy the service"

_PAYLOAD = {
    "title": "Deploying the service",
    "summary": "How deploys work here.",
    "pages": [
        {
            "kind": "CONTEXT",
            "title": "Why deploys are gated",
            "body": "…",
            "chunk_ids": ["c1"],
        },
        {
            "kind": "LESSON",
            "title": "How the pipeline works",
            "body": "…",
            "chunk_ids": ["c1"],
        },
    ],
    "verification": {
        "prompt": "Walk through a rollback.",
        "rubric": "Names the rollback step.",
    },
}

_REQUEST = {"competency_key": _KEY, "competency_label": _LABEL, "level": "beginner"}


def _store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="runbook.md",
                text="deploy runbook rollback release process",
                embedding=_EMBED,
            )
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


def test_stream_yields_stages_page_items_and_a_done_matching_the_sync_endpoint(
    client: TestClient,
) -> None:
    stream = client.post(f"{_URL}/stream", json=_REQUEST)
    assert stream.status_code == 200, stream.text
    assert stream.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(stream.text)
    types = [e["type"] for e in events]
    assert "stage" in types
    assert "item" in types
    assert types[-1] == "done"

    plain = client.post("/api/v1/onboarding/modules/propose", json=_REQUEST).json()
    assert events[-1]["result"]["module"] == plain["module"]
    assert events[-1]["result"]["status"] == plain["status"] == "proposed"


def test_stream_rejects_an_unknown_level(client: TestClient) -> None:
    response = client.post(f"{_URL}/stream", json={**_REQUEST, "level": "wizard"})

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
        response = TestClient(app).post(f"{_URL}/stream", json=_REQUEST)
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "ollama is down" in events[-1]["message"]
