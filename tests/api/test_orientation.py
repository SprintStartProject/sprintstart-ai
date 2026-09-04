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
_URL = "/api/v1/onboarding/orientation"
_TITLE = "Fix the stale cache header on /api/v1/reports"

_PAYLOAD = {
    "summary": "What you need to change the reports cache header.",
    "sections": [
        {
            "step": "SET_UP",
            "title": "Run it locally",
            "body": "Run `make dev`.",
            "chunk_ids": ["c1"],
        }
    ],
}


def _store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="README.md",
                text="run make dev to start the reports service locally",
                embedding=_EMBED,
                source_url="https://github.com/org/repo/blob/main/README.md",
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


def test_assembles_a_packet_whose_provenance_survives_to_the_client(
    client: TestClient,
) -> None:
    response = client.post(_URL, json={"task_title": _TITLE})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "assembled"
    section = body["packet"]["sections"][0]
    assert section["step"] == "SET_UP"
    # A hire has to be able to open the source, so the link travels with the claim.
    assert section["citations"][0]["source_url"].endswith("README.md")
    assert body["packet"]["sources"][0]["filename"] == "README.md"
    assert body["provenance"]["corpus_fingerprint"]


def test_an_empty_corpus_answers_skipped_with_no_packet() -> None:
    llm = StubLLMClient(generate_response=json.dumps(_PAYLOAD))
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = StubVectorStore
    try:
        response = TestClient(app).post(_URL, json={"task_title": _TITLE})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    assert response.json()["status"] == "skipped"
    assert response.json()["packet"] is None


def test_a_packet_needs_a_task(client: TestClient) -> None:
    response = client.post(_URL, json={"task_title": "   "})

    assert response.status_code == 422


def test_an_unavailable_llm_is_a_503_not_a_fabricated_packet() -> None:
    class _Down(StubLLMClient):
        def generate(self, messages: object, **kwargs: object) -> str:
            raise LLMUnavailableError("ollama is down")

    llm = _Down()
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = _store
    try:
        response = TestClient(app).post(_URL, json={"task_title": _TITLE})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503


def test_stream_yields_stages_items_and_a_done_matching_the_sync_endpoint(
    client: TestClient,
) -> None:
    stream = client.post(f"{_URL}/stream", json={"task_title": _TITLE})
    assert stream.status_code == 200, stream.text
    assert stream.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(stream.text)
    types = [e["type"] for e in events]
    assert "stage" in types
    assert "item" in types
    assert types[-1] == "done"

    # The stream is a view of the same computation: its final result equals what the
    # non-streaming endpoint returns for the same request.
    plain = client.post(_URL, json={"task_title": _TITLE}).json()
    assert events[-1]["result"]["packet"] == plain["packet"]
    assert events[-1]["result"]["status"] == plain["status"] == "assembled"


def test_stream_needs_a_task_title(client: TestClient) -> None:
    response = client.post(f"{_URL}/stream", json={"task_title": "  "})

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
        response = TestClient(app).post(f"{_URL}/stream", json={"task_title": _TITLE})
    finally:
        app.dependency_overrides.clear()

    # A mid-stream outage is a terminal `error` event on a 200 stream, not an HTTP
    # error -- some stages already reached the client before the LLM was called.
    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "ollama is down" in events[-1]["message"]
