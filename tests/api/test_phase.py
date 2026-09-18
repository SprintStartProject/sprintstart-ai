import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_URL = "/api/v1/onboarding/phase"

_PAYLOAD = {
    "steps": [
        {
            "title": "Read the README",
            "description": "Start here",
            "tasks": [{"title": "Open the README", "description": ""}],
            "resources": [],
            "estimated_minutes": 10,
            "expected_outcome": "Understand the project",
            "chunk_ids": ["c1"],
        }
    ],
    "questions": [],
}


def _store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="README.md",
                text="the README explains the project and how to run it locally",
                embedding=_EMBED,
                project_ids=("p1",),
            )
        ]
    )
    return store


def _request(**overrides: str) -> dict[str, object]:
    body: dict[str, object] = {
        "phase_title": "Project Overview",
        "phase_prompt": "Generate an overview for a new member.",
        "project_id": "p1",
    }
    body.update(overrides)
    return body


@pytest.fixture
def client() -> Generator[TestClient, Any, None]:
    llm = StubLLMClient(generate_response=json.dumps(_PAYLOAD))
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = _store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_assembles_a_phase_whose_content_survives_to_the_client(
    client: TestClient,
) -> None:
    response = client.post(_URL, json=_request())

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "assembled"
    assert body["steps"][0]["title"] == "Read the README"
    assert body["steps"][0]["tasks"][0]["title"] == "Open the README"
    assert body["chunks_retrieved"] >= 1


def test_streams_item_events_and_a_terminal_done(client: TestClient) -> None:
    response = client.post(_URL + "/stream", json=_request())

    assert response.status_code == 200, response.text
    events = parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert types[0] == "stage"
    assert "item" in types
    assert types[-1] == "done"
    done = events[-1]
    assert done["result"]["steps"][0]["title"] == "Read the README"


def test_an_empty_corpus_answers_skipped_with_no_content() -> None:
    llm = StubLLMClient(generate_response=json.dumps(_PAYLOAD))
    llm.embedding = _EMBED
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = StubVectorStore
    try:
        response = TestClient(app).post(_URL, json=_request())
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "skipped"
    assert body["steps"] == []


def test_a_phase_needs_a_title_and_prompt(client: TestClient) -> None:
    assert client.post(_URL, json=_request(phase_title="   ")).status_code == 422
    assert client.post(_URL, json=_request(phase_prompt="   ")).status_code == 422
    assert client.post(_URL, json=_request(project_id="   ")).status_code == 422
