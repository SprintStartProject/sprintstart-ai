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
_BASE = "/api/v1/onboarding/competency-graph"


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "competencies": [
                    {
                        "key": "kotlin",
                        "label": "Kotlin",
                        "kind": "SKILL",
                        "chunk_ids": ["c1"],
                    }
                ]
            }
        )
    )
    llm.embedding = _EMBED
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="build.gradle.kts",
                text="Kotlin is the primary backend language",
                embedding=_EMBED,
            )
        ]
    )

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), llm, store
    app.dependency_overrides.clear()


def _propose(http: TestClient, **body: Any) -> dict[str, Any]:
    response = http.post(f"{_BASE}/propose", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_propose_returns_grounded_competencies(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    outcome = _propose(http)

    assert outcome["status"] == "proposed"
    assert outcome["competencies"][0]["key"] == "kotlin"
    assert outcome["provenance"]["corpus_fingerprint"] is not None


def test_propose_skips_active_competency(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    outcome = _propose(
        http,
        active_competencies=[{"key": "kotlin", "label": "Kotlin", "kind": "SKILL"}],
    )

    assert outcome["status"] == "skipped"
    assert outcome["competencies"] == []


def test_propose_unchanged_corpus_with_matching_fingerprint_is_a_noop(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    first = _propose(http)
    again = _propose(http, last_fingerprint=first["provenance"]["corpus_fingerprint"])

    assert again["status"] == "unchanged"


def test_propose_stream_yields_a_node_item_and_a_done_matching_the_sync_endpoint(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    stream = http.post(f"{_BASE}/propose/stream", json={})
    assert stream.status_code == 200, stream.text
    assert stream.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(stream.text)
    types = [e["type"] for e in events]
    assert "stage" in types
    assert "item" in types
    assert types[-1] == "done"
    assert events[-1]["result"]["competencies"][0]["key"] == "kotlin"

    # The stream is a view of the same computation: its final result equals what the
    # non-streaming endpoint returns for the same request (fingerprint-timing aside).
    plain = _propose(http)
    assert events[-1]["result"]["status"] == plain["status"] == "proposed"
    assert [c["key"] for c in events[-1]["result"]["competencies"]] == [
        c["key"] for c in plain["competencies"]
    ]


def test_propose_stream_turns_an_llm_outage_into_a_terminal_error_event() -> None:
    class _Down(StubLLMClient):
        def generate(self, messages: object, **kwargs: object) -> str:
            raise LLMUnavailableError("ollama is down")

    llm = _Down()
    llm.embedding = _EMBED
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="build.gradle.kts",
                text="Kotlin is the primary backend language",
                embedding=_EMBED,
            )
        ]
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    try:
        response = TestClient(app).post(f"{_BASE}/propose/stream", json={})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["type"] == "error"
    assert "ollama is down" in events[-1]["message"]
