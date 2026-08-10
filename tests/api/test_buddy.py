"""Route-level contract for the agentic buddy endpoint, including compaction."""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_source_state_store, get_store
from ingestion.source_state_store import SourceStateStore
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_URL = "/api/v1/onboarding/buddy/agent"


@pytest.fixture
def client() -> Generator[TestClient, Any, None]:
    llm = StubLLMClient(generate_response="condensed memory note")
    store = StubVectorStore()
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_source_state_store] = lambda: SourceStateStore(
        ":memory:"
    )
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_the_prior_summary_stands_in_for_the_conversation_older_than_the_window(
    client: TestClient,
) -> None:
    response = client.post(
        _URL,
        json={
            "messages": [
                {"role": "user", "content": "m1"},
                {"role": "user", "content": "m2"},
            ],
            "prior_summary": "old notes",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final"] is True
    # The summary rides the system message the backend carries back verbatim on a
    # resume, and the window it stands in for is sent whole.
    assert body["messages"][0]["role"] == "system"
    assert "old notes" in body["messages"][0]["content"]
    contents = [m["content"] for m in body["messages"]]
    assert "m1" in contents
    assert "m2" in contents


def test_a_turn_never_folds_and_returns_no_summary(client: TestClient) -> None:
    """⚠️ Folding here ran ahead of the answer; it is `/onboarding/buddy/compact` now."""
    response = client.post(
        _URL,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["final"] is True
    assert "updated_summary" not in body
    assert body["messages"][0]["role"] == "system"


def test_compact_endpoint_returns_the_rewritten_note(client: TestClient) -> None:
    response = client.post(
        "/api/v1/onboarding/buddy/compact",
        json={
            "prior_summary": "old notes",
            "folded": [
                {"role": "user", "content": "how do I run the tests?"},
                {"role": "assistant", "content": "uv run pytest"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json() == {"memory": "condensed memory note"}


def test_compact_endpoint_503s_rather_than_returning_an_unchanged_note() -> None:
    """A caller must be able to tell "nothing folded" from "folded to the same words".

    Returning the prior note with a 200 would advance the caller's cursor past
    messages nothing had summarized -- the one way this design loses a transcript.
    """

    class _Unavailable(StubLLMClient):
        def generate(
            self, messages: list[Any], *, temperature: float | None = None
        ) -> str:
            raise LLMUnavailableError("model down")

    app.dependency_overrides[get_llm] = lambda: _Unavailable()
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/onboarding/buddy/compact",
                json={
                    "prior_summary": "old notes",
                    "folded": [{"role": "user", "content": "hello"}],
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503


def test_open_stream_greets_and_carries_no_memory_note() -> None:
    """The greeting arrives as SSE tokens, and the note is nobody's business here.

    ⚠️ It used to ride the terminal event, written by this same model call.
    """
    llm = StubLLMClient(generate_response="Welcome back, Sam!")
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/onboarding/buddy/open/stream",
                json={"memory": "old note", "recent": [], "state": "1 open PR"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    body = response.text
    assert '"type": "token"' in body
    assert "Welcome back, Sam!" in body
    assert '"type": "done"' in body
    # The caller cannot persist a note it is never handed.
    assert '"memory"' not in body
