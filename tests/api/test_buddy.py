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
    """Folding here ran ahead of the answer; it is `/onboarding/buddy/compact` now."""
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
        client = TestClient(app)
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

    The terminal event carries no note: folding is a separate call.
    """
    llm = StubLLMClient(generate_response="Welcome back, Sam!")
    app.dependency_overrides[get_llm] = lambda: llm
    try:
        client = TestClient(app)
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


def test_team_mode_reaches_the_persona_the_backend_carries_back(
    client: TestClient,
) -> None:
    response = client.post(
        _URL,
        json={
            "messages": [{"role": "user", "content": "who needs me?"}],
            "team_mode": True,
        },
    )

    assert response.status_code == 200
    persona = response.json()["messages"][0]["content"]
    assert "manager of one project" in persona


def test_capabilities_off_reaches_the_persona(client: TestClient) -> None:
    response = client.post(
        _URL,
        json={
            "messages": [{"role": "user", "content": "how does deployment work?"}],
            "capabilities_enabled": False,
        },
    )

    assert response.status_code == 200
    assert "`search_docs` and nothing else" in response.json()["messages"][0]["content"]


def test_omitting_both_modes_is_the_hire_mentor(client: TestClient) -> None:
    """A caller that has never heard of either field gets exactly today's buddy."""
    response = client.post(
        _URL,
        json={"messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 200
    persona = response.json()["messages"][0]["content"]
    assert "the tutor who guides a new hire" in persona
    assert "manager of one project" not in persona


def test_open_stream_takes_team_mode_and_greets_a_manager() -> None:
    """The stub echoes one fixed answer, so what the flag changes here is the prompt
    it was asked with -- which is the part the route is responsible for threading."""
    recorded: list[str] = []

    class _RecordingLLM(StubLLMClient):
        def stream(self, messages: list[Any]) -> Any:
            recorded.append(str(messages[0]["content"]))
            return super().stream(messages)

    app.dependency_overrides[get_llm] = lambda: _RecordingLLM(
        generate_response="Two people are waiting on a review."
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/onboarding/buddy/open/stream",
            json={
                "memory": None,
                "recent": [],
                "state": "Who needs attention: ...",
                "team_mode": True,
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "Two people are waiting on a review." in response.text
    assert "project's manager" in recorded[0]


def test_open_stream_without_team_mode_is_the_hire_greeting() -> None:
    recorded: list[str] = []

    class _RecordingLLM(StubLLMClient):
        def stream(self, messages: list[Any]) -> Any:
            recorded.append(str(messages[0]["content"]))
            return super().stream(messages)

    app.dependency_overrides[get_llm] = lambda: _RecordingLLM(
        generate_response="Welcome back, Sam!"
    )
    try:
        client = TestClient(app)
        response = client.post(
            "/api/v1/onboarding/buddy/open/stream",
            json={"memory": None, "recent": [], "state": "1 open PR"},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert "greeting a new hire" in recorded[0]
    assert "project's manager" not in recorded[0]
