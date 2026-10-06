"""Route-level contract for the agentic buddy endpoint, including compaction."""

from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_source_state_store, get_store
from ingestion.source_state_store import SourceStateStore
from llm.errors import LLMUnavailableError
from rag.filters import NO_FILTERED_RESULTS_MESSAGE
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import ScriptedLLMClient, StubLLMClient
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
    # Fenced (`onboarding.query_fence`), so each is inside its marker lines.
    assert any("\nm1\n" in c for c in contents)
    assert any("\nm2\n" in c for c in contents)


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
    persona = response.json()["messages"][0]["content"]
    assert "This turn you can only search" in persona


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


def _scripted_client(llm: ScriptedLLMClient) -> Generator[TestClient, Any, None]:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="auth.md",
                text="the login handler lives in auth.py",
                embedding=[0.0] * 768,
                source_system="GITHUB",
                project_ids=("p1",),
            )
        ]
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_source_state_store] = lambda: SourceStateStore(
        ":memory:"
    )
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def test_filters_reach_the_searches_and_an_empty_result_is_said_plainly() -> None:
    llm = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login handler"]})]], answer="ungrounded"
    )
    for client in _scripted_client(llm):
        response = client.post(
            _URL,
            json={
                "messages": [{"role": "user", "content": "where is login?"}],
                "capabilities_enabled": False,
                "filters": {"source_systems": ["jira"]},
                "project_ids": ["p1"],
            },
        )

    assert response.status_code == 200
    assert response.json()["text"] == NO_FILTERED_RESULTS_MESSAGE


def test_empty_source_systems_mean_all() -> None:
    llm = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login handler"]})]], answer="In auth.py."
    )
    for client in _scripted_client(llm):
        response = client.post(
            _URL,
            json={
                "messages": [{"role": "user", "content": "where is login?"}],
                "capabilities_enabled": False,
                "filters": {"source_systems": []},
                "project_ids": ["p1"],
            },
        )

    assert response.json()["text"] == "In auth.py."
    assert [c["artifact_id"] for c in response.json()["citations"]] == ["a1"]


def test_a_disabled_connector_is_neither_searched_nor_cited() -> None:
    """Regression: disabled connectors/sources must not reach the buddy.

    The route passes `source_state.get_exclusions()` into every turn, so a
    disabled connector's chunks must be neither retrieved nor cited even when
    the request carries no source/time narrowing of its own.
    """
    embedding = [1.0] + [0.0] * 767
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-excluded",
                artifact_id="artifact-excluded",
                filename="excluded.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=("p1",),
                connector_id="github",
                connector_source_id="owner/repo",
            ),
            Chunk(
                id="chunk-included",
                artifact_id="artifact-included",
                filename="included.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=("p1",),
                connector_id="jira",
                connector_source_id="PROJ",
            ),
        ]
    )

    source_state = SourceStateStore(path=":memory:")
    source_state.set_connector_enabled("github", enabled=False)

    llm = ScriptedLLMClient(
        turns=[[("search_docs", {"query": "blockers"})]], embedding=embedding
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_source_state_store] = lambda: source_state
    try:
        client = TestClient(app)
        response = client.post(
            _URL,
            json={
                "messages": [{"role": "user", "content": "What were the blockers?"}],
                "project_ids": ["p1"],
            },
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert [c["artifact_id"] for c in response.json()["citations"]] == [
        "artifact-included"
    ]


def _project_scoped_client(llm: ScriptedLLMClient) -> Generator[TestClient, Any, None]:
    """Three identical chunks: this project's, another project's, and no project's."""
    embedding = [1.0] + [0.0] * 767
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"chunk-{name}",
                artifact_id=f"artifact-{name}",
                filename=f"{name}-retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                source_system="GITHUB",
                project_ids=project_ids,
            )
            for name, project_ids in (
                ("own", ("p1",)),
                ("foreign", ("p2",)),
                ("orphan", ()),
            )
        ]
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_source_state_store] = lambda: SourceStateStore(
        ":memory:"
    )
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


_SEARCH_DOCS_CALL = ("search_docs", {"query": "blockers"})
_GREP_CALL = ("grep", {"patterns": ["Missing designs"]})


@pytest.mark.parametrize(
    ("call", "capabilities_enabled"),
    [(_SEARCH_DOCS_CALL, True), (_GREP_CALL, False)],
    ids=["search_docs", "grep"],
)
@pytest.mark.parametrize(
    "filters", [None, {"source_systems": ["github"]}], ids=["unfiltered", "filtered"]
)
def test_a_turn_stays_inside_the_requested_projects(
    call: tuple[str, dict[str, Any]],
    capabilities_enabled: bool,
    filters: dict[str, Any] | None,
) -> None:
    """Regression: another project's material, and material belonging to no
    project, must neither reach the model nor be cited.

    The scope is the route's to forward: `run_agent_turn` without `project_ids`
    searches everything, so a route that dropped it would leak every project.
    """
    llm = ScriptedLLMClient(turns=[[call]], embedding=[1.0] + [0.0] * 767)
    for client in _project_scoped_client(llm):
        response = client.post(
            _URL,
            json={
                "messages": [{"role": "user", "content": "What were the blockers?"}],
                "capabilities_enabled": capabilities_enabled,
                "filters": filters,
                "project_ids": ["p1"],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert [c["artifact_id"] for c in body["citations"]] == ["artifact-own"]
    seen = " ".join(m["content"] for m in body["messages"] if m["role"] == "tool")
    assert "own-retro.md" in seen
    assert "foreign-retro.md" not in seen
    assert "orphan-retro.md" not in seen


@pytest.mark.parametrize(
    ("call", "capabilities_enabled"),
    [(_SEARCH_DOCS_CALL, True), (_GREP_CALL, False)],
    ids=["search_docs", "grep"],
)
def test_an_empty_project_scope_admits_nothing(
    call: tuple[str, dict[str, Any]], capabilities_enabled: bool
) -> None:
    llm = ScriptedLLMClient(turns=[[call]], embedding=[1.0] + [0.0] * 767)
    for client in _project_scoped_client(llm):
        response = client.post(
            _URL,
            json={
                "messages": [{"role": "user", "content": "What were the blockers?"}],
                "capabilities_enabled": capabilities_enabled,
                "project_ids": [],
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["citations"] == []
    seen = " ".join(m["content"] for m in body["messages"] if m["role"] == "tool")
    assert "retro.md" not in seen


def test_reasoning_is_returned_and_never_carried_in_messages() -> None:
    llm = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login"]})]],
        reasoning="grep for it",
        reasoning_details=[{"type": "reasoning.text", "text": "grep for it"}],
    )
    for client in _scripted_client(llm):
        response = client.post(
            _URL,
            json={
                "messages": [{"role": "user", "content": "login?"}],
                "capabilities_enabled": False,
            },
        )

    body = response.json()
    assert body["reasoning"] == ["grep for it"]
    assert all(
        set(m) <= {"role", "content", "tool_calls", "tool_call_id"}
        for m in body["messages"]
    )
    assert "grep for it" not in str(body["messages"])


def test_a_caller_that_sends_no_filters_gets_todays_turn(client: TestClient) -> None:
    response = client.post(
        _URL, json={"messages": [{"role": "user", "content": "hello"}]}
    )

    assert response.status_code == 200
    assert response.json()["reasoning"] == []


# --- streaming agent turn -----------------------------------------------------

_STREAM_URL = "/api/v1/onboarding/buddy/agent/stream"


def test_the_stream_endpoint_emits_progress_then_the_buffered_response() -> None:
    llm = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login"]})]], answer="In auth.py."
    )
    payload = {
        "messages": [{"role": "user", "content": "where is login?"}],
        "capabilities_enabled": False,
        "project_ids": ["p1"],
    }
    for client in _scripted_client(llm):
        streamed = client.post(_STREAM_URL, json=payload)
    llm_again = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login"]})]], answer="In auth.py."
    )
    for client in _scripted_client(llm_again):
        buffered = client.post(_URL, json=payload)

    assert streamed.status_code == 200
    assert streamed.headers["content-type"].startswith("text/event-stream")
    events = parse_sse_events(streamed.text)
    kinds = [e["type"] for e in events]
    assert kinds == ["tool_use", "token", "result"]
    assert events[0]["name"] == "grep"
    assert events[0]["arguments"] == {"patterns": ["login"]}
    assert events[1]["content"] == "In auth.py."
    result = {k: v for k, v in events[-1].items() if k != "type"}
    # The terminal event is the buffered endpoint's response, field for field.
    assert result == buffered.json()
    assert result["citations"] == [
        {"artifact_id": "a1", "start_line": None, "start_page": None}
    ]


def test_the_stream_endpoint_emits_reasoning_deltas() -> None:
    llm = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login"]})]], reasoning="grep for it"
    )
    for client in _scripted_client(llm):
        response = client.post(
            _STREAM_URL,
            json={
                "messages": [{"role": "user", "content": "login?"}],
                "capabilities_enabled": False,
            },
        )

    events = parse_sse_events(response.text)
    assert {"type": "reasoning", "content": "grep for it"} in events
    assert events[-1]["reasoning"] == ["grep for it"]


def test_the_stream_endpoint_reports_an_unavailable_model_as_an_error_event() -> None:
    class _Down(StubLLMClient):
        def chat_stream(self, *args: Any, **kwargs: Any) -> Any:
            raise LLMUnavailableError("model down")

    app.dependency_overrides[get_llm] = lambda: _Down()
    app.dependency_overrides[get_store] = lambda: StubVectorStore()
    app.dependency_overrides[get_source_state_store] = lambda: SourceStateStore(
        ":memory:"
    )
    try:
        response = TestClient(app).post(
            _STREAM_URL, json={"messages": [{"role": "user", "content": "hi"}]}
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert [e["type"] for e in events] == ["error"]
    assert "model down" in events[0]["message"]


def test_the_stream_endpoint_applies_the_filters_to_its_answer() -> None:
    llm = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login handler"]})]], answer="ungrounded"
    )
    for client in _scripted_client(llm):
        response = client.post(
            _STREAM_URL,
            json={
                "messages": [{"role": "user", "content": "where is login?"}],
                "capabilities_enabled": False,
                "filters": {"source_systems": ["jira"]},
                "project_ids": ["p1"],
            },
        )

    events = parse_sse_events(response.text)
    streamed = "".join(e["content"] for e in events if e["type"] == "token")
    assert streamed == NO_FILTERED_RESULTS_MESSAGE
    assert events[-1]["text"] == NO_FILTERED_RESULTS_MESSAGE
