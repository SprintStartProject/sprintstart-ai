import json
from collections.abc import Generator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_source_state_store, get_store
from ingestion.source_state_store import SourceStateStore
from llm.base import (
    ChatResult,
    LLMChatStreamEvent,
    LLMStreamEvent,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolSpec,
)
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.conftest import llm_required, parse_sse_events
from tests.stubs.llm import ScriptedLLMClient, StubLLMClient, Turn
from tests.stubs.store import StubVectorStore


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient()
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), llm, store

    app.dependency_overrides.clear()


@pytest.fixture
def real_client() -> Generator[TestClient, Any, None]:
    yield TestClient(app)
    app.dependency_overrides.clear()


_PROJECT = "project-1"


def _parse_events(text: str) -> list[dict[str, object]]:
    return [
        json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: ")
    ]


def test_chat_streams_tokens_and_done(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?", "projectId": _PROJECT},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"


def test_chat_token_event_contains_llm_response(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client

    embedding = [1.0] + [0.0] * 767
    filtered_llm = StubLLMClient(embedding=embedding)
    filtered_llm.generate_response = "Missing designs and flaky CI."
    app.dependency_overrides[get_llm] = lambda: filtered_llm

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="Missing designs and flaky CI.",
                embedding=embedding,
                project_ids=(_PROJECT,),
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "prompt": "What were the blockers?",
            "projectId": _PROJECT,
            "min_score": 0.0,
        },
    )

    token_events = [e for e in parse_sse_events(response.text) if e["type"] == "token"]
    assert len(token_events) == 1
    assert token_events[0]["content"] == "Missing designs and flaky CI."


def test_filtered_chat_forwards_reasoning_without_mixing_it_into_tokens(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767

    class ReasoningLLM(StubLLMClient):
        def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
            yield ReasoningDelta("")
            yield ReasoningDelta("Comparing the selected sources.")
            yield TextDelta("The answer.")

    app.dependency_overrides[get_llm] = lambda: ReasoningLLM(embedding=embedding)
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="The answer.",
                embedding=embedding,
                project_ids=(_PROJECT,),
                source_system="GITHUB",
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "prompt": "What happened?",
            "projectId": _PROJECT,
            "filters": {"source_systems": ["GITHUB"]},
        },
    )

    events = parse_sse_events(response.text)
    assert {
        "type": "reasoning",
        "reasoning": "Comparing the selected sources.",
    } in events
    assert [event["content"] for event in events if event["type"] == "token"] == [
        "The answer."
    ]


def test_chat_emits_citation_when_chunks_exist(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client
    # Use a non-zero embedding so cosine similarity is > 0
    embedding = [1.0] + [0.0] * 767
    script: list[Turn] = [[("retrieve", {"query": "blockers"})]]
    app.dependency_overrides[get_llm] = lambda: ScriptedLLMClient(
        script, embedding=embedding
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=(_PROJECT,),
                start_line=5,
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?", "projectId": _PROJECT},
    )

    events = parse_sse_events(response.text)
    citation_events = [e for e in events if e["type"] == "citation"]
    assert len(citation_events) == 1
    assert citation_events[0]["artifact_id"] == "doc-1"
    assert citation_events[0]["start_line"] == 5
    assert citation_events[0]["start_page"] is None

    tool_uses = [
        {"name": e["name"], "kind": e["kind"]}
        for e in events
        if e["type"] == "tool_use"
    ]
    assert tool_uses == [{"name": "retrieve", "kind": "tool"}]
    assert events[-1] == {"type": "done"}


def test_chat_with_history_succeeds(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/chat",
        json={
            "prompt": "Can you summarize that?",
            "projectId": _PROJECT,
            "context": [
                {"role": "user", "content": "What were the blockers?"},
                {"role": "assistant", "content": "Missing designs and flaky CI."},
            ],
        },
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[-1]["type"] == "done"


def test_chat_missing_question_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post("/api/v1/chat", json={})

    assert response.status_code == 422


def test_chat_missing_project_id_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post("/api/v1/chat", json={"question": "What changed?"})

    assert response.status_code == 422


def test_chat_does_not_cite_another_projects_chunks(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    """The agentic path must stay inside the requesting project."""
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    script: list[Turn] = [[("retrieve", {"query": "blockers"})]]
    app.dependency_overrides[get_llm] = lambda: ScriptedLLMClient(
        script, embedding=embedding
    )

    store.add(
        [
            Chunk(
                id="chunk-own",
                artifact_id="artifact-own",
                filename="retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=(_PROJECT,),
            ),
            Chunk(
                id="chunk-foreign",
                artifact_id="artifact-foreign",
                filename="secret-retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=("project-2",),
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?", "projectId": _PROJECT},
    )

    events = parse_sse_events(response.text)
    citation_events = [e for e in events if e["type"] == "citation"]
    assert [e["artifact_id"] for e in citation_events] == ["artifact-own"]


def test_chat_with_filters_does_not_use_another_projects_chunks(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    """The filtered path applies the project scope on top of the user filters."""
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    store.add(
        [
            Chunk(
                id="chunk-foreign",
                artifact_id="artifact-foreign",
                filename="app.py",
                text="GitHub text",
                embedding=embedding,
                source_system="GITHUB",
                project_ids=("project-2",),
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed in code?",
            "projectId": _PROJECT,
            "filters": {"source_systems": ["GITHUB"]},
        },
    )

    events = _parse_events(response.text)
    token_events = [event for event in events if event["type"] == "token"]
    content = token_events[0]["content"]
    assert isinstance(content, str)
    assert "could not find any matching sources" in content
    assert [event for event in events if event["type"] == "citation"] == []


def test_chat_ignores_chunks_without_a_project(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    """Fail-closed: pre-project-separation chunks are invisible until re-synced."""
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    store.add(
        [
            Chunk(
                id="chunk-legacy",
                artifact_id="artifact-legacy",
                filename="old.md",
                text="GitHub text",
                embedding=embedding,
                source_system="GITHUB",
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed in code?",
            "projectId": _PROJECT,
            "filters": {"source_systems": ["GITHUB"]},
        },
    )

    events = _parse_events(response.text)
    assert [event for event in events if event["type"] == "citation"] == []


def test_chat_llm_unavailable_emits_error_event(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client

    embedding = [1.0] + [0.0] * 767

    class UnavailableLLM(StubLLMClient):
        """Down however the agent reaches it — deciding or streaming."""

        def __init__(self) -> None:
            super().__init__(embedding=embedding)

        def chat(
            self, messages: list[Message], tools: list[ToolSpec] | None = None
        ) -> ChatResult:
            raise LLMUnavailableError("http://localhost:11434")

        def chat_stream(
            self, messages: list[Message], tools: list[ToolSpec] | None = None
        ) -> Iterator[LLMChatStreamEvent]:
            raise LLMUnavailableError("http://localhost:11434")
            yield from ()  # pragma: no cover - makes this a generator

        def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
            raise LLMUnavailableError("http://localhost:11434")

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc-1",
                filename="retro.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=(_PROJECT,),
            )
        ]
    )

    app.dependency_overrides[get_llm] = lambda: UnavailableLLM()

    response = http_client.post(
        "/api/v1/chat",
        json={
            "prompt": "What were the blockers?",
            "projectId": _PROJECT,
            "min_score": 0.0,
        },
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    assert events[0]["type"] == "error"


@pytest.mark.integration
@llm_required
def test_chat_with_real_llm(real_client: TestClient) -> None:
    response = real_client.post(
        "/api/v1/chat",
        json={"prompt": "Reply with one word: hello.", "projectId": _PROJECT},
    )

    assert response.status_code == 200
    events = parse_sse_events(response.text)
    types = [e["type"] for e in events]
    assert "token" in types
    assert types[-1] == "done"


def test_chat_with_filter_no_matching_chunks_returns_fallback(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client

    store.add(
        [
            Chunk(
                id="chunk-upload",
                artifact_id="artifact-upload",
                filename="doc.md",
                text="Upload text",
                embedding=[1.0] + [0.0] * 767,
                project_ids=(_PROJECT,),
                source_system="UPLOAD",
            )
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed in code?",
            "projectId": _PROJECT,
            "filters": {"source_systems": ["GITHUB"]},
        },
    )

    assert response.status_code == 200

    events = parse_sse_events(response.text)
    token_events = [event for event in events if event["type"] == "token"]
    assert len(token_events) == 1
    content = token_events[0]["content"]
    assert isinstance(content, str)
    assert "could not find any matching sources" in content
    assert events[-1]["type"] == "done"


def test_chat_with_source_filter_uses_matching_chunks(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client

    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    store.add(
        [
            Chunk(
                id="chunk-upload",
                artifact_id="artifact-upload",
                filename="doc.md",
                text="Upload text",
                embedding=embedding,
                project_ids=(_PROJECT,),
                source_system="UPLOAD",
            ),
            Chunk(
                id="chunk-github",
                artifact_id="artifact-github",
                filename="app.py",
                text="GitHub text",
                embedding=embedding,
                project_ids=(_PROJECT,),
                source_system="GITHUB",
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed in code?",
            "projectId": _PROJECT,
            "filters": {"source_systems": ["GITHUB"]},
        },
    )

    assert response.status_code == 200

    events = _parse_events(response.text)
    citation_events = [event for event in events if event["type"] == "citation"]

    assert len(citation_events) == 1
    assert citation_events[0]["artifact_id"] == "artifact-github"


def test_chat_without_chunks_returns_fallback_without_llm_hallucination(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, _ = client

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed?",
            "projectId": _PROJECT,
            "filters": {"source_systems": ["GITHUB"]},
        },
    )

    assert response.status_code == 200

    events = _parse_events(response.text)
    citation_events = [event for event in events if event["type"] == "citation"]

    token_events = [event for event in events if event["type"] == "token"]
    assert len(token_events) == 1
    content = token_events[0]["content"]
    assert isinstance(content, str)
    assert "could not find any matching sources" in content
    assert citation_events == []
    assert events[-1]["type"] == "done"


def test_chat_applies_source_exclusions_in_unfiltered_path(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    """Regression test: /chat must honor disabled connectors/sources even
    when no explicit retrieval filters are given (issue found while
    auditing the connector/source enable-disable feature — the route built
    its ChatOrchestrator without exclusions, so disabling a connector had no
    effect on live chat).
    """
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767

    script: list[Turn] = [[("retrieve", {"query": "blockers"})]]
    app.dependency_overrides[get_llm] = lambda: ScriptedLLMClient(
        script, embedding=embedding
    )

    source_state = SourceStateStore(path=":memory:")
    source_state.set_connector_enabled("github", enabled=False)
    app.dependency_overrides[get_source_state_store] = lambda: source_state

    store.add(
        [
            Chunk(
                id="chunk-excluded",
                artifact_id="artifact-excluded",
                filename="excluded.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=(_PROJECT,),
                connector_id="github",
                connector_source_id="owner/repo",
            ),
            Chunk(
                id="chunk-included",
                artifact_id="artifact-included",
                filename="included.md",
                text="Missing designs blocked the auth feature.",
                embedding=embedding,
                project_ids=(_PROJECT,),
                connector_id="jira",
                connector_source_id="PROJ",
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={"prompt": "What were the blockers?", "projectId": _PROJECT},
    )

    events = parse_sse_events(response.text)
    citation_events = [e for e in events if e["type"] == "citation"]
    assert [e["artifact_id"] for e in citation_events] == ["artifact-included"]


def test_chat_applies_source_exclusions_with_retrieval_filters(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http_client, _, store = client
    embedding = [1.0] + [0.0] * 767
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(embedding=embedding)

    source_state = SourceStateStore(path=":memory:")
    source_state.set_sources_enabled("github", {"owner/repo": False})
    app.dependency_overrides[get_source_state_store] = lambda: source_state

    store.add(
        [
            Chunk(
                id="chunk-excluded",
                artifact_id="artifact-excluded",
                filename="excluded.py",
                text="GitHub text",
                embedding=embedding,
                project_ids=(_PROJECT,),
                source_system="GITHUB",
                connector_id="github",
                connector_source_id="owner/repo",
            ),
            Chunk(
                id="chunk-included",
                artifact_id="artifact-included",
                filename="included.py",
                text="GitHub text",
                embedding=embedding,
                project_ids=(_PROJECT,),
                source_system="GITHUB",
                connector_id="github",
                connector_source_id="owner/other-repo",
            ),
        ]
    )

    response = http_client.post(
        "/api/v1/chat",
        json={
            "question": "What changed in code?",
            "projectId": _PROJECT,
            "filters": {"source_systems": ["GITHUB"]},
        },
    )

    assert response.status_code == 200
    events = _parse_events(response.text)
    citation_events = [event for event in events if event["type"] == "citation"]
    assert [e["artifact_id"] for e in citation_events] == ["artifact-included"]
