import json
import threading

import pytest
from pydantic import BaseModel

from agents.tools.base import Tool, ToolRegistry, ToolResult, run_tool_calls
from agents.tools.grep import GrepTool
from agents.tools.retrieve import RetrieveTool
from llm.base import ToolCall
from rag.source_filter import SourceExclusions
from rag.types import Chunk, RetrievalFilters
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def _chunk(
    chunk_id: str,
    filename: str,
    text: str,
    embedding: list[float],
    connector_id: str | None = None,
    connector_source_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=chunk_id,
        artifact_id="doc-1",
        filename=filename,
        text=text,
        embedding=embedding,
        connector_id=connector_id,
        connector_source_id=connector_source_id,
    )


def test_retrieve_tool_returns_matching_chunks() -> None:
    embedding = [1.0] + [0.0] * 767
    store = StubVectorStore()
    store.add([_chunk("c1", "retro.md", "missing designs blocked auth", embedding)])
    llm = StubLLMClient(embedding=embedding)

    result = RetrieveTool(llm, store).execute({"query": "blockers"})

    assert isinstance(result, ToolResult)
    assert [c.id for c in result.chunks] == ["c1"]
    assert "1 chunk" in result.summary


def test_retrieve_tool_excludes_disabled_source() -> None:
    embedding = [1.0] + [0.0] * 767
    store = StubVectorStore()
    store.add(
        [
            _chunk(
                "c1",
                "retro.md",
                "missing designs blocked auth",
                embedding,
                connector_id="github",
                connector_source_id="owner/repo",
            )
        ]
    )
    llm = StubLLMClient(embedding=embedding)
    exclusions = SourceExclusions(sources=frozenset({("github", "owner/repo")}))

    result = RetrieveTool(llm, store, exclusions=exclusions).execute(
        {"query": "blockers"}
    )

    assert result.chunks == []


def test_retrieve_tool_rejects_bad_args() -> None:
    tool = RetrieveTool(StubLLMClient(), StubVectorStore())

    result = tool.execute({"wrong": "field"})

    assert result.chunks == []
    assert "Invalid arguments" in result.summary


def test_grep_tool_matches_substring_case_insensitively() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk("c1", "a.py", "def parse_config(): ...", [0.0] * 768),
            _chunk("c2", "b.py", "unrelated text", [0.0] * 768),
        ]
    )

    result = GrepTool(store).execute({"patterns": "PARSE_CONFIG"})

    assert [c.id for c in result.chunks] == ["c1"]


def test_grep_tool_excludes_disabled_connector() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk(
                "c1",
                "a.py",
                "def parse_config(): ...",
                [0.0] * 768,
                connector_id="github",
                connector_source_id="owner/repo",
            )
        ]
    )
    exclusions = SourceExclusions(connectors=frozenset({"github"}))

    result = GrepTool(store, exclusions=exclusions).execute(
        {"patterns": "PARSE_CONFIG"}
    )

    assert result.chunks == []


def test_grep_tool_coerces_single_string_pattern() -> None:
    store = StubVectorStore()
    store.add([_chunk("c1", "a.py", "token here", [0.0] * 768)])

    result = GrepTool(store).execute({"patterns": "token"})

    assert len(result.chunks) == 1


def test_retrieve_tool_includes_immediate_neighbours() -> None:
    embedding = [1.0] + [0.0] * 767
    unrelated = [0.0, 1.0] + [0.0] * 766
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{position}",
                artifact_id="doc-1",
                filename="auth.py",
                text="target symbol" if position == 1 else f"context {position}",
                embedding=embedding if position == 1 else unrelated,
                position=position,
            )
            for position in range(3)
        ]
    )

    result = RetrieveTool(StubLLMClient(embedding=embedding), store).execute(
        {"query": "target symbol"}
    )

    assert [chunk.id for chunk in result.chunks] == ["c1", "c0", "c2"]


def test_grep_tool_includes_immediate_neighbours() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{position}",
                artifact_id="doc-1",
                filename="auth.py",
                text="def parse_config():" if position == 1 else f"context {position}",
                embedding=[0.0] * 768,
                position=position,
            )
            for position in range(3)
        ]
    )

    result = GrepTool(store).execute({"patterns": "parse_config"})

    assert [chunk.id for chunk in result.chunks] == ["c1", "c0", "c2"]
    assert [chunk.id for chunk in result.matched_chunks()] == ["c1"]


def test_grep_context_expansion_preserves_source_and_time_filtered_hit() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c0",
                artifact_id="doc-1",
                filename="auth.py",
                text="context",
                embedding=[0.0] * 768,
                position=0,
                source_system="GITHUB",
                created_at="2025-06-01T00:00:00Z",
            ),
            Chunk(
                id="c1",
                artifact_id="doc-1",
                filename="auth.py",
                text="def parse_config():",
                embedding=[0.0] * 768,
                position=1,
                source_system="GITHUB",
                created_at="2025-06-01T00:00:00Z",
            ),
        ]
    )

    result = GrepTool(
        store,
        filters=RetrievalFilters(
            source_systems=["GITHUB"],
            time_from="2025-01-01T00:00:00Z",
            time_to="2025-12-31T23:59:59Z",
        ),
    ).execute({"patterns": "parse_config"})

    assert [chunk.id for chunk in result.chunks] == ["c1", "c0"]
    assert [chunk.id for chunk in result.matched_chunks()] == ["c1"]


class _NoArgs(BaseModel):
    pass


class _FakeTool(Tool[_NoArgs]):
    name = "fake"
    description = "does nothing"
    args_model = _NoArgs

    def run(self, args: _NoArgs) -> ToolResult:  # noqa: ARG002
        return ToolResult.empty("ran")


def test_tool_spec_exposes_json_schema() -> None:
    spec = GrepTool(StubVectorStore()).tool_spec()

    assert spec["name"] == "grep"
    assert spec["description"]
    assert "patterns" in json.dumps(spec["parameters"])


def test_registry_specs_lists_each_tool() -> None:
    registry = ToolRegistry([GrepTool(StubVectorStore())])

    names = [spec["name"] for spec in registry.specs()]

    assert names == ["grep"]


def test_registry_dispatches_by_name() -> None:
    registry = ToolRegistry([_FakeTool()])

    assert registry.names() == frozenset({"fake"})
    assert registry.execute("fake", {}).summary == "ran"


def test_registry_unknown_tool_returns_empty_result() -> None:
    registry = ToolRegistry([_FakeTool()])

    result = registry.execute("missing", {})

    assert result.chunks == []
    assert "Unknown tool" in result.summary


def test_registry_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="Duplicate tool name"):
        ToolRegistry([_FakeTool(), _FakeTool()])


def test_a_call_that_never_ran_is_flagged_as_an_error() -> None:
    """So no caller's "nothing matched" wording can hide why it did not run."""
    registry = ToolRegistry([_FakeTool()])

    assert registry.execute("missing", {}).is_error is True
    assert GrepTool(StubVectorStore()).execute({"wrong": "field"}).is_error is True
    assert _FakeTool().execute({}).is_error is False


def test_tool_calls_in_one_step_execute_concurrently() -> None:
    """Two searches in one step must overlap, not queue behind each other.

    The barrier is the assertion: it only releases once both tools are inside
    it at the same time, so a serial implementation deadlocks and trips the
    timeout instead of quietly taking twice as long.
    """
    barrier = threading.Barrier(2, timeout=5)

    class _BlockingTool(Tool[_NoArgs]):
        args_model = _NoArgs

        def __init__(self, name: str) -> None:
            self.name = name
            self.description = "blocks until its partner arrives"

        def run(self, args: _NoArgs) -> ToolResult:  # noqa: ARG002
            barrier.wait()
            return ToolResult(summary=f"{self.name} ran")

    registry = ToolRegistry([_BlockingTool("first"), _BlockingTool("second")])

    results = run_tool_calls(
        registry,
        [
            ToolCall(id="a", name="first", arguments={}),
            ToolCall(id="b", name="second", arguments={}),
        ],
    )

    assert [r.summary for r in results] == ["first ran", "second ran"]


def test_tool_results_come_back_in_call_order_not_completion_order() -> None:
    """What the model sees must not depend on which search finished first."""
    started = threading.Event()

    class _SlowTool(Tool[_NoArgs]):
        name = "slow"
        description = "finishes last"
        args_model = _NoArgs

        def run(self, args: _NoArgs) -> ToolResult:  # noqa: ARG002
            started.wait(timeout=5)
            return ToolResult(summary="slow ran")

    class _FastTool(Tool[_NoArgs]):
        name = "fast"
        description = "finishes first"
        args_model = _NoArgs

        def run(self, args: _NoArgs) -> ToolResult:  # noqa: ARG002
            started.set()
            return ToolResult(summary="fast ran")

    registry = ToolRegistry([_SlowTool(), _FastTool()])

    results = run_tool_calls(
        registry,
        [
            ToolCall(id="a", name="slow", arguments={}),
            ToolCall(id="b", name="fast", arguments={}),
        ],
    )

    # "fast" completed first; the results still follow the order requested.
    assert [r.summary for r in results] == ["slow ran", "fast ran"]


# --- project scoping ---------------------------------------------------------


def _project_chunk(chunk_id: str, project_ids: tuple[str, ...]) -> Chunk:
    return Chunk(
        id=chunk_id,
        artifact_id=f"doc-{chunk_id}",
        filename=f"{chunk_id}.md",
        text="missing designs blocked auth",
        embedding=[1.0] + [0.0] * 767,
        project_ids=project_ids,
    )


def _project_store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            _project_chunk("own", ("project-1",)),
            _project_chunk("foreign", ("project-2",)),
            _project_chunk("legacy", ()),
        ]
    )
    return store


def test_retrieve_tool_only_returns_the_requested_projects_chunks() -> None:
    store = _project_store()
    llm = StubLLMClient(embedding=[1.0] + [0.0] * 767)

    result = RetrieveTool(
        llm, store, filters=RetrievalFilters(project_id="project-1")
    ).execute({"query": "blockers"})

    assert isinstance(result, ToolResult)
    assert [c.id for c in result.chunks] == ["own"]


def test_grep_tool_only_returns_the_requested_projects_chunks() -> None:
    """Regression: grep scans the corpus in memory, bypassing the store query."""
    result = GrepTool(
        _project_store(), filters=RetrievalFilters(project_id="project-1")
    ).execute({"patterns": ["missing designs"]})

    assert isinstance(result, ToolResult)
    assert [c.id for c in result.chunks] == ["own"]
