import json

import pytest
from pydantic import BaseModel

from agents.tools.base import Tool, ToolRegistry, ToolResult
from agents.tools.fetch_file import FetchFileTool
from agents.tools.grep import GrepTool
from agents.tools.retrieve import RetrieveTool
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


def test_fetch_file_tool_matches_by_name_and_stem() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk("c1", "guide.md", "part one", [0.0] * 768),
            _chunk("c2", "guide.md", "part two", [0.0] * 768),
            _chunk("c3", "other.md", "nope", [0.0] * 768),
        ]
    )

    by_name = FetchFileTool(store).execute({"filename": "guide.md"})
    by_stem = FetchFileTool(store).execute({"filename": "guide"})

    assert {c.id for c in by_name.chunks} == {"c1", "c2"}
    assert {c.id for c in by_stem.chunks} == {"c1", "c2"}


def test_fetch_file_tool_excludes_disabled_source() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk(
                "c1",
                "guide.md",
                "part one",
                [0.0] * 768,
                connector_id="github",
                connector_source_id="owner/repo",
            )
        ]
    )
    exclusions = SourceExclusions(sources=frozenset({("github", "owner/repo")}))

    result = FetchFileTool(store, exclusions=exclusions).execute(
        {"filename": "guide.md"}
    )

    assert result.chunks == []


def test_fetch_file_with_extension_does_not_match_other_extensions() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk("c1", "auth.py", "code", [0.0] * 768),
            _chunk("c2", "auth.md", "docs", [0.0] * 768),
        ]
    )

    with_ext = FetchFileTool(store).execute({"filename": "auth.py"})
    bare = FetchFileTool(store).execute({"filename": "auth"})

    assert {c.id for c in with_ext.chunks} == {"c1"}
    assert {c.id for c in bare.chunks} == {"c1", "c2"}


class _NoArgs(BaseModel):
    pass


class _FakeTool(Tool[_NoArgs]):
    name = "fake"
    description = "does nothing"
    args_model = _NoArgs

    def run(self, args: _NoArgs) -> ToolResult:  # noqa: ARG002
        return ToolResult.empty("ran")


def test_tool_spec_exposes_json_schema() -> None:
    spec = FetchFileTool(StubVectorStore()).tool_spec()

    assert spec["name"] == "fetch_file"
    assert spec["description"]
    assert "filename" in json.dumps(spec["parameters"])


def test_registry_specs_lists_each_tool() -> None:
    registry = ToolRegistry([FetchFileTool(StubVectorStore())])

    names = [spec["name"] for spec in registry.specs()]

    assert names == ["fetch_file"]


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


def test_fetch_file_tool_only_returns_the_requested_projects_chunks() -> None:
    result = FetchFileTool(
        _project_store(), filters=RetrievalFilters(project_id="project-1")
    ).execute({"filename": "foreign.md"})

    assert isinstance(result, ToolResult)
    assert result.chunks == []
