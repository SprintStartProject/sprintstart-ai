from ingestion.mapper import to_chunk
from ingestion.models import ParsedChunk
from rag.filters import (
    decode_project_ids,
    encode_project_ids,
    matches_retrieval_filters,
    where_filter_for_chroma,
)
from rag.types import Chunk, RetrievalFilters


def test_source_system_filter_matches_allowed_system() -> None:
    chunk = Chunk(
        id="chunk-1",
        artifact_id="artifact-1",
        filename="app.py",
        text="Code",
        embedding=[1.0, 0.0],
        kind="code",
        source_system="GITHUB",
    )

    assert matches_retrieval_filters(
        chunk,
        RetrievalFilters(source_systems=["GITHUB", "JIRA"]),
    )
    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(source_systems=["UPLOAD"]),
    )


def test_source_timestamp_preferred_over_indexed_at() -> None:
    chunk = to_chunk(
        ParsedChunk(
            content="Old issue indexed today",
            kind="text",
            metadata={
                "filename": "issue-1.md",
                "indexed_at": "2026-06-01T00:00:00Z",
            },
        ),
        artifact_id="issue-1",
        embedding=[1.0, 0.0],
        artifact_type="ISSUE",
        source_updated_at="2025-01-01T00:00:00Z",
        source_system="JIRA",
    )

    assert chunk.created_at == "2025-01-01T00:00:00Z"
    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(time_from="2026-01-01T00:00:00Z"),
    )


def test_source_system_and_time_filters_are_combined_with_and() -> None:
    chunk = Chunk(
        id="chunk-1",
        artifact_id="artifact-1",
        filename="app.py",
        text="Code",
        embedding=[1.0, 0.0],
        kind="code",
        source_system="GITHUB",
        created_at="2026-03-01T00:00:00Z",
    )

    assert matches_retrieval_filters(
        chunk,
        RetrievalFilters(
            source_systems=["GITHUB"],
            time_from="2026-01-01T00:00:00Z",
            time_to="2026-07-01T00:00:00Z",
        ),
    )

    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(
            source_systems=["JIRA"],
            time_from="2026-01-01T00:00:00Z",
            time_to="2026-07-01T00:00:00Z",
        ),
    )

    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(
            source_systems=["GITHUB"],
            time_from="2026-04-01T00:00:00Z",
            time_to="2026-07-01T00:00:00Z",
        ),
    )


def _chunk(project_ids: tuple[str, ...]) -> Chunk:
    return Chunk(
        id="chunk-1",
        artifact_id="artifact-1",
        filename="app.py",
        text="Code",
        embedding=[1.0, 0.0],
        kind="code",
        source_system="GITHUB",
        project_ids=project_ids,
    )


def test_project_filter_matches_only_members_of_that_project() -> None:
    chunk = _chunk(("project-1", "project-2"))

    assert matches_retrieval_filters(chunk, RetrievalFilters(project_id="project-1"))
    assert matches_retrieval_filters(chunk, RetrievalFilters(project_id="project-2"))
    assert not matches_retrieval_filters(chunk, RetrievalFilters(project_id="other"))


def test_project_filter_is_fail_closed_for_chunks_without_a_project() -> None:
    """Pre-project-separation chunks must not leak into any project."""
    assert not matches_retrieval_filters(
        _chunk(()), RetrievalFilters(project_id="project-1")
    )


def test_project_filter_combines_with_source_system_filter() -> None:
    chunk = _chunk(("project-1",))

    assert not matches_retrieval_filters(
        chunk,
        RetrievalFilters(project_id="project-1", source_systems=["UPLOAD"]),
    )
    assert matches_retrieval_filters(
        chunk,
        RetrievalFilters(project_id="project-1", source_systems=["GITHUB"]),
    )


def test_where_filter_for_chroma_uses_the_project_marker_key() -> None:
    assert where_filter_for_chroma(RetrievalFilters(project_id="project-1")) == {
        "project:project-1": {"$eq": True}
    }


def test_where_filter_for_chroma_ands_project_with_other_conditions() -> None:
    where = where_filter_for_chroma(
        RetrievalFilters(project_id="project-1", source_systems=["GITHUB"])
    )

    assert where == {
        "$and": [
            {"project:project-1": {"$eq": True}},
            {"source_system": {"$in": ["GITHUB"]}},
        ]
    }


def test_project_ids_round_trip_through_the_delimited_encoding() -> None:
    assert encode_project_ids(("a", "b")) == "|a|b|"
    assert encode_project_ids(()) == ""
    assert decode_project_ids("|a|b|") == ("a", "b")
    assert decode_project_ids("") == ()
    assert decode_project_ids(None) == ()


def test_encode_project_ids_deduplicates_and_drops_blanks() -> None:
    assert encode_project_ids(["a", "", "a", "b"]) == "|a|b|"
