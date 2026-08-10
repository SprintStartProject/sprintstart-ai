from ingestion.mapper import to_chunk
from ingestion.models import ParsedChunk
from rag.filters import matches_retrieval_filters
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


# --- project scope ---------------------------------------------------------
#
# The corpus was one flat pool: every project's buddy searched every project's
# material, and could quote another team's process as this team's. These are the
# rules that stop it, and the one that stops it going too far.


def _chunk(project_ids: tuple[str, ...] = ()) -> Chunk:
    return Chunk(
        id="chunk-1",
        artifact_id="artifact-1",
        filename="process.md",
        text="How we work.",
        embedding=[1.0, 0.0],
        project_ids=project_ids,
    )


def test_material_from_another_project_is_not_searchable() -> None:
    assert not matches_retrieval_filters(
        _chunk(("project-b",)),
        RetrievalFilters(project_ids=frozenset({"project-a"})),
    )


def test_material_from_this_project_is() -> None:
    assert matches_retrieval_filters(
        _chunk(("project-a",)),
        RetrievalFilters(project_ids=frozenset({"project-a"})),
    )


def test_a_hire_on_two_projects_sees_both() -> None:
    # The reason the filter takes a set: narrowing to one of a person's projects
    # would hide their own material, and narrowing to none would show them
    # everybody else's.
    both = frozenset({"project-a", "project-b"})

    assert matches_retrieval_filters(
        _chunk(("project-a",)), RetrievalFilters(project_ids=both)
    )
    assert matches_retrieval_filters(
        _chunk(("project-b",)), RetrievalFilters(project_ids=both)
    )
    assert not matches_retrieval_filters(
        _chunk(("project-c",)), RetrievalFilters(project_ids=both)
    )


def test_material_shared_between_projects_belongs_to_both() -> None:
    # A repository serving two projects is one artifact serving both, which is
    # exactly why a chunk carries several ids rather than one.
    shared = _chunk(("project-a", "project-b"))

    assert matches_retrieval_filters(
        shared, RetrievalFilters(project_ids=frozenset({"project-a"}))
    )
    assert matches_retrieval_filters(
        shared, RetrievalFilters(project_ids=frozenset({"project-b"}))
    )


def test_unscoped_material_stays_visible_to_every_project() -> None:
    # Absent scope is not excluded scope -- the same rule a chunk with no
    # connector_id keeps, and a starter task with no track. Everything ingested
    # before this field existed has no project, and hiding all of it the moment a
    # caller starts scoping would look like the corpus had emptied.
    assert matches_retrieval_filters(
        _chunk(),
        RetrievalFilters(project_ids=frozenset({"project-a"})),
    )


def test_asking_for_no_project_still_returns_everything() -> None:
    # A single-project deployment passes nothing and must be unaffected.
    assert matches_retrieval_filters(_chunk(("project-b",)), RetrievalFilters())
