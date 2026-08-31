from ingestion.mapper import to_chunk
from ingestion.models import ParsedChunk


def test_to_chunk_carries_start_line_from_metadata():
    parsed = ParsedChunk(
        content="def foo():\n    pass",
        kind="code",
        metadata={
            "filename": "foo.py",
            "chunk_index": "0",
            "start_line": "12",
        },
    )

    chunk = to_chunk(parsed, artifact_id="artifact-1", embedding=[0.1])

    assert chunk.start_line == 12
    assert chunk.start_page is None


def test_to_chunk_carries_start_page_from_page_number_metadata():
    parsed = ParsedChunk(
        content="Some PDF text",
        kind="pdf",
        metadata={
            "filename": "doc.pdf",
            "chunk_index": "0",
            "page_number": "3",
        },
    )

    chunk = to_chunk(parsed, artifact_id="artifact-1", embedding=[0.1])

    assert chunk.start_page == 3
    assert chunk.start_line is None


def test_to_chunk_without_line_or_page_metadata_defaults_to_none():
    parsed = ParsedChunk(
        content="Plain text",
        kind="text",
        metadata={
            "filename": "notes.txt",
            "chunk_index": "0",
        },
    )

    chunk = to_chunk(parsed, artifact_id="artifact-1", embedding=[0.1])

    assert chunk.start_line is None
    assert chunk.start_page is None


def test_to_chunk_carries_project_ids():
    parsed = ParsedChunk(
        content="text",
        kind="text",
        metadata={"filename": "doc.md", "chunk_index": "0"},
    )

    chunk = to_chunk(
        parsed,
        artifact_id="artifact-1",
        embedding=[0.1],
        project_ids=["project-1", "project-2"],
    )

    assert chunk.project_ids == ("project-1", "project-2")


def test_to_chunk_deduplicates_and_drops_blank_project_ids():
    parsed = ParsedChunk(
        content="text",
        kind="text",
        metadata={"filename": "doc.md", "chunk_index": "0"},
    )

    chunk = to_chunk(
        parsed,
        artifact_id="artifact-1",
        embedding=[0.1],
        project_ids=["project-1", "", "project-1"],
    )

    assert chunk.project_ids == ("project-1",)


def test_to_chunk_without_project_ids_defaults_to_empty():
    parsed = ParsedChunk(
        content="text",
        kind="text",
        metadata={"filename": "doc.md", "chunk_index": "0"},
    )

    chunk = to_chunk(parsed, artifact_id="artifact-1", embedding=[0.1])

    assert chunk.project_ids == ()
