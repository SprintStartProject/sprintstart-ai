from ingestion.chunking.parsed_chunk import to_parsed_chunk


def test_a_chunk_carries_its_position_and_the_file_it_came_from() -> None:
    chunk = to_parsed_chunk("body", "text", "docs/guide.md", 2, 5)

    assert chunk.content == "body"
    assert chunk.kind == "text"
    assert chunk.metadata == {
        "filename": "guide.md",
        "type": ".md",
        "chunk_index": "2",
        "total_chunks": "5",
        "has_context_block": "false",
        "context_block_range": "",
        "has_overlap": "false",
        "overlap_range": "",
    }


def test_start_line_is_recorded_only_when_known() -> None:
    assert "start_line" not in to_parsed_chunk("b", "code", "a.py", 0, 1).metadata
    assert (
        to_parsed_chunk("b", "code", "a.py", 0, 1, start_line=12).metadata["start_line"]
        == "12"
    )
    assert (
        to_parsed_chunk("b", "code", "a.py", 0, 1, start_line=0).metadata["start_line"]
        == "0"
    )


def test_context_block_and_overlap_ranges_are_colon_separated() -> None:
    chunk = to_parsed_chunk(
        "ctx body",
        "text",
        "a.md",
        1,
        2,
        has_context_block=True,
        context_block_range=(0, 3),
        has_overlap=True,
        overlap_range=(4, 8),
    )

    assert chunk.metadata["has_context_block"] == "true"
    assert chunk.metadata["context_block_range"] == "0:3"
    assert chunk.metadata["has_overlap"] == "true"
    assert chunk.metadata["overlap_range"] == "4:8"
