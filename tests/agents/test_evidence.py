"""The evidence budget and formatting every search loop shares.

Moved here from the chat agent so the buddy runs the same budget; the chat
agent's own tests still pin it end to end through a turn.
"""

from agents.tools.base import ToolResult
from agents.tools.evidence import (
    MAX_EVIDENCE_CHARS,
    MAX_EVIDENCE_CHUNKS,
    SOURCE_CHARS,
    format_evidence,
    limit_evidence,
    order_evidence_for_display,
)
from rag.types import ScoredChunk


def _chunk(i: int, text: str = "x") -> ScoredChunk:
    return ScoredChunk(
        id=f"c{i}", artifact_id=f"a{i}", filename=f"f{i}.md", text=text, score=1.0
    )


def test_the_budget_is_reachable() -> None:
    """The char budget only bites if the chunk count lets enough chunks through."""
    assert MAX_EVIDENCE_CHUNKS * SOURCE_CHARS > MAX_EVIDENCE_CHARS


def test_limit_keeps_a_prefix_capped_by_chunk_count() -> None:
    chunks = [_chunk(i) for i in range(20)]

    assert limit_evidence(chunks) == chunks[:MAX_EVIDENCE_CHUNKS]


def test_limit_keeps_a_prefix_capped_by_chars() -> None:
    chunks = [_chunk(i, "y" * SOURCE_CHARS) for i in range(12)]

    assert len(limit_evidence(chunks)) == MAX_EVIDENCE_CHARS // SOURCE_CHARS


def test_limit_always_keeps_the_first_chunk_however_large() -> None:
    chunks = [_chunk(0, "z" * 50_000)]

    assert limit_evidence(chunks) == chunks


def test_format_uses_the_callers_header_and_empty_text() -> None:
    result = ToolResult(summary="retrieve('x'): 1 chunk(s).", chunks=[_chunk(1)])

    shown = format_evidence(result, [_chunk(1)], header=lambda c: f"<{c.filename}>")
    nothing = format_evidence(result, [], empty="Nothing indexed matched.")

    assert shown.startswith("<f1.md>\n")
    assert nothing == "Nothing indexed matched."


def test_format_falls_back_to_the_tool_summary_when_empty() -> None:
    result = ToolResult.empty("grep(['xyzzy']): 0 chunk(s).")

    assert format_evidence(result, []) == "grep(['xyzzy']): 0 chunk(s)."


def test_dropped_context_is_not_reported_as_an_omitted_match() -> None:
    chunks = [
        ScoredChunk(
            id=f"c{position}",
            artifact_id="d1",
            filename="auth.py",
            text=f"chunk {position}",
            score=1.0,
            position=position,
        )
        for position in range(3)
    ]
    result = ToolResult(
        summary="retrieve('blocker'): 1 chunk(s).",
        chunks=chunks,
        match_chunk_ids=frozenset({"c1"}),
    )

    message = format_evidence(result, chunks[:2])

    assert "omitted" not in message


def test_pdf_evidence_is_shown_in_page_then_position_order() -> None:
    chunks = [
        ScoredChunk(
            id=f"page-{page}-{position}",
            artifact_id="pdf-1",
            filename="guide.pdf",
            text=f"page {page}, chunk {position}",
            score=1.0,
            kind="pdf",
            position=position,
            start_page=page,
        )
        for page, position in ((2, 1), (1, 1), (2, 0), (1, 0))
    ]

    ordered = order_evidence_for_display(chunks)

    assert [chunk.id for chunk in ordered] == [
        "page-1-0",
        "page-1-1",
        "page-2-0",
        "page-2-1",
    ]
