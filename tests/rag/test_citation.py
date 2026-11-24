from corpus.types import ScoredChunk
from rag.citation import build_citations


def test_build_citations_maps_chunk_fields() -> None:
    chunks = [
        ScoredChunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="architecture.md",
            text="Some text",
            score=0.9,
        )
    ]

    citations = build_citations(chunks)

    assert len(citations) == 1
    assert citations[0].artifact_id == "artifact-1"


def test_build_citations_maps_start_line_for_code_chunks() -> None:
    chunks = [
        ScoredChunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="foo.py",
            text="def foo(): pass",
            score=0.9,
            kind="code",
            start_line=12,
        )
    ]

    citations = build_citations(chunks)

    assert citations[0].start_line == 12
    assert citations[0].start_page is None


def test_build_citations_maps_start_page_for_pdf_chunks() -> None:
    chunks = [
        ScoredChunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="doc.pdf",
            text="PDF text",
            score=0.9,
            kind="pdf",
            start_page=3,
        )
    ]

    citations = build_citations(chunks)

    assert citations[0].start_page == 3
    assert citations[0].start_line is None
