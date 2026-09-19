from rag.context_window import expand_context_window
from rag.hybrid import to_scored_chunk
from rag.source_filter import SourceExclusions
from rag.types import Chunk, RetrievalFilters, ScoredChunk
from tests.stubs.store import StubVectorStore


def _chunk(
    position: int,
    *,
    artifact_id: str = "artifact-1",
    project_ids: tuple[str, ...] = ("project-1",),
    connector_id: str | None = None,
) -> Chunk:
    return Chunk(
        id=f"{artifact_id}-{position}",
        artifact_id=artifact_id,
        filename=f"{artifact_id}.md",
        text=f"chunk {position}",
        embedding=[0.0],
        position=position,
        project_ids=project_ids,
        connector_id=connector_id,
    )


def _hit(chunk: Chunk, score: float) -> ScoredChunk:
    return to_scored_chunk(chunk, score)


def test_context_window_keeps_hits_first_and_deduplicates_overlaps() -> None:
    store = StubVectorStore()
    chunks = [_chunk(position) for position in range(5)]
    store.add(chunks)

    expanded = expand_context_window(
        [_hit(chunks[1], 0.9), _hit(chunks[3], 0.8)],
        store,
        filters=RetrievalFilters(project_id="project-1"),
    )

    assert [chunk.id for chunk in expanded] == [
        "artifact-1-1",
        "artifact-1-3",
        "artifact-1-0",
        "artifact-1-2",
        "artifact-1-4",
    ]
    assert expanded[3].score == 0.9


def test_context_window_never_adds_an_ineligible_neighbour() -> None:
    store = StubVectorStore()
    hit = _chunk(1)
    store.add(
        [
            _chunk(0, connector_id="disabled"),
            hit,
            _chunk(2, project_ids=("project-2",)),
        ]
    )

    expanded = expand_context_window(
        [_hit(hit, 0.9)],
        store,
        filters=RetrievalFilters(project_id="project-1"),
        exclusions=SourceExclusions(connectors=frozenset({"disabled"})),
    )

    assert [chunk.id for chunk in expanded] == ["artifact-1-1"]


def test_context_window_does_not_refilter_an_already_authorized_hit() -> None:
    store = StubVectorStore()
    hit = _hit(_chunk(1), 0.9)
    projected_hit = ScoredChunk(
        id=hit.id,
        artifact_id=hit.artifact_id,
        filename=hit.filename,
        text=hit.text,
        score=hit.score,
        position=hit.position,
    )

    expanded = expand_context_window(
        [projected_hit],
        store,
        filters=RetrievalFilters(project_id="project-1"),
    )

    assert expanded == [projected_hit]


def test_context_window_stays_within_the_total_chunk_budget() -> None:
    store = StubVectorStore()
    chunks = [_chunk(position) for position in range(5)]
    store.add(chunks)

    expanded = expand_context_window(
        [_hit(chunks[1], 0.9), _hit(chunks[3], 0.8)],
        store,
        max_chunks=3,
    )

    assert [chunk.id for chunk in expanded] == [
        "artifact-1-1",
        "artifact-1-3",
        "artifact-1-0",
    ]


def test_context_window_does_not_fetch_when_direct_hits_fill_the_budget() -> None:
    class NoNeighbourReadStore(StubVectorStore):
        def list_chunks_by_positions(
            self,
            artifact_id: str,
            positions: frozenset[int],
            start_page: int | None = None,
        ) -> list[Chunk]:
            raise AssertionError("neighbour read should not happen")

    store = NoNeighbourReadStore()
    chunks = [_chunk(position) for position in range(2)]
    store.add(chunks)

    expanded = expand_context_window(
        [_hit(chunks[0], 0.9), _hit(chunks[1], 0.8)],
        store,
        max_chunks=2,
    )

    assert [chunk.id for chunk in expanded] == [
        "artifact-1-0",
        "artifact-1-1",
    ]


def test_pdf_context_window_stays_on_the_matching_page() -> None:
    store = StubVectorStore()

    def pdf_chunk(page: int, position: int) -> Chunk:
        return Chunk(
            id=f"page-{page}-{position}",
            artifact_id="pdf-1",
            filename="guide.pdf",
            text=f"page {page}, chunk {position}",
            embedding=[0.0],
            kind="pdf",
            position=position,
            start_page=page,
        )

    chunks = [pdf_chunk(page, position) for page in (1, 2) for position in range(3)]
    store.add(chunks)
    hit = next(
        chunk for chunk in chunks if chunk.start_page == 2 and chunk.position == 1
    )

    expanded = expand_context_window([_hit(hit, 0.9)], store)

    assert [chunk.id for chunk in expanded] == [
        "page-2-1",
        "page-2-0",
        "page-2-2",
    ]


def test_legacy_pdf_without_page_locator_is_not_widened() -> None:
    store = StubVectorStore()
    hit = Chunk(
        id="legacy-hit",
        artifact_id="pdf-1",
        filename="guide.pdf",
        text="legacy PDF chunk",
        embedding=[0.0],
        kind="pdf",
        position=1,
        start_page=None,
    )
    store.add([hit, _chunk(0), _chunk(2)])

    expanded = expand_context_window([_hit(hit, 0.9)], store)

    assert [chunk.id for chunk in expanded] == ["legacy-hit"]
