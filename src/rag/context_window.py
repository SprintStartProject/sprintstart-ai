from collections import defaultdict

from rag.filters import matches_retrieval_filters
from rag.hybrid import to_scored_chunk
from rag.source_filter import SourceExclusions, is_excluded
from rag.types import Chunk, RetrievalFilters, ScoredChunk
from store.base import VectorStore

_DEFAULT_RADIUS = 1
_DEFAULT_MAX_CHUNKS = 12


def expand_context_window(
    hits: list[ScoredChunk],
    store: VectorStore,
    *,
    filters: RetrievalFilters | None = None,
    exclusions: SourceExclusions = SourceExclusions(),
    radius: int = _DEFAULT_RADIUS,
    max_chunks: int = _DEFAULT_MAX_CHUNKS,
) -> list[ScoredChunk]:
    """Append nearby chunks without displacing direct search matches.

    Direct matches keep their ranking and appear first. Neighbours are then
    appended by distance and hit order, so the downstream evidence budget
    always prefers actual matches and immediate context. Exact-position reads
    keep widening bounded and avoid loading an entire artifact merely to
    obtain two adjacent chunks. PDFs remain page-scoped because their global
    chunk index is not currently persisted in the retrieval store.
    """
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if max_chunks < 0:
        raise ValueError("max_chunks must be non-negative")

    # Hits have already passed their tool's retrieval-time eligibility checks.
    # Only neighbours loaded from the store need filtering here. Re-filtering
    # direct hits can wrongly discard results from tools that carry a reduced
    # presentation projection of the original store record.
    direct = _unique_hits(hits)
    remaining = max_chunks - len(direct)
    if radius == 0 or remaining <= 0:
        return direct

    target_positions: dict[tuple[str, int | None], set[int]] = defaultdict(set)
    for hit in direct:
        if hit.position is None:
            continue
        # PDF positions are page-local. Without a page we cannot distinguish
        # position 1 on page 2 from position 1 on every other page, so legacy
        # PDF chunks lacking this locator are deliberately not widened.
        if hit.kind == "pdf" and hit.start_page is None:
            continue
        locator = (hit.artifact_id, hit.start_page if hit.kind == "pdf" else None)
        for distance in range(1, radius + 1):
            if hit.position >= distance:
                target_positions[locator].add(hit.position - distance)
            target_positions[locator].add(hit.position + distance)

    neighbours: dict[tuple[str, int | None, int], Chunk] = {}
    for (artifact_id, start_page), positions in target_positions.items():
        for chunk in store.list_chunks_by_positions(
            artifact_id,
            frozenset(positions),
            start_page=start_page,
        ):
            if chunk.position is not None:
                neighbours[(artifact_id, start_page, chunk.position)] = chunk

    expanded = list(direct)
    seen = {chunk.id for chunk in direct}
    for distance in range(1, radius + 1):
        for hit in direct:
            if hit.position is None:
                continue
            if hit.kind == "pdf" and hit.start_page is None:
                continue
            start_page = hit.start_page if hit.kind == "pdf" else None
            for position in (hit.position - distance, hit.position + distance):
                if position < 0:
                    continue
                neighbour = neighbours.get((hit.artifact_id, start_page, position))
                if (
                    neighbour is None
                    or neighbour.id in seen
                    or not _is_eligible(neighbour, filters, exclusions)
                ):
                    continue
                expanded.append(to_scored_chunk(neighbour, hit.score))
                seen.add(neighbour.id)
                if len(expanded) >= max_chunks:
                    return expanded

    return expanded


def _unique_hits(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    unique: list[ScoredChunk] = []
    seen: set[str] = set()
    for chunk in chunks:
        if chunk.id in seen:
            continue
        unique.append(chunk)
        seen.add(chunk.id)
    return unique


def _is_eligible(
    chunk: Chunk | ScoredChunk,
    filters: RetrievalFilters | None,
    exclusions: SourceExclusions,
) -> bool:
    return matches_retrieval_filters(chunk, filters) and not is_excluded(
        chunk,
        exclusions,
    )
