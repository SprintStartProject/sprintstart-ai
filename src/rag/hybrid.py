import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, cast

from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from ingestion.source_role import SourceRole
from llm.base import LLMClient
from rag.filters import matches_retrieval_filters
from rag.source_filter import SourceExclusions, is_excluded
from rag.types import Chunk, RetrievalFilters, ScoredChunk
from store.base import VectorStore

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

RRF_K = 60
RRF_MIN_RATIO = 0.15
SEMANTIC_ONLY_CHUNK_LIMIT = 10_000

# First-guess over-fetch when a role, source, or attribute filter is active.
# This is a round-trip optimisation only: both retrievers drop ineligible
# candidates before truncating, and the semantic half widens its fetch until it
# has enough eligible results, so correctness never depends on this factor.
_ROLE_FILTER_OVERFETCH = 5


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_./:-]+", text.lower())


def to_scored_chunk(chunk: Chunk | ScoredChunk, score: float) -> ScoredChunk:
    return ScoredChunk(
        id=chunk.id,
        artifact_id=chunk.artifact_id,
        filename=chunk.filename,
        position=chunk.position,
        kind=chunk.kind,
        text=chunk.text,
        score=score,
        source_role=chunk.source_role,
        connector_id=chunk.connector_id,
        connector_source_id=chunk.connector_source_id,
        source_url=chunk.source_url,
        artifact_type=chunk.artifact_type,
        language=chunk.language,
        source_system=chunk.source_system,
        created_at=chunk.created_at,
        start_line=chunk.start_line,
        start_page=chunk.start_page,
        project_ids=chunk.project_ids,
    )


def build_eligibility(
    exclude_roles: frozenset[SourceRole],
    exclusions: SourceExclusions,
    filters: RetrievalFilters | None,
) -> "Callable[[Chunk | ScoredChunk], bool]":
    """The single "may this chunk be retrieved?" predicate, for both retrievers.

    Semantic and BM25 retrieval must agree on eligibility exactly — a chunk one
    half considers foreign and the other serves is a project leak. Deriving both
    from this one predicate is what keeps them from drifting apart.
    """

    def is_eligible(chunk: Chunk | ScoredChunk) -> bool:
        if exclude_roles and chunk.source_role in exclude_roles:
            return False
        if is_excluded(chunk, exclusions):
            return False
        return matches_retrieval_filters(chunk, filters)

    return is_eligible


def _semantic_candidates(
    store: VectorStore,
    embedding: list[float],
    target: int,
    min_score: float,
    corpus_size: int,
    is_eligible: "Callable[[Chunk | ScoredChunk], bool]",
    exclude_roles: frozenset[SourceRole],
    exclusions: SourceExclusions,
    filters: RetrievalFilters | None,
) -> list[ScoredChunk]:
    """Up to ``target`` eligible semantic hits, ranked, filtered before cutoff.

    The constraints are handed to the store so it can filter before it limits
    (Chroma does this in its ``where`` clause). The predicate is re-applied here
    and the fetch widens on a short result because the store is a Protocol: an
    implementation that silently ignored a constraint, or a chunk whose
    eligibility isn't expressible in the backend's query language, would
    otherwise let higher-ranked ineligible chunks fill the window and hide every
    eligible hit. Widening ends at the corpus size, so the invariant holds
    without relying on a fixed over-fetch multiplier.
    """
    if corpus_size <= 0 or target <= 0:
        return []

    fetch_size = min(target, corpus_size)

    while True:
        candidates = store.query(
            embedding=embedding,
            top_k=fetch_size,
            min_score=min_score,
            filters=filters,
            exclude_roles=exclude_roles,
            exclusions=exclusions,
        )
        eligible = [chunk for chunk in candidates if is_eligible(chunk)]

        # A short result means the corpus is exhausted at this score floor:
        # scores fall monotonically, so nothing further down clears min_score.
        exhausted = len(candidates) < fetch_size or fetch_size >= corpus_size
        if len(eligible) >= target or exhausted:
            return eligible[:target]

        fetch_size = min(fetch_size * 2, corpus_size)


def reciprocal_rank_fusion(
    ranked_lists: list[list[ScoredChunk]],
    k: int = RRF_K,
) -> list[ScoredChunk]:
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, ScoredChunk] = {}

    for ranked_chunks in ranked_lists:
        for rank, chunk in enumerate(ranked_chunks, start=1):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.id] = chunk

    fused = [
        to_scored_chunk(chunks_by_id[chunk_id], score)
        for chunk_id, score in scores.items()
    ]

    fused.sort(key=lambda chunk: chunk.score, reverse=True)
    return fused


def filter_by_rrf_ratio(
    chunks: list[ScoredChunk],
    min_ratio: float = RRF_MIN_RATIO,
) -> list[ScoredChunk]:
    if not chunks:
        return []

    top_score = chunks[0].score
    threshold = top_score * min_ratio

    return [chunk for chunk in chunks if chunk.score >= threshold]


class BM25Index:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self.tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.index: Any | None = (
            BM25Okapi(self.tokenized_corpus) if self.tokenized_corpus else None
        )

    def query(
        self,
        question: str,
        top_k: int,
        is_eligible: "Callable[[Chunk], bool] | None" = None,
    ) -> list[ScoredChunk]:
        """The ``top_k`` best-scoring *eligible* chunks.

        ``is_eligible`` is applied to the full ranking before it is truncated.
        Filtering the truncated list instead would let highly-ranked ineligible
        chunks (another project's, an excluded connector's) crowd out every
        eligible chunk and return nothing — over-fetching only lowers the odds
        of that, it doesn't remove it.
        """
        if not self.chunks or self.index is None:
            return []

        tokenized_question = tokenize(question)
        raw_scores = self.index.get_scores(tokenized_question)
        scores = cast("Sequence[float]", raw_scores)

        ranked = sorted(
            zip(self.chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        results: list[ScoredChunk] = []
        for chunk, score in ranked:
            if score <= 0:
                # Descending order: nothing below this scores above zero.
                break
            if is_eligible is not None and not is_eligible(chunk):
                continue
            results.append(to_scored_chunk(chunk, float(score)))
            if len(results) == top_k:
                break

        return results


class BM25IndexCache:
    def __init__(self) -> None:
        # The fingerprint and its index are stored together as one immutable
        # tuple so the fast path reads a single reference. Reading the id-set and
        # the index separately would let a concurrent rebuild swap them mid-check
        # and return an index that doesn't match the fingerprint we validated.
        self._snapshot: tuple[frozenset[str], BM25Index] | None = None
        self._lock = threading.Lock()

    def get(self, store: VectorStore) -> BM25Index:
        # Fingerprint over ids *and* the metadata the index filters on (project
        # membership, source role, connector, source system, timestamp), but not
        # text or embeddings, so a cache hit stays cheap. Chunk ids are
        # content-hashed and therefore cover the text; they do not cover
        # membership, so an ids-only fingerprint would let an artifact moved
        # from project A to project B keep serving BM25 hits to project A for as
        # long as the cached index lived.
        current_fingerprints = store.retrieval_fingerprints()

        snapshot = self._snapshot
        if snapshot is not None and snapshot[0] == current_fingerprints:
            return snapshot[1]

        with self._lock:
            # Another thread may have rebuilt for this exact corpus state while
            # we were computing the fingerprint above; avoid rebuilding twice.
            snapshot = self._snapshot
            if snapshot is not None and snapshot[0] == current_fingerprints:
                return snapshot[1]

            index = BM25Index(store.all_chunks_without_embeddings())
            self._snapshot = (current_fingerprints, index)
            return index


def hybrid_retrieve(
    question: str,
    llm: LLMClient,
    store: VectorStore,
    top_k: int,
    min_score: float,
    bm25_cache: BM25IndexCache,
    exclude_roles: frozenset[SourceRole] = frozenset(),
    exclusions: SourceExclusions = SourceExclusions(),
    filters: RetrievalFilters | None = None,
) -> list[ScoredChunk]:
    """Hybrid (semantic + BM25) retrieval with reciprocal-rank fusion.

    ``exclude_roles`` drops chunks of the given :data:`SourceRole`\\ s (e.g.
    ``"test"``), ``exclusions`` drops chunks belonging to disabled
    connectors/sources, and ``filters`` drops chunks by project, source system
    and time range. Both halves enforce all three the same way — rank, drop the
    ineligible, *then* take the top results — so a run of higher-ranked
    ineligible chunks can never hide the eligible ones. Legacy chunks without a
    role or connector default to unfiltered (``primary`` role, never excluded) —
    but a project filter is fail-closed, so chunks without a project are dropped.
    """
    chunk_count = store.count()
    has_filters = (
        bool(exclude_roles)
        or bool(exclusions)
        or (
            filters is not None
            and (
                filters.project_id is not None
                or filters.project_ids is not None
                or bool(filters.source_systems)
                or filters.time_from is not None
                or filters.time_to is not None
            )
        )
    )
    fetch_k = top_k * _ROLE_FILTER_OVERFETCH if has_filters else top_k

    is_eligible = build_eligibility(exclude_roles, exclusions, filters)

    def semantic_candidates(target: int) -> list[ScoredChunk]:
        return _semantic_candidates(
            store=store,
            embedding=llm.embed(question),
            target=target,
            min_score=min_score,
            corpus_size=chunk_count,
            is_eligible=is_eligible,
            exclude_roles=exclude_roles,
            exclusions=exclusions,
            filters=filters,
        )

    if chunk_count > SEMANTIC_ONLY_CHUNK_LIMIT:
        return semantic_candidates(top_k)

    bm25_index = bm25_cache.get(store)

    with ThreadPoolExecutor(max_workers=2) as executor:
        # Both halves fetch the same budget of already-eligible candidates, so
        # fusion sees two comparable rankings.
        semantic_future = executor.submit(lambda: semantic_candidates(fetch_k))
        bm25_future = executor.submit(
            lambda: bm25_index.query(
                question=question,
                top_k=fetch_k,
                is_eligible=is_eligible,
            )
        )

        semantic_results = semantic_future.result()
        bm25_results = bm25_future.result()

    if not semantic_results:
        return bm25_results[:top_k]

    fused = reciprocal_rank_fusion(
        ranked_lists=[semantic_results, bm25_results],
        k=RRF_K,
    )

    filtered = filter_by_rrf_ratio(
        chunks=fused,
        min_ratio=RRF_MIN_RATIO,
    )

    return filtered[:top_k]
