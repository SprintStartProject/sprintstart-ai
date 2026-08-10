from datetime import UTC, datetime, timedelta

from src.rag.hybrid import (
    BM25IndexCache,
    hybrid_retrieve,
    reciprocal_rank_fusion,
)
from src.rag.source_filter import SourceExclusions
from src.rag.types import Chunk, RetrievalFilters, ScoredChunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def make_chunk(
    chunk_id: str,
    text: str,
    embedding: list[float],
    source_role: str = "primary",
    connector_id: str | None = None,
    connector_source_id: str | None = None,
    project_ids: tuple[str, ...] = (),
    artifact_id: str = "artifact-1",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        artifact_id=artifact_id,
        filename="doc.md",
        text=text,
        embedding=embedding,
        source_role=source_role,  # type: ignore[arg-type]
        connector_id=connector_id,
        connector_source_id=connector_source_id,
        project_ids=project_ids,
    )


def make_scored_chunk(chunk_id: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        id=chunk_id,
        artifact_id="artifact-1",
        filename="doc.md",
        text=text,
        score=0.0,
    )


def test_rrf_merge_with_known_rankings() -> None:
    chunk_a = make_scored_chunk("a", "alpha")
    chunk_b = make_scored_chunk("b", "beta")
    chunk_c = make_scored_chunk("c", "gamma")

    semantic_results = [chunk_a, chunk_b]
    bm25_results = [chunk_b, chunk_c]

    result = reciprocal_rank_fusion(
        ranked_lists=[semantic_results, bm25_results],
        k=60,
    )

    assert result[0].id == "b"
    assert {chunk.id for chunk in result} == {"a", "b", "c"}


def test_bm25_cache_invalidates_when_content_replaces_same_count() -> None:
    """Regression test for issue #129 #1: editing a chunk's text (same total
    count, new content-hashed id) must invalidate the cache, not just a count
    change. Stale BM25 hits would otherwise cite chunks Chroma no longer has.
    """
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [make_chunk(chunk_id="chunk-1", text="original text", embedding=[1.0, 0.0])]
    )
    first_index = cache.get(store)
    assert first_index.chunks[0].text == "original text"

    store.delete("artifact-1")
    store.add(
        [
            make_chunk(
                chunk_id="chunk-1-edited", text="edited text", embedding=[1.0, 0.0]
            )
        ]
    )

    second_index = cache.get(store)

    assert first_index is not second_index
    assert second_index.chunks[0].text == "edited text"


def test_bm25_cache_hit_does_not_rebuild_index() -> None:
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [make_chunk(chunk_id="chunk-1", text="first chunk", embedding=[1.0, 0.0])]
    )

    first_index = cache.get(store)
    second_index = cache.get(store)

    assert first_index is second_index


def test_bm25_cache_invalidates_when_chunk_count_changes() -> None:
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk(
                chunk_id="chunk-1",
                text="first chunk",
                embedding=[1.0, 0.0],
            )
        ]
    )

    first_index = cache.get(store)

    store.add(
        [
            make_chunk(
                chunk_id="chunk-2",
                text="second chunk",
                embedding=[0.0, 1.0],
            )
        ]
    )

    second_index = cache.get(store)

    assert first_index is not second_index


def test_large_corpus_uses_semantic_only_fallback() -> None:
    class LargeCorpusStore(StubVectorStore):
        def count(self) -> int:
            return 10_001

    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = LargeCorpusStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk(
                chunk_id="semantic-match",
                text="semantic match",
                embedding=[1.0, 0.0],
            ),
            make_chunk(
                chunk_id="keyword-only",
                text="exact SPECIAL_KEYWORD",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    result = hybrid_retrieve(
        question="SPECIAL_KEYWORD",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.8,
        bm25_cache=cache,
    )

    assert len(result) == 1
    assert result[0].id == "semantic-match"


def test_bm25_only_match_is_returned_when_semantic_finds_nothing() -> None:
    """Regression test for issue #129 #4: an exact-identifier query that embeds
    poorly (nothing clears min_score) must still surface the BM25 hit instead
    of being discarded.
    """
    llm = StubLLMClient(embedding=[0.0, 1.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    # BM25's idf is degenerate on a single-document corpus (score <= 0 even for
    # an exact match), so a couple of unrelated chunks keep the corpus realistic.
    store.add(
        [
            make_chunk(
                chunk_id="keyword-only", text="FooBarHandler", embedding=[1.0, 0.0]
            ),
            make_chunk(
                chunk_id="unrelated-1",
                text="unrelated document one",
                embedding=[1.0, 0.0],
            ),
            make_chunk(
                chunk_id="unrelated-2",
                text="unrelated document two",
                embedding=[1.0, 0.0],
            ),
        ]
    )

    result = hybrid_retrieve(
        question="FooBarHandler",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.99,
        bm25_cache=cache,
    )

    assert {chunk.id for chunk in result} == {"keyword-only"}


def test_exclude_roles_drops_test_chunks() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk("primary", "onboarding setup guide", [1.0, 0.0]),
            make_chunk("test", "onboarding setup guide", [1.0, 0.0], "test"),
        ]
    )

    result = hybrid_retrieve(
        question="onboarding setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclude_roles=frozenset({"test"}),
    )

    ids = {chunk.id for chunk in result}
    assert ids == {"primary"}


def test_exclude_roles_keeps_legacy_unmarked_chunks() -> None:
    """Chunks without a role default to 'primary' and survive a test filter."""
    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    # Built without source_role → defaults to "primary".
    store.add([make_chunk("legacy", "onboarding setup guide", [1.0, 0.0])])

    result = hybrid_retrieve(
        question="onboarding setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclude_roles=frozenset({"test"}),
    )

    assert {chunk.id for chunk in result} == {"legacy"}


def test_exclude_roles_in_semantic_only_fallback() -> None:
    class LargeCorpusStore(StubVectorStore):
        def count(self) -> int:
            return 10_001

    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = LargeCorpusStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk("primary", "match", [1.0, 0.0]),
            make_chunk("test", "match", [1.0, 0.0], "test"),
        ]
    )

    result = hybrid_retrieve(
        question="match",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclude_roles=frozenset({"test"}),
    )

    assert {chunk.id for chunk in result} == {"primary"}


def test_exclusions_drop_disabled_source_chunks() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk(
                "enabled-repo",
                "onboarding setup guide",
                [1.0, 0.0],
                connector_id="github",
                connector_source_id="owner/enabled-repo",
            ),
            make_chunk(
                "disabled-repo",
                "onboarding setup guide",
                [1.0, 0.0],
                connector_id="github",
                connector_source_id="owner/disabled-repo",
            ),
        ]
    )

    result = hybrid_retrieve(
        question="onboarding setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclusions=SourceExclusions(
            sources=frozenset({("github", "owner/disabled-repo")})
        ),
    )

    assert {chunk.id for chunk in result} == {"enabled-repo"}


def test_exclusions_drop_disabled_connector_chunks() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk(
                "jira-chunk",
                "onboarding setup guide",
                [1.0, 0.0],
                connector_id="jira",
                connector_source_id="PROJ",
            ),
            make_chunk(
                "github-chunk",
                "onboarding setup guide",
                [1.0, 0.0],
                connector_id="github",
                connector_source_id="owner/repo",
            ),
        ]
    )

    result = hybrid_retrieve(
        question="onboarding setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclusions=SourceExclusions(connectors=frozenset({"github"})),
    )

    assert {chunk.id for chunk in result} == {"jira-chunk"}


def test_exclusions_keep_legacy_chunks_without_connector() -> None:
    """Chunks without a connector_id are never excluded."""
    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add([make_chunk("legacy", "onboarding setup guide", [1.0, 0.0])])

    result = hybrid_retrieve(
        question="onboarding setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=cache,
        exclusions=SourceExclusions(connectors=frozenset({"github"})),
    )

    assert {chunk.id for chunk in result} == {"legacy"}


def test_hybrid_retrieval_applies_source_and_time_filters() -> None:
    recent_date = datetime.now(UTC).isoformat()
    old_date = (datetime.now(UTC) - timedelta(days=400)).isoformat()

    llm = StubLLMClient(embedding=[1.0, 0.0])
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-docs",
                artifact_id="artifact-docs",
                filename="doc.md",
                text="database setup",
                embedding=[1.0, 0.0],
                source_system="UPLOAD",
                created_at=recent_date,
            ),
            Chunk(
                id="chunk-old-code",
                artifact_id="artifact-old-code",
                filename="old.py",
                text="database setup",
                embedding=[1.0, 0.0],
                kind="code",
                source_system="GITHUB",
                created_at=old_date,
            ),
            Chunk(
                id="chunk-recent-code",
                artifact_id="artifact-recent-code",
                filename="app.py",
                text="database setup",
                embedding=[1.0, 0.0],
                kind="code",
                source_system="GITHUB",
                created_at=recent_date,
            ),
        ]
    )

    result = hybrid_retrieve(
        question="database setup",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        bm25_cache=BM25IndexCache(),
        filters=RetrievalFilters(
            source_systems=["GITHUB"],
            time_from=(datetime.now(UTC) - timedelta(days=183)).isoformat(),
        ),
    )

    assert [chunk.id for chunk in result] == ["chunk-recent-code"]


def test_bm25_cache_invalidates_when_project_membership_changes() -> None:
    """Reassigning an artifact between projects must invalidate the cache.

    Chunk ids are content-hashed, so moving an unchanged artifact from one
    project to another leaves the id set identical. An ids-only fingerprint
    would keep serving the cached index, whose chunks still carry the old
    membership, and BM25 would go on answering the project the artifact was
    moved out of.
    """
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add(
        [
            make_chunk(
                chunk_id="chunk-1",
                text="deployment runbook",
                embedding=[1.0, 0.0],
                project_ids=("project-a",),
            )
        ]
    )
    first_index = cache.get(store)
    assert first_index.chunks[0].project_ids == ("project-a",)

    store.add(
        [
            make_chunk(
                chunk_id="chunk-1",
                text="deployment runbook",
                embedding=[1.0, 0.0],
                project_ids=("project-b",),
            )
        ]
    )
    second_index = cache.get(store)

    assert first_index is not second_index
    assert second_index.chunks[0].project_ids == ("project-b",)


def test_bm25_cache_invalidates_when_source_role_changes() -> None:
    store = StubVectorStore()
    cache = BM25IndexCache()

    store.add([make_chunk("chunk-1", "fixture data", [1.0, 0.0])])
    first_index = cache.get(store)

    store.add([make_chunk("chunk-1", "fixture data", [1.0, 0.0], source_role="test")])
    second_index = cache.get(store)

    assert first_index is not second_index
    assert second_index.chunks[0].source_role == "test"


def test_bm25_keeps_project_chunk_ranked_below_foreign_chunks() -> None:
    """Ineligible chunks must be dropped from the ranking before truncation.

    The one chunk of project-b scores worst on BM25, and there are more
    higher-scoring project-a chunks than the fetch budget. Filtering after
    truncation would cut the ranking down to project-a chunks only and then
    filter all of them away, leaving project-b with nothing.
    """
    llm = StubLLMClient(embedding=[0.0, 1.0])
    store = StubVectorStore()

    # Filler so the query terms stay discriminative enough for BM25 to score
    # the documents below at all.
    filler = [
        make_chunk(
            chunk_id=f"filler-{index}",
            text=f"unrelated release notes topic {index}",
            embedding=[1.0, 0.0],
            project_ids=("project-a",),
            artifact_id=f"artifact-filler-{index}",
        )
        for index in range(100)
    ]
    # 60 full-term matches from another project — four times the fetch budget
    # (top_k * the filter over-fetch factor) — all ranked above the one chunk
    # this project is allowed to see, which matches two of the three terms.
    foreign = [
        make_chunk(
            chunk_id=f"foreign-{index}",
            text="database migration rollback",
            embedding=[1.0, 0.0],
            project_ids=("project-a",),
            artifact_id=f"artifact-foreign-{index}",
        )
        for index in range(60)
    ]
    mine = make_chunk(
        chunk_id="mine",
        text="database migration",
        embedding=[1.0, 0.0],
        project_ids=("project-b",),
        artifact_id="artifact-mine",
    )
    store.add([*filler, *foreign, mine])

    result = hybrid_retrieve(
        question="database migration rollback",
        llm=llm,
        store=store,
        top_k=3,
        min_score=0.9,  # semantic half returns nothing: this is the BM25 path
        bm25_cache=BM25IndexCache(),
        filters=RetrievalFilters(project_id="project-b"),
    )

    assert [chunk.id for chunk in result] == ["mine"]
