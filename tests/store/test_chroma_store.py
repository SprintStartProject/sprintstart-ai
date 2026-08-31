import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import chromadb
import pytest

from ingestion.source_role import SourceRole
from rag.source_filter import SourceExclusions
from rag.types import Chunk, RetrievalFilters
from store import chroma_store as chroma_store_module
from store.chroma_store import ChromaVectorStore


def test_persistent_client_cold_cache_is_constructed_once_concurrently(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "shared-chroma")
    fake_client = object()
    worker_count = 8
    callers_ready = threading.Barrier(worker_count + 1)
    release_constructor = threading.Event()
    construction_count = 0
    count_lock = threading.Lock()

    def construct(*, path: str, settings: object) -> object:
        del path, settings
        nonlocal construction_count
        with count_lock:
            construction_count += 1
        release_constructor.wait(timeout=2)
        return fake_client

    def resolve() -> object:
        callers_ready.wait(timeout=2)
        return chroma_store_module._get_persistent_client(path)

    monkeypatch.setattr(chroma_store_module.chromadb, "PersistentClient", construct)
    chroma_store_module._CLIENT_CACHE.pop(path, None)

    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [executor.submit(resolve) for _ in range(worker_count)]
            callers_ready.wait(timeout=2)
            time.sleep(0.05)
            release_constructor.set()
            clients = [future.result(timeout=2) for future in futures]
    finally:
        release_constructor.set()
        chroma_store_module._CLIENT_CACHE.pop(path, None)

    assert construction_count == 1
    assert all(client is fake_client for client in clients)


def test_failed_persistent_client_construction_is_not_cached(
    monkeypatch,
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "retry-chroma")
    fake_client = object()
    attempts = 0

    def construct(*, path: str, settings: object) -> object:
        del path, settings
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary Chroma failure")
        return fake_client

    monkeypatch.setattr(chroma_store_module.chromadb, "PersistentClient", construct)
    chroma_store_module._CLIENT_CACHE.pop(path, None)

    try:
        with pytest.raises(RuntimeError, match="temporary Chroma failure"):
            chroma_store_module._get_persistent_client(path)

        assert path not in chroma_store_module._CLIENT_CACHE
        assert chroma_store_module._get_persistent_client(path) is fake_client
        assert attempts == 2
    finally:
        chroma_store_module._CLIENT_CACHE.pop(path, None)


def test_chroma_query_returns_chunks_above_min_score() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_query",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                position=1,
                kind="text",
                text="Relevant text",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-1",
                filename="doc.md",
                position=2,
                kind="text",
                text="Irrelevant text",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.8,
    )

    assert len(result) == 1
    assert result[0].id == "chunk-1"
    assert result[0].score >= 0.8


def test_chroma_query_round_trips_connector_fields() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_connector_fields",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Some text",
                embedding=[1.0, 0.0],
                connector_id="github",
                connector_source_id="owner/repo",
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.0)

    assert result[0].connector_id == "github"
    assert result[0].connector_source_id == "owner/repo"

    [listed] = store.list_chunks(limit=5)
    assert listed.connector_id == "github"
    assert listed.connector_source_id == "owner/repo"


def test_chroma_round_trips_several_project_ids_through_one_scalar() -> None:
    # Chroma metadata holds scalars, so a chunk's projects are stored delimited
    # and decoded on the way out. If that encoding is wrong the failure is silent
    # -- material simply stops matching its own project.
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_project_ids",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Some text",
                embedding=[1.0, 0.0],
                project_ids=("project-a", "project-b"),
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-2",
                filename="shared.md",
                text="Other text",
                embedding=[1.0, 0.0],
            ),
        ]
    )

    by_id = {
        chunk.id: chunk
        for chunk in store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.0)
    }

    assert by_id["chunk-1"].project_ids == ("project-a", "project-b")
    # Unscoped stays genuinely empty rather than becoming a one-element tuple of "".
    assert by_id["chunk-2"].project_ids == ()

    listed = {chunk.id: chunk for chunk in store.list_chunks(limit=5)}
    assert listed["chunk-1"].project_ids == ("project-a", "project-b")


def test_chroma_query_returns_empty_list_when_threshold_too_high() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_empty",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Some text",
                embedding=[0.0, 1.0],
            )
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.8,
    )

    assert result == []


def test_chroma_delete_removes_only_matching_artifact() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_delete",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="a.md",
                text="Text A",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-2",
                filename="b.md",
                text="Text B",
                embedding=[1.0, 0.0],
            ),
        ]
    )

    store.delete("artifact-1")

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.1,
    )

    assert len(result) == 1
    assert result[0].id == "chunk-2"
    assert result[0].artifact_id == "artifact-2"


def test_chroma_query_returns_empty_list_when_collection_is_empty() -> None:
    store = ChromaVectorStore(collection_name="test_empty_collection")

    result = store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.0)

    assert result == []


def test_chroma_add_empty_list_is_noop() -> None:
    store = ChromaVectorStore(collection_name="test_add_empty")

    store.add([])

    result = store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.0)
    assert result == []


def test_chroma_add_upserts_duplicate_ids() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_upsert",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Original text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Updated text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=5, min_score=0.1)

    assert len(result) == 1
    assert result[0].text == "Updated text"


def test_chroma_query_result_has_score() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_query_score",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Some text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert isinstance(result[0].score, float)


def test_chroma_ephemeral_constructor_requires_no_args() -> None:
    store = ChromaVectorStore()

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Hello",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert len(result) == 1


def test_chroma_round_trips_source_role() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_source_role", client=client)

    store.add(
        [
            Chunk(
                id="chunk-test",
                artifact_id="artifact-1",
                filename="test_foo.py",
                text="test material",
                embedding=[1.0, 0.0],
                source_role="test",
            ),
        ]
    )

    scored = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)
    assert scored[0].source_role == "test"

    listed = store.list_chunks(limit=store.count())
    assert listed[0].source_role == "test"


def test_chroma_legacy_chunks_default_to_primary() -> None:
    """A chunk stored without a source_role reads back as 'primary'."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="legacy", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["legacy-1"],
        documents=["legacy text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"artifact_id": "a", "filename": "doc.md", "kind": "text"}],
    )
    store = ChromaVectorStore(collection_name="legacy", client=client)

    scored = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert scored[0].source_role == "primary"


def test_chroma_all_chunks_without_embeddings_omits_embeddings() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_no_embeddings", client=client)

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Some text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    chunks = store.all_chunks_without_embeddings()

    assert len(chunks) == 1
    assert chunks[0].text == "Some text"
    assert chunks[0].embedding == []


def test_chroma_paged_reads_return_every_chunk_exactly_once(monkeypatch) -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_paged_reads", client=client)
    store.add(
        [
            Chunk(
                id=f"chunk-{index}",
                artifact_id=f"artifact-{index}",
                filename=f"{index}.md",
                text=f"Text {index}",
                embedding=[1.0, 0.0],
                project_ids=("project-1",),
            )
            for index in range(5)
        ]
    )
    monkeypatch.setattr(chroma_store_module, "_MAX_GET_PAGE", 2)

    chunks = list(store.iter_chunks_without_embeddings())
    chunk_ids = [chunk.id for chunk in chunks]

    metadata_reads: list[dict[str, object]] = []
    original_get = store._collection.get

    def recording_get(**kwargs):
        metadata_reads.append(kwargs)
        return original_get(**kwargs)

    monkeypatch.setattr(store._collection, "get", recording_get)
    fingerprints = store.retrieval_fingerprints()

    assert len(chunk_ids) == len(set(chunk_ids)) == 5
    assert set(chunk_ids) == {f"chunk-{index}" for index in range(5)}
    assert all(chunk.embedding == [] for chunk in chunks)
    assert {value.split("\x00", 1)[0] for value in fingerprints} == set(chunk_ids)
    assert [read["offset"] for read in metadata_reads] == [0, 2, 4, 5]
    assert all(read["limit"] == 2 for read in metadata_reads)
    assert all(read["include"] == ["metadatas"] for read in metadata_reads)


def test_chroma_all_ids_returns_every_chunk_id() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_all_ids", client=client)

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="a.md",
                text="Text A",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-2",
                filename="b.md",
                text="Text B",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    assert store.all_ids() == frozenset({"chunk-1", "chunk-2"})


def test_chroma_all_ids_changes_when_content_replaces_same_count() -> None:
    """Content-hashed ids mean a same-count edit still changes the id set."""
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_all_ids_churn", client=client)

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="a.md",
                text="Original text",
                embedding=[1.0, 0.0],
            )
        ]
    )
    before = store.all_ids()

    store.delete("artifact-1")
    store.add(
        [
            Chunk(
                id="chunk-1-edited",
                artifact_id="artifact-1",
                filename="a.md",
                text="Edited text",
                embedding=[1.0, 0.0],
            )
        ]
    )
    after = store.all_ids()

    assert len(before) == len(after) == 1
    assert before != after


def test_chroma_persistent_constructor(tmp_path: Path) -> None:
    store = ChromaVectorStore(
        collection_name="test_persistent",
        path=str(tmp_path),
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Persisted text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert len(result) == 1


def test_chroma_round_trips_start_line_and_start_page() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(collection_name="test_start_line_page", client=client)

    store.add(
        [
            Chunk(
                id="chunk-code",
                artifact_id="artifact-1",
                filename="foo.py",
                text="def foo(): pass",
                embedding=[1.0, 0.0],
                start_line=12,
            ),
            Chunk(
                id="chunk-pdf",
                artifact_id="artifact-1",
                filename="doc.pdf",
                text="PDF text",
                embedding=[0.0, 1.0],
                start_page=3,
            ),
        ]
    )

    scored_code = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)
    assert scored_code[0].start_line == 12
    assert scored_code[0].start_page is None

    scored_pdf = store.query(embedding=[0.0, 1.0], top_k=1, min_score=0.1)
    assert scored_pdf[0].start_page == 3
    assert scored_pdf[0].start_line is None

    listed = store.list_chunks(limit=store.count())
    by_id = {chunk.id: chunk for chunk in listed}
    assert by_id["chunk-code"].start_line == 12
    assert by_id["chunk-pdf"].start_page == 3


def test_chroma_legacy_chunks_without_line_or_page_default_to_none() -> None:
    """A chunk stored before start_line/start_page existed reads back as None."""
    client = chromadb.EphemeralClient()
    collection = client.get_or_create_collection(
        name="legacy_line_page", metadata={"hnsw:space": "cosine"}
    )
    collection.add(
        ids=["legacy-1"],
        documents=["legacy text"],
        embeddings=[[1.0, 0.0]],
        metadatas=[{"artifact_id": "a", "filename": "doc.md", "kind": "text"}],
    )
    store = ChromaVectorStore(collection_name="legacy_line_page", client=client)

    scored = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.1)

    assert scored[0].start_line is None
    assert scored[0].start_page is None


def test_chroma_query_applies_source_type_filter() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_query_source_type_filter",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-docs",
                artifact_id="artifact-docs",
                filename="doc.md",
                text="Docs text",
                embedding=[1.0, 0.0],
                source_system="UPLOAD",
            ),
            Chunk(
                id="chunk-code",
                artifact_id="artifact-code",
                filename="app.py",
                text="Code text",
                embedding=[1.0, 0.0],
                source_system="GITHUB",
                kind="code",
            ),
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.0,
        filters=RetrievalFilters(source_systems=["GITHUB"]),
    )

    assert len(result) == 1
    assert result[0].id == "chunk-code"
    assert result[0].source_system == "GITHUB"


def test_chroma_query_applies_time_range_filter() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_query_time_range_filter",
        client=client,
    )

    old_date = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    recent_date = datetime.now(UTC).isoformat()

    store.add(
        [
            Chunk(
                id="chunk-old",
                artifact_id="artifact-old",
                filename="old.md",
                text="Old text",
                embedding=[1.0, 0.0],
                source_system="UPLOAD",
                created_at=old_date,
            ),
            Chunk(
                id="chunk-recent",
                artifact_id="artifact-recent",
                filename="recent.md",
                text="Recent text",
                embedding=[1.0, 0.0],
                source_system="UPLOAD",
                created_at=recent_date,
            ),
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.0,
        filters=RetrievalFilters(
            time_from=(datetime.now(UTC) - timedelta(days=183)).isoformat(),
        ),
    )

    assert len(result) == 1
    assert result[0].id == "chunk-recent"


def test_chroma_query_combines_filters_with_and() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_query_combined_filters",
        client=client,
    )

    old_date = (datetime.now(UTC) - timedelta(days=400)).isoformat()
    recent_date = datetime.now(UTC).isoformat()

    store.add(
        [
            Chunk(
                id="chunk-recent-docs",
                artifact_id="artifact-docs",
                filename="doc.md",
                text="Recent docs",
                embedding=[1.0, 0.0],
                source_system="UPLOAD",
                created_at=recent_date,
            ),
            Chunk(
                id="chunk-old-code",
                artifact_id="artifact-old-code",
                filename="old.py",
                text="Old code",
                embedding=[1.0, 0.0],
                source_system="GITHUB",
                kind="code",
                created_at=old_date,
            ),
            Chunk(
                id="chunk-recent-code",
                artifact_id="artifact-recent-code",
                filename="app.py",
                text="Recent code",
                embedding=[1.0, 0.0],
                source_system="GITHUB",
                kind="code",
                created_at=recent_date,
            ),
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.0,
        filters=RetrievalFilters(
            source_systems=["GITHUB"],
            time_from=(datetime.now(UTC) - timedelta(days=183)).isoformat(),
        ),
    )

    assert len(result) == 1
    assert result[0].id == "chunk-recent-code"


def test_chroma_query_filters_by_project() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_project_filter",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-own",
                artifact_id="artifact-own",
                filename="own.md",
                text="Shared wording",
                embedding=[1.0, 0.0],
                project_ids=("project-1",),
            ),
            Chunk(
                id="chunk-foreign",
                artifact_id="artifact-foreign",
                filename="foreign.md",
                text="Shared wording",
                embedding=[1.0, 0.0],
                project_ids=("project-2",),
            ),
            Chunk(
                id="chunk-legacy",
                artifact_id="artifact-legacy",
                filename="legacy.md",
                text="Shared wording",
                embedding=[1.0, 0.0],
            ),
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=10,
        min_score=0.0,
        filters=RetrievalFilters(project_id="project-1"),
    )

    assert [chunk.id for chunk in result] == ["chunk-own"]


def test_chroma_query_returns_chunks_shared_by_two_projects() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_shared_project",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-shared",
                artifact_id="artifact-shared",
                filename="shared.md",
                text="Shared doc",
                embedding=[1.0, 0.0],
                project_ids=("project-1", "project-2"),
            )
        ]
    )

    for project_id in ("project-1", "project-2"):
        result = store.query(
            embedding=[1.0, 0.0],
            top_k=10,
            min_score=0.0,
            filters=RetrievalFilters(project_id=project_id),
        )
        assert [chunk.id for chunk in result] == ["chunk-shared"]


def test_chroma_round_trips_project_ids() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_project_round_trip",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Text",
                embedding=[1.0, 0.0],
                project_ids=("project-1", "project-2"),
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-2",
                filename="legacy.md",
                text="Text",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    by_id = {chunk.id: chunk for chunk in store.list_chunks(limit=store.count())}
    assert by_id["chunk-1"].project_ids == ("project-1", "project-2")
    assert by_id["chunk-2"].project_ids == ()

    scored = store.query(embedding=[1.0, 0.0], top_k=1, min_score=0.0)
    assert scored[0].project_ids == ("project-1", "project-2")

    without_embeddings = {
        chunk.id: chunk for chunk in store.all_chunks_without_embeddings()
    }
    assert without_embeddings["chunk-1"].project_ids == ("project-1", "project-2")


def test_chroma_reingest_replaces_project_membership() -> None:
    """Moving an artifact between projects must not leave a stale marker."""
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_chunks_project_reingest",
        client=client,
    )

    def chunk(project_ids: tuple[str, ...]) -> Chunk:
        return Chunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="doc.md",
            text="Text",
            embedding=[1.0, 0.0],
            project_ids=project_ids,
        )

    store.add([chunk(("project-1",))])
    store.add([chunk(("project-2",))])

    assert (
        store.query(
            embedding=[1.0, 0.0],
            top_k=10,
            min_score=0.0,
            filters=RetrievalFilters(project_id="project-1"),
        )
        == []
    )
    assert [
        c.id
        for c in store.query(
            embedding=[1.0, 0.0],
            top_k=10,
            min_score=0.0,
            filters=RetrievalFilters(project_id="project-2"),
        )
    ] == ["chunk-1"]


def test_chroma_retrieval_fingerprints_change_with_project_membership() -> None:
    """The BM25 cache key must move when membership does, ids alone don't.

    Content-hashed ids are identical before and after the move, so a cache
    keyed on ids would keep an index whose chunks still carry project-1.
    """
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_fingerprints_project",
        client=client,
    )

    def chunk(project_ids: tuple[str, ...]) -> Chunk:
        return Chunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="doc.md",
            text="Text",
            embedding=[1.0, 0.0],
            project_ids=project_ids,
        )

    store.add([chunk(("project-1",))])
    before = store.retrieval_fingerprints()

    store.add([chunk(("project-2",))])
    after = store.retrieval_fingerprints()

    assert store.all_ids() == frozenset({"chunk-1"})
    assert len(before) == len(after) == 1
    assert before != after


def test_chroma_retrieval_fingerprints_change_with_source_role() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_fingerprints_role",
        client=client,
    )

    def chunk(source_role: SourceRole) -> Chunk:
        return Chunk(
            id="chunk-1",
            artifact_id="artifact-1",
            filename="doc.md",
            text="Text",
            embedding=[1.0, 0.0],
            source_role=source_role,
        )

    store.add([chunk("primary")])
    before = store.retrieval_fingerprints()

    store.add([chunk("test")])

    assert store.retrieval_fingerprints() != before


def test_chroma_retrieval_fingerprints_stable_for_unchanged_corpus() -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_fingerprints_stable",
        client=client,
    )

    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Text",
                embedding=[1.0, 0.0],
                project_ids=("project-1",),
            )
        ]
    )

    assert store.retrieval_fingerprints() == store.retrieval_fingerprints()


def test_chroma_project_ids_for_artifact_reads_indexed_membership(monkeypatch) -> None:
    client = chromadb.EphemeralClient()
    store = ChromaVectorStore(
        collection_name="test_project_ids_for_artifact",
        client=client,
    )

    store.add(
        [
            Chunk(
                id=f"chunk-{index}",
                artifact_id="artifact-1",
                filename="doc.md",
                text=f"Text {index}",
                embedding=[1.0, 0.0],
                project_ids=(f"project-{index % 2 + 1}",),
            )
            for index in range(5)
        ]
        + [
            Chunk(
                id="chunk-other",
                artifact_id="artifact-2",
                filename="other.md",
                text="Other",
                embedding=[0.0, 1.0],
                project_ids=("project-3",),
            ),
        ]
    )

    monkeypatch.setattr(chroma_store_module, "_MAX_GET_PAGE", 2)
    metadata_reads: list[dict[str, object]] = []
    original_get = store._collection.get

    def recording_get(**kwargs):
        metadata_reads.append(kwargs)
        return original_get(**kwargs)

    monkeypatch.setattr(store._collection, "get", recording_get)

    assert store.project_ids_for_artifact("artifact-1") == frozenset(
        {"project-1", "project-2"}
    )
    assert [read["offset"] for read in metadata_reads] == [0, 2, 4, 5]
    assert all(read["limit"] == 2 for read in metadata_reads)
    assert all(read["include"] == ["metadatas"] for read in metadata_reads)
    assert all(
        read["where"] == {"artifact_id": "artifact-1"} for read in metadata_reads
    )

    assert store.project_ids_for_artifact("artifact-2") == frozenset({"project-3"})
    assert store.project_ids_for_artifact("missing") == frozenset()


_EXCLUSION_DIM = 16
_QUERY_EMBEDDING = [1.0] + [0.0] * (_EXCLUSION_DIM - 1)


def _ranked_embedding(offset: float, slot: int) -> list[float]:
    """A unit-ish vector whose cosine to ``_QUERY_EMBEDDING`` falls with ``offset``.

    ``slot`` spreads the offset over different dimensions so the vectors are
    genuinely distinct. Near-duplicate vectors are a degenerate case for HNSW
    graph construction — a corpus of *identical* embeddings makes Chroma's
    filtered search lose recall non-deterministically, which would make this
    test flaky for a reason that has nothing to do with what it checks.
    """
    embedding = [0.0] * _EXCLUSION_DIM
    embedding[0] = 1.0
    embedding[1 + slot % (_EXCLUSION_DIM - 1)] = offset
    return embedding


def _exclusion_corpus(store: ChromaVectorStore) -> None:
    """240 ineligible chunks, all ranked above the two eligible ones."""

    # Offsets stay well under the eligible chunks' 0.30, so every ineligible
    # chunk is a closer match to the query than either eligible chunk.
    def ineligible_offset(index: int) -> float:
        return 0.01 + 0.002 * index

    store.add(
        [
            Chunk(
                id=f"test-role-{index}",
                artifact_id=f"artifact-test-{index}",
                filename="test_doc.md",
                text="Text",
                embedding=_ranked_embedding(ineligible_offset(index), index),
                source_role="test",
            )
            for index in range(80)
        ]
        + [
            Chunk(
                id=f"disabled-source-{index}",
                artifact_id=f"artifact-disabled-{index}",
                filename="doc.md",
                text="Text",
                embedding=_ranked_embedding(ineligible_offset(index), index + 1),
                connector_id="github",
                connector_source_id="owner/disabled-repo",
            )
            for index in range(80)
        ]
        + [
            Chunk(
                id=f"disabled-connector-{index}",
                artifact_id=f"artifact-jira-{index}",
                filename="doc.md",
                text="Text",
                embedding=_ranked_embedding(ineligible_offset(index), index + 2),
                connector_id="jira",
                connector_source_id="PROJ",
            )
            for index in range(80)
        ]
        + [
            Chunk(
                id="eligible-enabled-repo",
                artifact_id="artifact-enabled",
                filename="doc.md",
                text="Text",
                embedding=_ranked_embedding(0.30, 3),
                connector_id="github",
                connector_source_id="owner/enabled-repo",
            ),
            Chunk(
                id="eligible-legacy",
                artifact_id="artifact-legacy",
                filename="doc.md",
                text="Text",
                embedding=_ranked_embedding(0.31, 5),
            ),
        ]
    )


def test_chroma_query_applies_exclusions_before_the_top_k_cutoff() -> None:
    """Regression: exclusions must reach Chroma's ``where``, not post-filtering.

    All 240 ineligible chunks outrank the two eligible ones, so asking for the
    top 5 and filtering the result returns nothing at all. Putting the
    exclusions in the ``where`` clause — which Chroma applies before it limits
    to ``n_results`` — returns the eligible chunks instead.
    """
    store = ChromaVectorStore(
        collection_name="test_chunks_exclusion_pushdown",
        client=chromadb.EphemeralClient(),
    )
    _exclusion_corpus(store)

    result = store.query(
        embedding=_QUERY_EMBEDDING,
        top_k=5,
        min_score=0.0,
        exclude_roles=frozenset({"test"}),
        exclusions=SourceExclusions(
            connectors=frozenset({"jira"}),
            sources=frozenset({("github", "owner/disabled-repo")}),
        ),
    )

    assert {chunk.id for chunk in result} == {
        "eligible-enabled-repo",
        "eligible-legacy",
    }


def test_chroma_where_exclusions_keep_legacy_chunks() -> None:
    """A chunk with no connector and no role is never excluded server-side.

    Chroma treats a missing/empty metadata value as matching ``$ne``/``$nin``,
    which must agree with ``is_excluded``/``_source_role_from_metadata`` treating
    an absent connector as un-excludable and an absent role as ``primary``.
    """
    store = ChromaVectorStore(
        collection_name="test_chunks_exclusion_legacy",
        client=chromadb.EphemeralClient(),
    )
    store.add(
        [
            Chunk(
                id="legacy",
                artifact_id="artifact-legacy",
                filename="doc.md",
                text="Text",
                embedding=[1.0, 0.0],
            )
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.0,
        exclude_roles=frozenset({"test"}),
        exclusions=SourceExclusions(
            connectors=frozenset({"github"}),
            sources=frozenset({("github", "owner/repo")}),
        ),
    )

    assert [chunk.id for chunk in result] == ["legacy"]


def test_chroma_source_exclusion_keeps_other_sources_of_same_connector() -> None:
    """Excluding one repo must not exclude the whole connector."""
    store = ChromaVectorStore(
        collection_name="test_chunks_exclusion_sibling_source",
        client=chromadb.EphemeralClient(),
    )
    store.add(
        [
            Chunk(
                id="disabled",
                artifact_id="artifact-disabled",
                filename="doc.md",
                text="Text",
                embedding=[1.0, 0.0],
                connector_id="github",
                connector_source_id="owner/disabled-repo",
            ),
            Chunk(
                id="sibling",
                artifact_id="artifact-sibling",
                filename="doc.md",
                text="Text",
                embedding=[1.0, 0.0],
                connector_id="github",
                connector_source_id="owner/other-repo",
            ),
        ]
    )

    result = store.query(
        embedding=[1.0, 0.0],
        top_k=5,
        min_score=0.0,
        exclusions=SourceExclusions(
            sources=frozenset({("github", "owner/disabled-repo")})
        ),
    )

    assert [chunk.id for chunk in result] == ["sibling"]
