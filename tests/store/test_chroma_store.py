from datetime import UTC, datetime, timedelta
from pathlib import Path

import chromadb

from rag.types import Chunk, RetrievalFilters
from store.chroma_store import ChromaVectorStore


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

    listed = store.all_chunks()
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

    listed = store.all_chunks()
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
