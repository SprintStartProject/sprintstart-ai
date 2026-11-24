from corpus.source_filter import SourceExclusions
from corpus.types import Chunk, RetrievalFilters
from rag.retriever import retrieve
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def test_retrieve_returns_chunks_above_min_score() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])

    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Relevant text",
                embedding=[1.0, 0.0],
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Irrelevant text",
                embedding=[0.0, 1.0],
            ),
        ]
    )

    result = retrieve(
        question="What is this about?",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.8,
    )

    assert len(result) == 1
    assert result[0].id == "chunk-1"


def test_retrieve_returns_empty_list_when_no_chunk_passes_threshold() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])

    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Not similar",
                embedding=[0.0, 1.0],
            )
        ]
    )

    result = retrieve(
        question="What is this about?",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.8,
    )

    assert result == []


def test_retrieve_applies_source_exclusions() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])

    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Relevant text",
                embedding=[1.0, 0.0],
                connector_id="github",
                connector_source_id="owner/repo",
            )
        ]
    )

    result = retrieve(
        question="What is this about?",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        exclusions=SourceExclusions(sources=frozenset({("github", "owner/repo")})),
    )

    assert result == []


def test_retrieve_passes_source_filter_to_store() -> None:
    llm = StubLLMClient(embedding=[1.0, 0.0])

    store = StubVectorStore()
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

    result = retrieve(
        question="What changed?",
        llm=llm,
        store=store,
        top_k=5,
        min_score=0.0,
        filters=RetrievalFilters(source_systems=["GITHUB"]),
    )

    assert len(result) == 1
    assert result[0].id == "chunk-code"
    assert result[0].source_system == "GITHUB"
