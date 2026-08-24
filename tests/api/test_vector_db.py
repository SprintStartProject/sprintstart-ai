from collections.abc import Iterable

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.base import LLMStreamEvent, Message, TextDelta
from rag.types import Chunk, ScoredChunk


class StubVectorStore:
    def __init__(self) -> None:
        self.chunks = [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="config.md",
                text="Set OLLAMA_EMBED_MODEL for embeddings.",
                embedding=[1.0, 0.0, 0.0],
                position=0,
                kind="text",
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-2",
                filename="storage.md",
                text="CHROMA_PATH controls vector database persistence.",
                embedding=[0.0, 1.0, 0.0],
                position=1,
                kind="text",
            ),
        ]

    def add(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
    ) -> list[ScoredChunk]:
        results = [
            ScoredChunk(
                id=chunk.id,
                artifact_id=chunk.artifact_id,
                filename=chunk.filename,
                text=chunk.text,
                score=0.95,
                position=chunk.position,
                kind=chunk.kind,
            )
            for chunk in self.chunks
        ]

        return results[:top_k]

    def delete(
        self,
        artifact_id: str,
        exclude_ids: list[str] | None = None,
    ) -> int:
        excluded = set(exclude_ids or [])
        chunks_to_delete = [
            chunk
            for chunk in self.chunks
            if chunk.artifact_id == artifact_id and chunk.id not in excluded
        ]

        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.artifact_id != artifact_id or chunk.id in excluded
        ]

        return len(chunks_to_delete)

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]:
        return self.chunks[offset : offset + limit]

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]:
        chunks = [chunk for chunk in self.chunks if chunk.artifact_id == artifact_id]
        return chunks[offset : offset + limit]

    def count_by_artifact(self, artifact_id: str) -> int:
        return len([chunk for chunk in self.chunks if chunk.artifact_id == artifact_id])

    def all_chunks(self) -> list[Chunk]:
        return self.chunks

    def count(self) -> int:
        return len(self.chunks)


class StubLLMClient:
    def generate(self, messages: list[Message]) -> str:
        return "answer"

    def stream(self, messages: list[Message]) -> Iterable[LLMStreamEvent]:
        yield TextDelta("answer")

    def embed(self, text: str) -> list[float]:
        return [1.0, 0.0, 0.0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def caption_image(self, image_bytes: bytes) -> str:
        return "caption"


def override_store() -> StubVectorStore:
    return StubVectorStore()


def override_llm() -> StubLLMClient:
    return StubLLMClient()


def setup_module() -> None:
    app.dependency_overrides[get_store] = override_store
    app.dependency_overrides[get_llm] = override_llm


def teardown_module() -> None:
    app.dependency_overrides.clear()


client = TestClient(app)


def test_vector_db_status() -> None:
    response = client.get("/api/v1/vector-db/status")

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 2


def test_list_chunks() -> None:
    response = client.get("/api/v1/vector-db/chunks")

    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 2
    assert body["items"][0]["id"] == "chunk-1"
    assert body["items"][0]["filename"] == "config.md"
    assert "heading_path" not in body["items"][0]
    assert "embedding" not in body["items"][0]


def test_list_chunks_with_pagination() -> None:
    response = client.get("/api/v1/vector-db/chunks?limit=1&offset=1")

    assert response.status_code == 200

    body = response.json()
    assert body["limit"] == 1
    assert body["offset"] == 1
    assert body["total"] == 2
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == "chunk-2"


def test_list_chunks_by_artifact() -> None:
    response = client.get("/api/v1/vector-db/artifacts/artifact-1/chunks")

    assert response.status_code == 200

    body = response.json()
    assert body["total"] == 1
    assert len(body["items"]) == 1
    assert body["items"][0]["artifact_id"] == "artifact-1"
    assert body["items"][0]["id"] == "chunk-1"
    assert "heading_path" not in body["items"][0]


def test_delete_artifact_chunks() -> None:
    response = client.delete("/api/v1/vector-db/artifacts/artifact-1")

    assert response.status_code == 204
    assert response.content == b""


def test_delete_unknown_artifact_returns_404() -> None:
    response = client.delete("/api/v1/vector-db/artifacts/missing-artifact")

    assert response.status_code == 404

    body = response.json()
    assert "detail" in body
    assert "missing-artifact" in body["detail"]


def test_search_vector_db() -> None:
    response = client.post(
        "/api/v1/vector-db/search",
        json={
            "query": "Where is OLLAMA_EMBED_MODEL configured?",
            "top_k": 1,
            "min_score": 0.0,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert len(body["items"]) == 1
    assert body["items"][0]["id"] == "chunk-1"
    assert body["items"][0]["score"] == 0.95
    assert "heading_path" not in body["items"][0]
    assert "embedding" not in body["items"][0]
