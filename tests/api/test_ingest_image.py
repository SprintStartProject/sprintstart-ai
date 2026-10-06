from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.errors import LLMUnavailableError
from rag.retriever import retrieve
from store.chroma_store import ChromaVectorStore
from tests.conftest import llm_required, vision_required
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

# Minimal 1×1 red PNG encoded as base64 string (what the client sends)
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
)


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubVectorStore], Any, None]:
    llm = StubLLMClient(caption="A tiny red pixel on a white background.")
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), store

    app.dependency_overrides.clear()


def test_ingest_png_returns_one_chunk(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, store = client

    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "image-1",
            "filename": "diagram.png",
            "content": _TINY_PNG_B64,
        },
    )

    assert response.status_code == 200
    assert response.json()["artifact_id"] == "image-1"
    assert response.json()["chunk_count"] == 1
    assert len(store.chunks) == 1
    assert store.chunks[0].kind == "image"
    assert "red pixel" in store.chunks[0].text


def test_ingest_jpg_returns_one_chunk(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, _ = client

    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "image-jpg",
            "filename": "photo.jpg",
            "content": _TINY_PNG_B64,
        },
    )

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 1


def test_ingest_image_vision_unavailable_returns_zero_chunks(
    client: tuple[TestClient, StubVectorStore],
) -> None:
    http_client, _ = client

    class NoVisionLLM(StubLLMClient):
        def caption_image(self, image_bytes: bytes) -> str:
            raise LLMUnavailableError("http://localhost:11434")

    app.dependency_overrides[get_llm] = lambda: NoVisionLLM()

    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "image-2",
            "filename": "diagram.png",
            "content": _TINY_PNG_B64,
        },
    )

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 0


@pytest.fixture
def real_image_client() -> Generator[tuple[TestClient, ChromaVectorStore], Any, None]:
    store = ChromaVectorStore(collection_name="image-integration-test")
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), store
    app.dependency_overrides.clear()


@pytest.mark.integration
@llm_required
@vision_required
def test_ingest_image_and_query_with_real_llm(
    real_image_client: tuple[TestClient, ChromaVectorStore],
) -> None:
    http_client, store = real_image_client

    response = http_client.post(
        "/api/v1/ingest",
        json={
            "artifact_id": "integration-image",
            "filename": "test_diagram.png",
            "content": _TINY_PNG_B64,
        },
    )
    assert response.status_code == 200
    assert response.json()["chunk_count"] == 1

    chunks = retrieve(
        "What does the uploaded image show?",
        get_llm(),
        store,
        min_score=0.1,
    )
    assert "test_diagram.png" in {chunk.filename for chunk in chunks}
