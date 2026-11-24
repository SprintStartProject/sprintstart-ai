import base64
from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from httpx import Response

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from store.metadata_store import IngestionMetadataStore
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_URL = "/api/v1/ingest/sync"
_PNG = base64.b64encode(b"x" * 20).decode("ascii")


@pytest.fixture
def metadata_store(tmp_path: Path) -> Iterable[IngestionMetadataStore]:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    yield store
    store.close()


@pytest.fixture
def vector_store() -> StubVectorStore:
    return StubVectorStore()


@pytest.fixture
def client(
    vector_store: StubVectorStore, metadata_store: IngestionMetadataStore
) -> Iterable[TestClient]:
    app.dependency_overrides[get_store] = lambda: vector_store
    app.dependency_overrides[get_llm] = lambda: StubLLMClient()
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    yield TestClient(app)
    app.dependency_overrides.clear()


def _sync(client: TestClient, body_text: str, **extra: object) -> Response:
    artifact = {
        "artifactId": "a1",
        "sourceSystem": "GITHUB",
        "sourceId": "github:owner/repo:FILE:notes.txt",
        "artifactType": "FILE",
        "title": None,
        "bodyText": body_text,
        "mime": None,
        **extra,
    }
    return client.post(
        _URL, json={"artifactsToIngest": [artifact], "artifactsToDeindex": []}
    )


def _only(response: Response) -> dict[str, object]:
    assert response.status_code == 200, response.text
    (entry,) = response.json()["artifacts"]
    return entry


def test_text_over_the_limit_is_skipped_and_the_old_chunks_are_removed(
    client: TestClient,
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _only(_sync(client, "short"))["chunk_count"] == 1
    monkeypatch.setenv("INGEST_MAX_CONTENT_LENGTH", "20")

    entry = _only(_sync(client, "y" * 21))

    assert (entry["status"], entry["chunk_count"]) == ("completed", 0)
    assert vector_store.count() == 0
    record = metadata_store.get_artifact("a1")
    assert record is not None
    assert (record.status, record.chunk_count) == ("completed", 0)


def test_text_exactly_at_the_limit_is_ingested(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGEST_MAX_CONTENT_LENGTH", "20")

    assert _only(_sync(client, "y" * 20))["chunk_count"] == 1


def test_content_that_parses_to_nothing_completes_and_clears_old_chunks(
    client: TestClient,
    vector_store: StubVectorStore,
    metadata_store: IngestionMetadataStore,
) -> None:
    assert _only(_sync(client, "some text"))["chunk_count"] == 1

    entry = _only(_sync(client, "   "))

    assert (entry["status"], entry["chunk_count"]) == ("completed", 0)
    assert vector_store.count() == 0
    record = metadata_store.get_artifact("a1")
    assert record is not None
    assert (record.status, record.chunk_count) == ("completed", 0)


def test_a_pdf_the_parser_cannot_read_completes_with_no_chunks(
    client: TestClient, metadata_store: IngestionMetadataStore
) -> None:
    not_a_pdf = base64.b64encode(b"this is not a pdf").decode("ascii")

    entry = _only(
        _sync(
            client,
            not_a_pdf,
            sourceSystem="UPLOAD",
            title="spec.pdf",
            mime="application/pdf",
        )
    )

    assert (entry["status"], entry["chunk_count"]) == ("completed", 0)
    record = metadata_store.get_artifact("a1")
    assert record is not None and record.status == "completed"


def test_a_binary_exactly_at_the_binary_limit_is_ingested(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGEST_MAX_BINARY_BYTES", "20")

    entry = _only(
        _sync(client, _PNG, sourceSystem="UPLOAD", title="d.png", mime="image/png")
    )

    assert (entry["status"], entry["chunk_count"]) == ("completed", 1)


def test_a_binary_one_byte_over_the_binary_limit_is_skipped(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("INGEST_MAX_BINARY_BYTES", "19")

    entry = _only(
        _sync(client, _PNG, sourceSystem="UPLOAD", title="d.png", mime="image/png")
    )

    assert (entry["status"], entry["chunk_count"]) == ("completed", 0)


def test_the_record_keeps_the_source_fields_of_the_artifact(
    client: TestClient, metadata_store: IngestionMetadataStore
) -> None:
    _only(
        _sync(
            client,
            "some text",
            sourceUrl="https://example.org/a",
            language="markdown",
            artifactType="FILE",
            state="OPEN",
            hasAssignee=True,
            labels=["good first issue"],
        )
    )

    record = metadata_store.get_artifact("a1")
    assert record is not None
    assert record.source_id == "github:owner/repo:FILE:notes.txt"
    assert record.source_url == "https://example.org/a"
    assert record.language == "markdown"
    assert record.artifact_type == "FILE"
    assert record.state == "OPEN"
    assert record.has_assignee is True
    assert record.labels == ["good first issue"]
    assert record.source_type == "github"
    assert record.content_type == "text/plain"
    assert record.filename == "notes.txt"
