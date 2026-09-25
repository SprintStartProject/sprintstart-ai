from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore

_URL = "/api/v1/ingest/status"
_CREATED = "2026-01-01T00:00:00+00:00"
_UPDATED = "2026-01-02T00:00:00+00:00"


@pytest.fixture
def metadata_store(tmp_path: Path) -> Iterable[IngestionMetadataStore]:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    yield store
    store.close()


@pytest.fixture
def client(metadata_store: IngestionMetadataStore) -> Iterable[TestClient]:
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    yield TestClient(app)

    app.dependency_overrides.clear()


def _save(
    store: IngestionMetadataStore,
    artifact_id: str,
    chunk_count: int = 2,
) -> None:
    store.save_completed_artifact(
        ArtifactRecord(
            id=artifact_id,
            filename=f"{artifact_id}.md",
            content_type="text/markdown",
            source_type="github",
            size_bytes=10,
            chunk_count=chunk_count,
            status="completed",
            created_at=_CREATED,
            updated_at=_UPDATED,
        )
    )


def test_completed_artifact_is_reported_as_indexed(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    _save(metadata_store, "a1", chunk_count=3)

    response = client.get(_URL, params={"artifact_ids": ["a1"]})

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "artifact_id": "a1",
                "status": "indexed",
                "updated_at": _UPDATED,
                "chunk_count": 3,
            }
        ]
    }


def test_unknown_id_is_reported_with_null_details(client: TestClient) -> None:
    response = client.get(_URL, params={"artifact_ids": ["never-ingested"]})

    assert response.status_code == 200
    assert response.json() == {
        "items": [
            {
                "artifact_id": "never-ingested",
                "status": "unknown",
                "updated_at": None,
                "chunk_count": None,
            }
        ]
    }


def test_every_recorded_status_keeps_its_name_except_completed(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    for artifact_id in ("done", "broken", "gone"):
        _save(metadata_store, artifact_id)
    metadata_store.mark_failed("broken", "parse error", _UPDATED)
    metadata_store.mark_deindexed("gone", _UPDATED)
    metadata_store.save_artifact(
        ArtifactRecord(
            id="busy",
            filename="busy.md",
            content_type="text/markdown",
            source_type="github",
            size_bytes=10,
            chunk_count=0,
            status="processing",
            created_at=_CREATED,
            updated_at=_CREATED,
        )
    )

    response = client.get(
        _URL, params={"artifact_ids": ["done", "broken", "gone", "busy"]}
    )

    statuses = {i["artifact_id"]: i["status"] for i in response.json()["items"]}
    assert statuses == {
        "done": "indexed",
        "broken": "failed",
        "gone": "deindexed",
        "busy": "processing",
    }


def test_items_follow_request_order_without_duplicates(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    _save(metadata_store, "a1")
    _save(metadata_store, "a2")

    response = client.get(
        _URL, params={"artifact_ids": ["a2", "x", "a1", "a2", "x", "a1"]}
    )

    assert response.status_code == 200
    ids = [item["artifact_id"] for item in response.json()["items"]]
    assert ids == ["a2", "x", "a1"]


def test_batch_of_exactly_the_limit_is_accepted(client: TestClient) -> None:
    ids = [f"id-{i}" for i in range(100)]

    response = client.get(_URL, params={"artifact_ids": ids})

    assert response.status_code == 200
    assert [i["artifact_id"] for i in response.json()["items"]] == ids


def test_batch_above_the_limit_is_rejected(client: TestClient) -> None:
    ids = [f"id-{i}" for i in range(101)]

    response = client.get(_URL, params={"artifact_ids": ids})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["query", "artifact_ids"]


def test_no_ids_returns_no_items(client: TestClient) -> None:
    response = client.get(_URL)

    assert response.status_code == 200
    assert response.json() == {"items": []}


def test_blank_id_is_rejected(client: TestClient) -> None:
    response = client.get(f"{_URL}?artifact_ids=a1&artifact_ids=")

    assert response.status_code == 422


def test_status_lookup_writes_nothing(
    client: TestClient,
    metadata_store: IngestionMetadataStore,
) -> None:
    _save(metadata_store, "a1")
    revision = metadata_store.corpus_revision()
    changes = metadata_store._connection.total_changes

    response = client.get(_URL, params={"artifact_ids": ["a1", "missing"]})

    assert response.status_code == 200
    assert metadata_store._connection.total_changes == changes
    assert metadata_store.corpus_revision() == revision
    assert metadata_store.get_artifact("missing") is None
