from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_store
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from rag.types import Chunk
from tests.stubs.store import StubVectorStore


def _record(artifact_id: str, project_ids: tuple[str, ...]) -> ArtifactRecord:
    return ArtifactRecord(
        id=artifact_id,
        filename=f"{artifact_id}.md",
        content_type="text/markdown",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at="2026-08-01T00:00:00+00:00",
        updated_at="2026-08-01T00:00:00+00:00",
        project_ids=project_ids,
    )


@pytest.fixture
def clients() -> Iterator[tuple[TestClient, StubVectorStore, IngestionMetadataStore]]:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="artifact-1",
                filename="doc.md",
                text="Shared text",
                embedding=[1.0, 0.0],
                project_ids=("project-a",),
            ),
            Chunk(
                id="chunk-2",
                artifact_id="artifact-2",
                filename="other.md",
                text="Other text",
                embedding=[0.0, 1.0],
                project_ids=("project-a", "project-b"),
            ),
        ]
    )

    metadata_store = IngestionMetadataStore(":memory:")
    metadata_store.save_completed_artifact(_record("artifact-1", ("project-a",)))
    metadata_store.save_completed_artifact(
        _record("artifact-2", ("project-a", "project-b"))
    )

    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store

    yield TestClient(app), store, metadata_store

    app.dependency_overrides.clear()
    metadata_store.close()


def test_sync_links_an_artifact_to_another_project(
    clients: tuple[TestClient, StubVectorStore, IngestionMetadataStore],
) -> None:
    client, store, metadata_store = clients

    response = client.post(
        "/api/v1/artifacts/projects/sync",
        json={
            "artifacts": [
                {
                    "artifactId": "artifact-1",
                    "projectIds": ["project-a", "project-b"],
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "artifacts": [
            {
                "artifact_id": "artifact-1",
                "chunk_count": 1,
                "status": "completed",
                "error_message": None,
            }
        ]
    }
    assert store.project_ids_for_artifact("artifact-1") == frozenset(
        {"project-a", "project-b"}
    )
    stored = metadata_store.get_artifact("artifact-1")
    assert stored is not None
    assert set(stored.project_ids) == {"project-a", "project-b"}


def test_sync_unlinks_an_artifact_from_one_project_only(
    clients: tuple[TestClient, StubVectorStore, IngestionMetadataStore],
) -> None:
    client, store, _ = clients

    response = client.post(
        "/api/v1/artifacts/projects/sync",
        json={"artifacts": [{"artifactId": "artifact-2", "projectIds": ["project-b"]}]},
    )

    assert response.status_code == 200
    assert store.project_ids_for_artifact("artifact-2") == frozenset({"project-b"})
    # Unlinking must not delete the chunks: the artifact is still in project-b.
    assert store.count_by_artifact("artifact-2") == 1


def test_sync_is_idempotent(
    clients: tuple[TestClient, StubVectorStore, IngestionMetadataStore],
) -> None:
    client, store, _ = clients
    payload = {"artifacts": [{"artifactId": "artifact-1", "projectIds": ["project-b"]}]}

    client.post("/api/v1/artifacts/projects/sync", json=payload)
    client.post("/api/v1/artifacts/projects/sync", json=payload)

    assert store.project_ids_for_artifact("artifact-1") == frozenset({"project-b"})
    assert store.count_by_artifact("artifact-1") == 1


def test_sync_reports_an_unknown_artifact_as_zero_chunks(
    clients: tuple[TestClient, StubVectorStore, IngestionMetadataStore],
) -> None:
    client, _, _ = clients

    response = client.post(
        "/api/v1/artifacts/projects/sync",
        json={"artifacts": [{"artifactId": "missing", "projectIds": ["project-a"]}]},
    )

    assert response.status_code == 200
    entry = response.json()["artifacts"][0]
    assert entry["chunk_count"] == 0
    assert entry["status"] == "completed"


def test_sync_rejects_a_project_id_containing_the_delimiter(
    clients: tuple[TestClient, StubVectorStore, IngestionMetadataStore],
) -> None:
    client, _, _ = clients

    response = client.post(
        "/api/v1/artifacts/projects/sync",
        json={"artifacts": [{"artifactId": "artifact-1", "projectIds": ["a|b"]}]},
    )

    assert response.status_code == 422


def test_delete_project_memberships_purges_one_project(
    clients: tuple[TestClient, StubVectorStore, IngestionMetadataStore],
) -> None:
    client, store, metadata_store = clients

    response = client.delete("/api/v1/projects/project-a/memberships")

    assert response.status_code == 200
    assert response.json() == {
        "project_id": "project-a",
        "chunk_count": 2,
        "artifact_count": 2,
    }
    assert store.project_ids_for_artifact("artifact-1") == frozenset()
    assert store.project_ids_for_artifact("artifact-2") == frozenset({"project-b"})
    # The artifacts survive -- artifact-2 is still readable from project-b.
    assert store.count() == 2

    stored = metadata_store.get_artifact("artifact-2")
    assert stored is not None
    assert stored.project_ids == ("project-b",)


def test_delete_project_memberships_of_an_unused_project_is_a_no_op(
    clients: tuple[TestClient, StubVectorStore, IngestionMetadataStore],
) -> None:
    client, store, _ = clients

    response = client.delete("/api/v1/projects/project-zzz/memberships")

    assert response.status_code == 200
    assert response.json()["chunk_count"] == 0
    assert store.count() == 2
