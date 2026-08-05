import sqlite3
from pathlib import Path

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore

_NOW = "2026-01-01T00:00:00+00:00"


def _record(
    artifact_id: str,
    project_ids: tuple[str, ...] = (),
) -> ArtifactRecord:
    return ArtifactRecord(
        id=artifact_id,
        filename=f"{artifact_id}.md",
        content_type="text/markdown",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at=_NOW,
        updated_at=_NOW,
        project_ids=project_ids,
    )


def test_project_ids_round_trip() -> None:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1", ("project-1", "project-2")))

    record = store.get_artifact("a1")

    assert record is not None
    assert record.project_ids == ("project-1", "project-2")


def test_artifact_without_projects_reads_back_empty() -> None:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1"))

    record = store.get_artifact("a1")

    assert record is not None
    assert record.project_ids == ()


def test_list_artifacts_filters_by_project() -> None:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1", ("project-1",)))
    store.save_completed_artifact(_record("a2", ("project-2",)))
    store.save_completed_artifact(_record("a3", ("project-1", "project-2")))
    store.save_completed_artifact(_record("a4"))

    ids = {r.id for r in store.list_artifacts(project_id="project-1")}

    assert ids == {"a1", "a3"}


def test_list_artifacts_without_project_returns_everything() -> None:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1", ("project-1",)))
    store.save_completed_artifact(_record("a2"))

    ids = {r.id for r in store.list_artifacts()}

    assert ids == {"a1", "a2"}


def test_opening_a_pre_project_database_migrates_the_schema(tmp_path: Path) -> None:
    """A database created before project separation must still open."""
    path = str(tmp_path / "legacy.db")
    connection = sqlite3.connect(path)
    connection.execute(
        """
        CREATE TABLE artifacts (
            id TEXT PRIMARY KEY,
            filename TEXT NOT NULL,
            content_type TEXT NOT NULL,
            source_type TEXT NOT NULL,
            size_bytes INTEGER NOT NULL,
            chunk_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error_message TEXT,
            source_id TEXT,
            source_url TEXT,
            artifact_type TEXT,
            language TEXT
        )
        """
    )
    connection.execute(
        "INSERT INTO artifacts VALUES "
        "('a1', 'old.md', 'text/markdown', 'github', 10, 1, 'completed', "
        f"'{_NOW}', '{_NOW}', NULL, NULL, NULL, NULL, NULL)"
    )
    connection.commit()
    connection.close()

    store = IngestionMetadataStore(path)
    try:
        record = store.get_artifact("a1")
        assert record is not None
        assert record.project_ids == ()

        # The migrated column is writable.
        store.save_completed_artifact(_record("a2", ("project-1",)))
        assert [r.id for r in store.list_artifacts(project_id="project-1")] == ["a2"]
    finally:
        store.close()
