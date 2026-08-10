import sqlite3
from pathlib import Path

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore


def _record(**overrides: object) -> ArtifactRecord:
    defaults: dict[str, object] = dict(
        id="a1",
        filename="issue-1.md",
        content_type="text/plain",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        artifact_type="ISSUE",
    )
    defaults.update(overrides)
    return ArtifactRecord(**defaults)  # type: ignore[arg-type]


def test_round_trips_state_and_labels(tmp_path: Path) -> None:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    try:
        store.save_artifact(_record(state="OPEN", labels=["bug", "good first issue"]))

        loaded = store.get_artifact("a1")

        assert loaded is not None
        assert loaded.state == "OPEN"
        assert loaded.labels == ["bug", "good first issue"]
    finally:
        store.close()


def test_defaults_to_none_state_and_empty_labels(tmp_path: Path) -> None:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    try:
        store.save_artifact(_record())

        loaded = store.get_artifact("a1")

        assert loaded is not None
        assert loaded.state is None
        assert loaded.labels == []
    finally:
        store.close()


def test_list_artifacts_includes_state_and_labels(tmp_path: Path) -> None:
    store = IngestionMetadataStore(path=str(tmp_path / "metadata.db"))
    try:
        store.save_artifact(_record(state="CLOSED", labels=["wontfix"]))

        [loaded] = store.list_artifacts(status="completed")

        assert loaded.state == "CLOSED"
        assert loaded.labels == ["wontfix"]
    finally:
        store.close()


def test_adds_state_and_labels_columns_to_a_pre_existing_database(
    tmp_path: Path,
) -> None:
    # Simulates a DB file created before this migration: no state/labels columns at all.
    db_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(str(db_path))
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
    connection.commit()
    connection.close()

    store = IngestionMetadataStore(path=str(db_path))
    try:
        store.save_artifact(_record(state="OPEN", labels=["good first issue"]))

        loaded = store.get_artifact("a1")

        assert loaded is not None
        assert loaded.state == "OPEN"
        assert loaded.labels == ["good first issue"]
    finally:
        store.close()


def test_reopening_an_up_to_date_database_does_not_fail(tmp_path: Path) -> None:
    path = str(tmp_path / "metadata.db")
    store = IngestionMetadataStore(path=path)
    store.close()

    # The ALTER TABLE columns already exist on the second open; must not raise.
    reopened = IngestionMetadataStore(path=path)
    reopened.close()
