import sqlite3
from pathlib import Path

import pytest

from ingestion import metadata_store
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore

_NOW = "2026-01-01T00:00:00+00:00"


def _record(
    artifact_id: str = "a1",
    project_ids: tuple[str, ...] = (),
    **overrides: object,
) -> ArtifactRecord:
    defaults: dict[str, object] = dict(
        id=artifact_id,
        filename=f"{artifact_id}.md",
        content_type="text/markdown",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at=_NOW,
        updated_at=_NOW,
        artifact_type="ISSUE",
        project_ids=project_ids,
    )
    defaults.update(overrides)
    return ArtifactRecord(**defaults)  # type: ignore[arg-type]


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


def test_list_artifacts_filters_by_project_ids_plural() -> None:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1", ("project-1",)))
    store.save_completed_artifact(_record("a2", ("project-2",)))
    store.save_completed_artifact(_record("a3", ("project-3",)))
    store.save_completed_artifact(_record("a4"))

    ids = {r.id for r in store.list_artifacts(project_ids=["project-1", "project-2"])}
    assert ids == {"a1", "a2"}

    # Empty project_ids admits nothing (fail-closed)
    assert store.list_artifacts(project_ids=[]) == []


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


def test_remove_project_leaves_wildcard_lookalikes_untouched() -> None:
    """``_`` is a ``LIKE`` wildcard, so ``ab_d`` would otherwise match ``abcd``."""
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1", ("ab_d",)))
    store.save_completed_artifact(_record("a2", ("abcd",)))

    assert store.remove_project("ab_d", "2026-02-02T00:00:00+00:00") == 1

    lookalike = store.get_artifact("a2")
    assert lookalike is not None
    assert lookalike.project_ids == ("abcd",)
    assert lookalike.updated_at == _NOW


def test_remove_project_leaves_case_variants_untouched() -> None:
    """SQLite's ``LIKE`` is ASCII case-insensitive; project ids are not."""
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1", ("Proj-A",)))
    store.save_completed_artifact(_record("a2", ("proj-a",)))

    assert store.remove_project("Proj-A", "2026-02-02T00:00:00+00:00") == 1

    variant = store.get_artifact("a2")
    assert variant is not None
    assert variant.project_ids == ("proj-a",)
    assert variant.updated_at == _NOW


def test_corpus_revision_is_shared_and_monotonic_across_connections(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "metadata.db")
    first = IngestionMetadataStore(path)
    second = IngestionMetadataStore(path)

    try:
        assert first.corpus_revision() == second.corpus_revision() == 0

        assert first.bump_corpus_revision() == 1
        assert second.corpus_revision() == 1

        assert second.bump_corpus_revision() == 2
        assert first.corpus_revision() == 2
    finally:
        first.close()
        second.close()


def test_revocation_tombstones_are_shared_revisioned_and_idempotent(
    tmp_path: Path,
) -> None:
    path = str(tmp_path / "metadata.db")
    writer = IngestionMetadataStore(path)
    reader = IngestionMetadataStore(path)

    try:
        corpus_before = reader.corpus_revision()
        assert writer.revoke_artifacts(["artifact-1", "artifact-1"]) is True

        revision, revoked = reader.revocation_snapshot()
        assert revision == reader.revocation_revision() == 1
        assert revoked == frozenset({"artifact-1"})
        assert reader.corpus_revision() == corpus_before + 1

        # An idempotent retry changes neither shared revision.
        assert reader.revoke_artifacts(["artifact-1"]) is False
        assert reader.revocation_revision() == revision
        assert reader.corpus_revision() == corpus_before + 1

        assert writer.clear_artifact_revocations(["artifact-1"]) is True
        assert reader.revocation_snapshot() == (2, frozenset())
        assert reader.corpus_revision() == corpus_before + 2

        assert reader.clear_artifact_revocations(["artifact-1"]) is False
        assert reader.revocation_snapshot() == (2, frozenset())
    finally:
        writer.close()
        reader.close()


def test_revocation_reasons_cannot_clear_each_other(tmp_path: Path) -> None:
    store = IngestionMetadataStore(str(tmp_path / "metadata.db"))

    try:
        assert store.revoke_artifacts(["artifact-1"], reason="delete") is True
        assert store.revoke_artifacts(["artifact-1"], reason="membership") is True
        assert store.revocation_snapshot() == (
            2,
            frozenset({"artifact-1"}),
        )

        assert (
            store.clear_artifact_revocations(
                ["artifact-1"],
                reason="membership",
            )
            is True
        )
        assert store.revocation_snapshot() == (
            3,
            frozenset({"artifact-1"}),
        )

        assert (
            store.clear_artifact_revocations(
                ["artifact-1"],
                reason="delete",
            )
            is True
        )
        assert store.revocation_snapshot() == (4, frozenset())
    finally:
        store.close()


def test_newer_revocation_operation_cannot_be_cleared_by_older_owner_token(
    tmp_path: Path,
) -> None:
    store = IngestionMetadataStore(str(tmp_path / "metadata.db"))

    try:
        first = store.begin_revocation_operations(
            ["artifact-1"],
            reason="membership",
            owner="project:a",
        )
        second = store.begin_revocation_operations(
            ["artifact-1"],
            reason="membership",
            owner="project:a",
        )

        assert first == {"artifact-1": 1}
        assert second == {"artifact-1": 2}
        assert (
            store.complete_revocation_operations(
                first,
                reason="membership",
                owner="project:a",
            )
            is False
        )
        assert store.revocation_snapshot()[1] == frozenset({"artifact-1"})

        assert (
            store.complete_revocation_operations(
                second,
                reason="membership",
                owner="project:a",
            )
            is True
        )
        assert store.revocation_snapshot()[1] == frozenset()
    finally:
        store.close()


def test_only_failed_revocation_operations_are_recoverable(tmp_path: Path) -> None:
    store = IngestionMetadataStore(str(tmp_path / "metadata.db"))

    try:
        operation = store.begin_revocation_operations(
            ["artifact-1"],
            reason="membership",
            owner="project:a",
        )

        assert (
            store.revocation_operations_for_artifact(
                "artifact-1",
                reason="membership",
            )
            == {}
        )
        assert (
            store.fail_revocation_operations(
                operation,
                reason="membership",
                owner="project:a",
            )
            is True
        )
        assert store.revocation_operations_for_artifact(
            "artifact-1",
            reason="membership",
        ) == {"project:a": 1}

        retry = store.begin_revocation_operations(
            ["artifact-1"],
            reason="membership",
            owner="project:a",
        )
        assert retry == {"artifact-1": 2}
        assert (
            store.revocation_operations_for_artifact(
                "artifact-1",
                reason="membership",
            )
            == {}
        )
        assert (
            store.fail_revocation_operations(
                operation,
                reason="membership",
                owner="project:a",
            )
            is False
        )
    finally:
        store.close()


def test_get_artifacts_returns_known_ids_and_omits_unknown() -> None:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1", chunk_count=3))
    store.save_completed_artifact(_record("a2", status="failed"))

    records = store.get_artifacts(["a2", "missing", "a1", "a2"])

    assert set(records) == {"a1", "a2"}
    assert records["a1"].chunk_count == 3
    assert records["a2"].status == "failed"


def test_get_artifacts_without_ids_returns_empty() -> None:
    store = IngestionMetadataStore(":memory:")
    store.save_completed_artifact(_record("a1"))

    assert store.get_artifacts([]) == {}


def test_get_artifacts_reads_a_batch_in_one_query() -> None:
    store = IngestionMetadataStore(":memory:")
    ids = [f"a{i}" for i in range(100)]
    for artifact_id in ids:
        store.save_completed_artifact(_record(artifact_id))
    statements: list[str] = []
    store._connection.set_trace_callback(statements.append)

    records = store.get_artifacts(ids)

    assert set(records) == set(ids)
    selects = [s for s in statements if s.lstrip().upper().startswith("SELECT")]
    assert len(selects) == 1


def test_get_artifacts_splits_batches_above_the_parameter_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(metadata_store, "_MAX_IN_PARAMS", 2)
    store = IngestionMetadataStore(":memory:")
    for artifact_id in ("a1", "a2", "a3", "a4", "a5"):
        store.save_completed_artifact(_record(artifact_id))

    records = store.get_artifacts(["a5", "a1", "missing", "a3", "a2", "a4"])

    assert set(records) == {"a1", "a2", "a3", "a4", "a5"}
