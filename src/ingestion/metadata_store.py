from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Literal, cast

from rag.filters import (
    decode_project_ids,
    encode_project_ids,
    encoded_project_marker,
)

IngestionStatus = Literal["processing", "completed", "failed", "deindexed"]
RevocationReason = Literal["delete", "membership"]

_COLUMNS = (
    "id",
    "filename",
    "content_type",
    "source_type",
    "size_bytes",
    "chunk_count",
    "status",
    "created_at",
    "updated_at",
    "error_message",
    "source_id",
    "source_url",
    "artifact_type",
    "language",
    "project_ids",
    "state",
    "has_assignee",
    "labels",
)


@dataclass(frozen=True)
class ArtifactRecord:
    id: str
    filename: str
    content_type: str
    source_type: str
    size_bytes: int
    chunk_count: int
    status: IngestionStatus
    created_at: str
    updated_at: str
    error_message: str | None = None
    source_id: str | None = None
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None
    # Projects the artifact belongs to; empty for artifacts ingested before the
    # backend started sending them (see ``list_artifacts``).
    project_ids: tuple[str, ...] = ()
    # Issue state at the tracker (e.g. "OPEN"/"CLOSED") and labels (e.g. "good
    # first issue"); both unset for non-issue artifacts. Used by starter-work
    # mining to deterministically exclude closed issues rather than relying on
    # an LLM to notice.
    state: str | None = None
    # Whether somebody at the source is already assigned to this issue, or None
    # when the connector cannot tell. Mining withholds an issue on a definite
    # True -- work somebody else has taken is not work a hire can pick up.
    # None is "unknown", never "nobody": GitHub issues have assignees this
    # system does not ingest, and reading that absence as "free" would be the
    # same defect as reading an absent history as "beginner".
    has_assignee: bool | None = None
    labels: list[str] = field(default_factory=list[str])


class IngestionMetadataStore:
    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = RLock()

        if path != ":memory:":
            Path(path).parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifacts (
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
                    language TEXT,
                    project_ids TEXT,
                    state TEXT,
                    has_assignee INTEGER,
                    labels TEXT
                )
                """
            )
            self._migrate_columns()
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS corpus_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    revision INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO corpus_state (singleton, revision) VALUES (1, 0)"
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS artifact_revocations (
                    artifact_id TEXT NOT NULL,
                    reason TEXT NOT NULL
                        CHECK (reason IN ('delete', 'membership')),
                    owner TEXT NOT NULL DEFAULT '',
                    generation INTEGER NOT NULL DEFAULT 1,
                    revoked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (artifact_id, reason, owner)
                )
                """
            )
            revocation_columns = {
                str(row["name"])
                for row in self._connection.execute(
                    "PRAGMA table_info(artifact_revocations)"
                ).fetchall()
            }
            if "reason" not in revocation_columns:
                # Migrate databases created by the first, pre-review version
                # of this branch without dropping a live delete tombstone.
                self._connection.execute(
                    "ALTER TABLE artifact_revocations "
                    "RENAME TO artifact_revocations_legacy"
                )
                self._connection.execute(
                    """
                    CREATE TABLE artifact_revocations (
                        artifact_id TEXT NOT NULL,
                        reason TEXT NOT NULL
                            CHECK (reason IN ('delete', 'membership')),
                        owner TEXT NOT NULL DEFAULT '',
                        generation INTEGER NOT NULL DEFAULT 1,
                        revoked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (artifact_id, reason, owner)
                    )
                    """
                )
                self._connection.execute(
                    """
                    INSERT INTO artifact_revocations
                        (artifact_id, reason, revoked_at)
                    SELECT artifact_id, 'delete', revoked_at
                    FROM artifact_revocations_legacy
                    """
                )
                self._connection.execute("DROP TABLE artifact_revocations_legacy")
            else:
                revocation_columns = {
                    str(row["name"])
                    for row in self._connection.execute(
                        "PRAGMA table_info(artifact_revocations)"
                    ).fetchall()
                }
                if "generation" not in revocation_columns:
                    self._connection.execute(
                        "ALTER TABLE artifact_revocations "
                        "ADD COLUMN generation INTEGER NOT NULL DEFAULT 1"
                    )
                if "owner" not in revocation_columns:
                    self._connection.execute(
                        "ALTER TABLE artifact_revocations "
                        "RENAME TO artifact_revocations_legacy_owner"
                    )
                    self._connection.execute(
                        """
                        CREATE TABLE artifact_revocations (
                            artifact_id TEXT NOT NULL,
                            reason TEXT NOT NULL
                                CHECK (reason IN ('delete', 'membership')),
                            owner TEXT NOT NULL DEFAULT '',
                            generation INTEGER NOT NULL DEFAULT 1,
                            revoked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                            PRIMARY KEY (artifact_id, reason, owner)
                        )
                        """
                    )
                    self._connection.execute(
                        """
                        INSERT INTO artifact_revocations
                            (artifact_id, reason, owner, generation, revoked_at)
                        SELECT artifact_id, reason, '', generation, revoked_at
                        FROM artifact_revocations_legacy_owner
                        """
                    )
                    self._connection.execute(
                        "DROP TABLE artifact_revocations_legacy_owner"
                    )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS revocation_state (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    revision INTEGER NOT NULL
                )
                """
            )
            self._connection.execute(
                "INSERT OR IGNORE INTO revocation_state "
                "(singleton, revision) VALUES (1, 0)"
            )

            self._connection.commit()

    def _migrate_columns(self) -> None:
        """Add columns introduced after a database was first created.

        ``project_ids`` was added with the project-separation work, and
        ``state``/``has_assignee``/``labels`` with starter-work mining; databases
        created before either exist in the wild, and SQLite has no
        ``ADD COLUMN IF NOT EXISTS``.
        """
        cursor = self._connection.execute("PRAGMA table_info(artifacts)")
        existing = {str(row["name"]) for row in cursor.fetchall()}
        for column, decl in (
            ("project_ids", "TEXT"),
            ("state", "TEXT"),
            ("has_assignee", "INTEGER"),
            ("labels", "TEXT"),
        ):
            if column not in existing:
                self._connection.execute(
                    f"ALTER TABLE artifacts ADD COLUMN {column} {decl}"
                )

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def corpus_revision(self) -> int:
        """Return the corpus revision shared by every instance using this database."""
        with self._lock:
            row = self._connection.execute(
                "SELECT revision FROM corpus_state WHERE singleton = 1"
            ).fetchone()

        if row is None:
            raise RuntimeError("Corpus revision row is missing")

        return int(row["revision"])

    def bump_corpus_revision(self) -> int:
        """Atomically advance and return the shared corpus revision."""
        with self._lock:
            self._connection.execute(
                "UPDATE corpus_state SET revision = revision + 1 WHERE singleton = 1"
            )
            row = self._connection.execute(
                "SELECT revision FROM corpus_state WHERE singleton = 1"
            ).fetchone()
            self._connection.commit()

        if row is None:
            raise RuntimeError("Corpus revision row is missing")

        return int(row["revision"])

    def revocation_revision(self) -> int:
        """Return the cheap, shared revision for the live tombstone set."""
        with self._lock:
            row = self._connection.execute(
                "SELECT revision FROM revocation_state WHERE singleton = 1"
            ).fetchone()

        if row is None:
            raise RuntimeError("Revocation revision row is missing")

        return int(row["revision"])

    def revocation_snapshot(self) -> tuple[int, frozenset[str]]:
        """Read one consistent revision/tombstone snapshot in a single query."""
        with self._lock:
            rows = self._connection.execute(
                """
                SELECT revocation_state.revision, artifact_revocations.artifact_id
                FROM revocation_state
                LEFT JOIN artifact_revocations ON 1 = 1
                WHERE revocation_state.singleton = 1
                """
            ).fetchall()

        if not rows:
            raise RuntimeError("Revocation revision row is missing")

        return (
            int(rows[0]["revision"]),
            frozenset(
                str(row["artifact_id"])
                for row in rows
                if row["artifact_id"] is not None
            ),
        )

    def revoke_artifacts(
        self,
        artifact_ids: Sequence[str],
        reason: RevocationReason = "delete",
    ) -> bool:
        """Durably hide artifacts for one operation class."""
        normalized = tuple(dict.fromkeys(value for value in artifact_ids if value))
        if not normalized:
            return False

        with self._lock:
            try:
                before = self._connection.total_changes
                self._connection.executemany(
                    "INSERT OR IGNORE INTO artifact_revocations "
                    "(artifact_id, reason) VALUES (?, ?)",
                    ((artifact_id, reason) for artifact_id in normalized),
                )
                changed = self._connection.total_changes > before
                if changed:
                    self._connection.execute(
                        "UPDATE revocation_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                    self._connection.execute(
                        "UPDATE corpus_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

        return changed

    def begin_revocation_operations(
        self,
        artifact_ids: Sequence[str],
        reason: RevocationReason,
        owner: str,
    ) -> dict[str, int]:
        """Start revocations and return tokens that only their owner can clear.

        Retrying an operation advances the generation of an existing tombstone.
        A completion carrying an older token therefore cannot clear protection
        installed by a newer concurrent operation.
        """
        normalized = tuple(dict.fromkeys(value for value in artifact_ids if value))
        if not normalized:
            return {}

        generations: dict[str, int] = {}
        visible_change = False
        with self._lock:
            try:
                for artifact_id in normalized:
                    inserted = self._connection.execute(
                        "INSERT OR IGNORE INTO artifact_revocations "
                        "(artifact_id, reason, owner, generation) "
                        "VALUES (?, ?, ?, 1)",
                        (artifact_id, reason, owner),
                    )
                    if inserted.rowcount > 0:
                        generation = 1
                        visible_change = True
                    else:
                        self._connection.execute(
                            "UPDATE artifact_revocations "
                            "SET generation = generation + 1 "
                            "WHERE artifact_id = ? AND reason = ? AND owner = ?",
                            (artifact_id, reason, owner),
                        )
                        row = self._connection.execute(
                            "SELECT generation FROM artifact_revocations "
                            "WHERE artifact_id = ? AND reason = ? AND owner = ?",
                            (artifact_id, reason, owner),
                        ).fetchone()
                        if row is None:
                            raise RuntimeError("Revocation operation row is missing")
                        generation = int(row["generation"])
                    generations[artifact_id] = generation

                if visible_change:
                    self._connection.execute(
                        "UPDATE revocation_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                    self._connection.execute(
                        "UPDATE corpus_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

        return generations

    def complete_revocation_operations(
        self,
        generations: Mapping[str, int],
        reason: RevocationReason,
        owner: str,
    ) -> bool:
        """Clear only tombstones still owned by the supplied operations."""
        normalized = {
            artifact_id: generation
            for artifact_id, generation in generations.items()
            if artifact_id
        }
        if not normalized:
            return False

        with self._lock:
            try:
                before = self._connection.total_changes
                self._connection.executemany(
                    "DELETE FROM artifact_revocations "
                    "WHERE artifact_id = ? AND reason = ? AND owner = ? "
                    "AND generation = ?",
                    (
                        (artifact_id, reason, owner, generation)
                        for artifact_id, generation in normalized.items()
                    ),
                )
                changed = self._connection.total_changes > before
                if changed:
                    self._connection.execute(
                        "UPDATE revocation_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                    self._connection.execute(
                        "UPDATE corpus_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

        return changed

    def revocation_artifacts(
        self,
        reason: RevocationReason,
        owner: str,
    ) -> frozenset[str]:
        """Return artifacts protected by one retryable operation scope."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT artifact_id FROM artifact_revocations "
                "WHERE reason = ? AND owner = ?",
                (reason, owner),
            ).fetchall()
        return frozenset(str(row["artifact_id"]) for row in rows)

    def revocation_operations_for_artifact(
        self,
        artifact_id: str,
        reason: RevocationReason,
    ) -> dict[str, int]:
        """Snapshot live operation owners so a later cleanup is conditional."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT owner, generation FROM artifact_revocations "
                "WHERE artifact_id = ? AND reason = ?",
                (artifact_id, reason),
            ).fetchall()
        return {str(row["owner"]): int(row["generation"]) for row in rows}

    def clear_artifact_revocations(
        self,
        artifact_ids: Sequence[str],
        reason: RevocationReason = "delete",
    ) -> bool:
        """Clear only the confirmed operation class for each artifact."""
        normalized = tuple(dict.fromkeys(value for value in artifact_ids if value))
        if not normalized:
            return False

        with self._lock:
            try:
                before = self._connection.total_changes
                self._connection.executemany(
                    "DELETE FROM artifact_revocations "
                    "WHERE artifact_id = ? AND reason = ?",
                    ((artifact_id, reason) for artifact_id in normalized),
                )
                changed = self._connection.total_changes > before
                if changed:
                    self._connection.execute(
                        "UPDATE revocation_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                    self._connection.execute(
                        "UPDATE corpus_state "
                        "SET revision = revision + 1 WHERE singleton = 1"
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

        return changed

    def save_artifact(self, artifact: ArtifactRecord) -> None:
        with self._lock:
            self._upsert_artifact(artifact)
            self._connection.commit()

    def save_completed_artifact(self, artifact: ArtifactRecord) -> None:
        with self._lock:
            self._upsert_artifact(artifact)
            self._connection.commit()

    def mark_failed(
        self,
        artifact_id: str,
        error_message: str,
        updated_at: str,
    ) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE artifacts
                SET status = ?,
                    updated_at = ?,
                    error_message = ?
                WHERE id = ?
                """,
                ("failed", updated_at, error_message, artifact_id),
            )
            self._connection.commit()

    def mark_deindexed(self, artifact_id: str, updated_at: str) -> None:
        with self._lock:
            self._connection.execute(
                """
                UPDATE artifacts
                SET status = ?,
                    chunk_count = ?,
                    updated_at = ?,
                    error_message = ?
                WHERE id = ?
                """,
                ("deindexed", 0, updated_at, None, artifact_id),
            )
            self._connection.commit()

    def get_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        with self._lock:
            cursor = self._connection.execute(
                f"SELECT {', '.join(_COLUMNS)} FROM artifacts WHERE id = ?",
                (artifact_id,),
            )
            row = cast(sqlite3.Row | None, cursor.fetchone())

        if row is None:
            return None

        return self._row_to_record(row)

    def list_artifacts(
        self,
        status: IngestionStatus | None = "completed",
        project_id: str | None = None,
        project_ids: Sequence[str] | None = None,
    ) -> list[ArtifactRecord]:
        """Return all artifacts, optionally filtered by status and project.

        Used by corpus-wide insights (e.g. knowledge-gap detection) and starter-work
        mining that need to enumerate the ingestion index rather than look up a single
        id. Defaults to ``completed`` so callers see only fully-indexed material.

        ``project_id`` and ``project_ids`` scope the result to one or more projects.
        Like retrieval, it is fail-closed: artifacts with no recorded project are
        excluded, and an empty ``project_ids`` admits nothing.
        """
        query = f"SELECT {', '.join(_COLUMNS)} FROM artifacts"
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)

        with self._lock:
            cursor = self._connection.execute(query, params)
            rows = cursor.fetchall()

        records = [self._row_to_record(cast(sqlite3.Row, row)) for row in rows]

        if project_id is not None:
            records = [record for record in records if project_id in record.project_ids]

        if project_ids is not None:
            target_pids = set(project_ids)
            records = [
                record
                for record in records
                if any(pid in target_pids for pid in record.project_ids)
            ]

        return records

    def set_project_ids(
        self,
        artifact_id: str,
        project_ids: tuple[str, ...],
        updated_at: str,
    ) -> bool:
        """Replace an artifact's recorded project membership.

        Kept in step with the vector store so project-scoped insights (which
        read this index, not the chunks) see the same memberships retrieval
        does. Returns whether a row was actually updated.
        """
        normalized = tuple(dict.fromkeys(pid for pid in project_ids if pid))
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE artifacts
                SET project_ids = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (encode_project_ids(normalized), updated_at, artifact_id),
            )
            self._connection.commit()

        return cursor.rowcount > 0

    def remove_project(self, project_id: str, updated_at: str) -> int:
        """Drop one project from every artifact that records it.

        Matched on the encoded form (``|id|``) rather than a bare substring, so
        an id that happens to be a prefix of another never matches.

        The ``LIKE`` is only a prefilter: ``_`` and ``%`` in an id are wildcards
        and SQLite's ``LIKE`` is ASCII case-insensitive, so it can select rows
        that carry a *different* project (``ab_d`` matching ``abcd``). Both of
        those widen the match and neither can narrow it, so deciding membership
        on the decoded ids below is what makes the rewrite exact -- and counting
        only the rows actually rewritten is what makes the returned
        ``artifact_count`` true.
        """
        marker = encoded_project_marker(project_id)
        with self._lock:
            rows = self._connection.execute(
                "SELECT id, project_ids FROM artifacts WHERE project_ids LIKE ?",
                (f"%{marker}%",),
            ).fetchall()

            updated = 0
            for row in rows:
                recorded = decode_project_ids(row["project_ids"])
                if project_id not in recorded:
                    continue

                remaining = tuple(pid for pid in recorded if pid != project_id)
                self._connection.execute(
                    "UPDATE artifacts SET project_ids = ?, updated_at = ? WHERE id = ?",
                    (encode_project_ids(remaining), updated_at, str(row["id"])),
                )
                updated += 1

            self._connection.commit()

        return updated

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> ArtifactRecord:
        return ArtifactRecord(
            id=str(row["id"]),
            filename=str(row["filename"]),
            content_type=str(row["content_type"]),
            source_type=str(row["source_type"]),
            size_bytes=int(row["size_bytes"]),
            chunk_count=int(row["chunk_count"]),
            status=cast(IngestionStatus, str(row["status"])),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            error_message=(
                None if row["error_message"] is None else str(row["error_message"])
            ),
            source_id=None if row["source_id"] is None else str(row["source_id"]),
            source_url=None if row["source_url"] is None else str(row["source_url"]),
            artifact_type=(
                None if row["artifact_type"] is None else str(row["artifact_type"])
            ),
            language=None if row["language"] is None else str(row["language"]),
            project_ids=decode_project_ids(row["project_ids"]),
            state=None if row["state"] is None else str(row["state"]),
            has_assignee=(
                None if row["has_assignee"] is None else bool(row["has_assignee"])
            ),
            labels=json.loads(row["labels"]) if row["labels"] else [],
        )

    def _upsert_artifact(self, artifact: ArtifactRecord) -> None:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        self._connection.execute(
            f"INSERT OR REPLACE INTO artifacts ({', '.join(_COLUMNS)}) "
            f"VALUES ({placeholders})",
            (
                artifact.id,
                artifact.filename,
                artifact.content_type,
                artifact.source_type,
                artifact.size_bytes,
                artifact.chunk_count,
                artifact.status,
                artifact.created_at,
                artifact.updated_at,
                artifact.error_message,
                artifact.source_id,
                artifact.source_url,
                artifact.artifact_type,
                artifact.language,
                encode_project_ids(artifact.project_ids),
                artifact.state,
                artifact.has_assignee,
                json.dumps(artifact.labels) if artifact.labels else None,
            ),
        )
