from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from threading import RLock
from typing import Literal, cast

IngestionStatus = Literal["processing", "completed", "failed", "deindexed"]


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
    # Issue state at the tracker (e.g. "OPEN"/"CLOSED") and labels (e.g. "good
    # first issue"); both unset for non-issue artifacts. Used by starter-work
    # mining to deterministically exclude closed issues rather than relying on
    # an LLM to notice.
    state: str | None = None
    # Whether somebody at the source is already assigned to this issue, or None
    # when the connector cannot tell. Mining withholds an issue on a definite
    # True -- work somebody else has taken is not work a hire can pick up.
    # ⚠️ None is "unknown", never "nobody": GitHub issues have assignees this
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
                    state TEXT,
                    has_assignee INTEGER,
                    labels TEXT
                )
                """
            )
            # A pre-existing DB file predates the state/labels/has_assignee
            # columns; CREATE
            # TABLE IF NOT EXISTS alone won't add them to it. This SQLite build
            # has no ADD COLUMN IF NOT EXISTS guarantee, so add them
            # defensively and ignore "duplicate column".
            for column in ("state TEXT", "labels TEXT", "has_assignee INTEGER"):
                try:
                    self._connection.execute(
                        f"ALTER TABLE artifacts ADD COLUMN {column}"
                    )
                except sqlite3.OperationalError as exc:
                    if "duplicate column" not in str(exc).lower():
                        raise

            self._connection.execute("DROP TABLE IF EXISTS artifact_chunks")
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

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
                """
                SELECT
                    id,
                    filename,
                    content_type,
                    source_type,
                    size_bytes,
                    chunk_count,
                    status,
                    created_at,
                    updated_at,
                    error_message,
                    source_id,
                    source_url,
                    artifact_type,
                    language,
                    state,
                    has_assignee,
                    labels
                FROM artifacts
                WHERE id = ?
                """,
                (artifact_id,),
            )
            row = cast(sqlite3.Row | None, cursor.fetchone())

        if row is None:
            return None

        return self._row_to_record(row)

    def list_artifacts(
        self, status: IngestionStatus | None = "completed"
    ) -> list[ArtifactRecord]:
        """Return all artifacts, optionally filtered by status.

        Used by corpus-wide insights (e.g. knowledge-gap detection) that need to
        enumerate the whole ingestion index rather than look up a single id.
        Defaults to ``completed`` so callers see only fully-indexed material.
        """
        query = (
            "SELECT id, filename, content_type, source_type, size_bytes, "
            "chunk_count, status, created_at, updated_at, error_message, "
            "source_id, source_url, artifact_type, language, state, "
            "has_assignee, labels "
            "FROM artifacts"
        )
        params: tuple[str, ...] = ()
        if status is not None:
            query += " WHERE status = ?"
            params = (status,)

        with self._lock:
            cursor = self._connection.execute(query, params)
            rows = cursor.fetchall()

        return [self._row_to_record(cast(sqlite3.Row, row)) for row in rows]

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
            state=None if row["state"] is None else str(row["state"]),
            has_assignee=(
                None if row["has_assignee"] is None else bool(row["has_assignee"])
            ),
            labels=json.loads(row["labels"]) if row["labels"] else [],
        )

    def _upsert_artifact(self, artifact: ArtifactRecord) -> None:
        self._connection.execute(
            """
            INSERT OR REPLACE INTO artifacts (
                id,
                filename,
                content_type,
                source_type,
                size_bytes,
                chunk_count,
                status,
                created_at,
                updated_at,
                error_message,
                source_id,
                source_url,
                artifact_type,
                language,
                state,
                has_assignee,
                labels
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
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
                artifact.state,
                artifact.has_assignee,
                json.dumps(artifact.labels) if artifact.labels else None,
            ),
        )
