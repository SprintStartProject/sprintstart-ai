from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Literal, cast

from rag.filters import decode_project_ids, encode_project_ids

IngestionStatus = Literal["processing", "completed", "failed", "deindexed"]

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
                    project_ids TEXT
                )
                """
            )

            self._migrate_columns()

            self._connection.execute("DROP TABLE IF EXISTS artifact_chunks")
            self._connection.commit()

    def _migrate_columns(self) -> None:
        """Add columns introduced after a database was first created.

        ``project_ids`` was added with the project-separation work; databases
        created before it exist in the wild, and SQLite has no
        ``ADD COLUMN IF NOT EXISTS``.
        """
        cursor = self._connection.execute("PRAGMA table_info(artifacts)")
        existing = {str(row["name"]) for row in cursor.fetchall()}
        if "project_ids" not in existing:
            self._connection.execute(
                "ALTER TABLE artifacts ADD COLUMN project_ids TEXT"
            )

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
    ) -> list[ArtifactRecord]:
        """Return all artifacts, optionally filtered by status and project.

        Used by corpus-wide insights (e.g. knowledge-gap detection) that need to
        enumerate the ingestion index rather than look up a single id. Defaults
        to ``completed`` so callers see only fully-indexed material.

        ``project_id`` scopes the result to one project. Like retrieval, it is
        fail-closed: artifacts with no recorded project are excluded, so an
        insight never reports on material outside the requested project.
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

        if project_id is None:
            return records

        return [record for record in records if project_id in record.project_ids]

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
            ),
        )
