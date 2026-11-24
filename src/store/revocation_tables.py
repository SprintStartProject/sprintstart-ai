"""The tombstone and revision tables of the ingestion metadata store.

Retrieval hides an artifact the moment a delete or a membership rewrite begins, and
shows it again only when that operation confirms. The tables live in the same database
as the artifact records so every process sharing the file agrees on what is hidden, and
on the corpus revision that invalidates retrieval caches.
"""

import sqlite3
from collections.abc import Mapping, Sequence
from threading import RLock
from typing import Literal

RevocationReason = Literal["delete", "membership"]


class RevocationTables:
    """Mixed into the store that owns ``_connection`` and ``_lock``."""

    _connection: sqlite3.Connection
    _lock: RLock

    def _create_revocation_tables(self) -> None:
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
                state TEXT NOT NULL DEFAULT 'active'
                    CHECK (state IN ('active', 'failed')),
                revoked_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (artifact_id, reason, owner)
            )
            """
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
            "INSERT OR IGNORE INTO revocation_state (singleton, revision) VALUES (1, 0)"
        )

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
                        "(artifact_id, reason, owner, generation, state) "
                        "VALUES (?, ?, ?, 1, 'active')",
                        (artifact_id, reason, owner),
                    )
                    if inserted.rowcount > 0:
                        generation = 1
                        visible_change = True
                    else:
                        self._connection.execute(
                            "UPDATE artifact_revocations "
                            "SET generation = generation + 1, state = 'active' "
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

    def fail_revocation_operations(
        self,
        generations: Mapping[str, int],
        reason: RevocationReason,
        owner: str,
    ) -> bool:
        """Mark matching live operations as failed and safe for recovery.

        The generation predicate prevents an older failure from changing the
        state of a newer retry. Visibility is unchanged, so corpus and
        revocation revisions do not need to advance.
        """
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
                    "UPDATE artifact_revocations SET state = 'failed' "
                    "WHERE artifact_id = ? AND reason = ? AND owner = ? "
                    "AND generation = ? AND state = 'active'",
                    (
                        (artifact_id, reason, owner, generation)
                        for artifact_id, generation in normalized.items()
                    ),
                )
                changed = self._connection.total_changes > before
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

        return changed

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
        """Snapshot failed operation owners for conditional recovery."""
        with self._lock:
            rows = self._connection.execute(
                "SELECT owner, generation FROM artifact_revocations "
                "WHERE artifact_id = ? AND reason = ? AND state = 'failed'",
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
