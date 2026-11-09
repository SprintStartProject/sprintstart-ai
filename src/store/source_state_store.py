from __future__ import annotations

import sqlite3
from pathlib import Path
from threading import RLock

from corpus.source_filter import SourceExclusions


class SourceStateStore:
    """Tracks connector/source enable-disable state.

    Only the *disabled* set is stored — a connector or source with no row is
    enabled by default, matching the backend's default state.
    """

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
                CREATE TABLE IF NOT EXISTS disabled_connectors (
                    connector_id TEXT PRIMARY KEY
                )
                """
            )
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS disabled_sources (
                    connector_id TEXT NOT NULL,
                    source_id TEXT NOT NULL,
                    PRIMARY KEY (connector_id, source_id)
                )
                """
            )
            self._connection.commit()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def set_connector_enabled(self, connector_id: str, enabled: bool) -> None:
        with self._lock:
            if enabled:
                self._connection.execute(
                    "DELETE FROM disabled_connectors WHERE connector_id = ?",
                    (connector_id,),
                )
            else:
                self._connection.execute(
                    "INSERT OR REPLACE INTO disabled_connectors (connector_id) "
                    "VALUES (?)",
                    (connector_id,),
                )
            self._connection.commit()

    def set_sources_enabled(self, connector_id: str, statuses: dict[str, bool]) -> None:
        with self._lock:
            for source_id, enabled in statuses.items():
                if enabled:
                    self._connection.execute(
                        "DELETE FROM disabled_sources "
                        "WHERE connector_id = ? AND source_id = ?",
                        (connector_id, source_id),
                    )
                else:
                    self._connection.execute(
                        "INSERT OR REPLACE INTO disabled_sources "
                        "(connector_id, source_id) VALUES (?, ?)",
                        (connector_id, source_id),
                    )
            self._connection.commit()

    def get_exclusions(self) -> SourceExclusions:
        with self._lock:
            connector_rows = self._connection.execute(
                "SELECT connector_id FROM disabled_connectors"
            ).fetchall()
            source_rows = self._connection.execute(
                "SELECT connector_id, source_id FROM disabled_sources"
            ).fetchall()

        return SourceExclusions(
            connectors=frozenset(str(row["connector_id"]) for row in connector_rows),
            sources=frozenset(
                (str(row["connector_id"]), str(row["source_id"])) for row in source_rows
            ),
        )
