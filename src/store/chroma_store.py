from collections.abc import Mapping
from typing import Any, cast

import chromadb
import chromadb.api
from chromadb.api.types import Metadata, PyEmbeddings

from ingestion.source_role import SourceRole
from rag.filters import (
    normalize_source_system,
    timestamp_from_iso,
    where_filter_for_chroma,
)
from rag.types import Chunk, RetrievalFilters, ScoredChunk, is_chunk_kind

_NO_POSITION: int = -1

#: Separates project ids inside the single scalar Chroma will store for them.
_PROJECT_DELIMITER = "|"


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "chunks",
        client: chromadb.api.ClientAPI | None = None,
        path: str | None = None,
    ) -> None:
        if client is not None:
            self._client: chromadb.api.ClientAPI = client
        elif path is not None:
            self._client = chromadb.PersistentClient(path=path)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embeddings: list[list[float]] = [chunk.embedding for chunk in chunks]

        metadatas: list[dict[str, str | int | float]] = []
        for chunk in chunks:
            metadata: dict[str, str | int | float] = {
                "artifact_id": chunk.artifact_id,
                "filename": chunk.filename,
                "position": (
                    chunk.position if chunk.position is not None else _NO_POSITION
                ),
                "kind": chunk.kind,
                "source_role": chunk.source_role,
                "source_url": chunk.source_url or "",
                "artifact_type": chunk.artifact_type or "",
                "language": chunk.language or "",
                "connector_id": chunk.connector_id or "",
                "connector_source_id": chunk.connector_source_id or "",
                "source_system": chunk.source_system or "",
                "created_at": chunk.created_at or "",
                "created_at_ts": timestamp_from_iso(chunk.created_at),
                # Chroma metadata holds scalars, so several projects become one
                # delimited string. Delimited on both ends so a naive substring
                # check can never match a partial id.
                "project_ids": _encode_projects(chunk.project_ids),
            }
            if chunk.start_line is not None:
                metadata["start_line"] = chunk.start_line
            if chunk.start_page is not None:
                metadata["start_page"] = chunk.start_page
            metadatas.append(metadata)

        self._collection.upsert(
            ids=[chunk.id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=cast(PyEmbeddings, embeddings),
            metadatas=cast(list[Metadata], metadatas),
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
        filters: RetrievalFilters | None = None,
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0:
            return []

        where_filter = where_filter_for_chroma(filters)

        raw_result = self._collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        ids = raw_result["ids"][0]
        documents = (raw_result["documents"] or [[]])[0]
        metadatas = cast(
            list[Mapping[str, object]],
            (raw_result["metadatas"] or [[]])[0],
        )
        distances = (raw_result["distances"] or [[]])[0]

        results: list[ScoredChunk] = []

        for chunk_id, text, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=True,
        ):
            score = 1.0 - distance

            if score < min_score:
                continue

            raw_position = metadata.get("position")
            position = (
                None
                if not isinstance(raw_position, (int, float))
                or raw_position == _NO_POSITION
                else int(raw_position)
            )

            kind_str = str(metadata.get("kind", "text"))
            if not is_chunk_kind(kind_str):
                raise ValueError(f"Unknown chunk kind {kind_str!r}")

            source_system = normalize_source_system(
                _optional_str(metadata.get("source_system"))
            )

            results.append(
                ScoredChunk(
                    id=str(chunk_id),
                    artifact_id=str(metadata["artifact_id"]),
                    filename=str(metadata["filename"]),
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    score=score,
                    source_role=_source_role_from_metadata(metadata),
                    source_url=_optional_str(metadata.get("source_url")),
                    artifact_type=_optional_str(metadata.get("artifact_type")),
                    language=_optional_str(metadata.get("language")),
                    connector_id=_optional_str(metadata.get("connector_id")),
                    connector_source_id=_optional_str(
                        metadata.get("connector_source_id")
                    ),
                    source_system=source_system,
                    created_at=_optional_str(metadata.get("created_at")),
                    start_line=_optional_int(metadata.get("start_line")),
                    start_page=_optional_int(metadata.get("start_page")),
                    project_ids=_decode_projects(metadata.get("project_ids")),
                )
            )

        results.sort(key=lambda c: c.score, reverse=True)
        return results

    def delete(self, artifact_id: str, exclude_ids: list[str] | None = None) -> int:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )

        ids = list(raw_result["ids"])

        if exclude_ids:
            ids = [i for i in ids if i not in exclude_ids]

        if ids:
            self._collection.delete(ids=ids)

        return len(ids)

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]:
        raw_result = self._collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        return _chunks_from_get_result(raw_result)

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        return _chunks_from_get_result(raw_result)

    def count_by_artifact(self, artifact_id: str) -> int:
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )
        return len(raw_result["ids"])

    def all_chunks(self) -> list[Chunk]:
        raw_result = self._collection.get(
            include=["documents", "metadatas", "embeddings"],
        )
        return _chunks_from_get_result(raw_result)

    def all_chunks_without_embeddings(self) -> list[Chunk]:
        total = self.count()
        if total == 0:
            return []
        return self.list_chunks_without_embeddings(limit=total, offset=0)

    def list_chunks_without_embeddings(
        self, limit: int, offset: int = 0
    ) -> list[Chunk]:
        raw_result = self._collection.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )

        ids = raw_result["ids"]
        documents = raw_result["documents"] or []
        metadatas = raw_result["metadatas"] or []

        chunks: list[Chunk] = []

        for chunk_id, text, metadata in zip(
            ids,
            documents,
            metadatas,
            strict=True,
        ):
            raw_position = metadata.get("position")
            position = (
                None
                if not isinstance(raw_position, (int, float))
                or raw_position == _NO_POSITION
                else int(raw_position)
            )

            kind_str = str(metadata.get("kind", "text"))
            if not is_chunk_kind(kind_str):
                raise ValueError(f"Unknown chunk kind {kind_str!r}")

            source_system = normalize_source_system(
                _optional_str(metadata.get("source_system"))
            )

            chunks.append(
                Chunk(
                    id=str(chunk_id),
                    artifact_id=str(metadata["artifact_id"]),
                    filename=str(metadata["filename"]),
                    position=position,
                    kind=kind_str,
                    text=str(text),
                    embedding=[],  # no embeddings in text-only fetch
                    source_role=_source_role_from_metadata(metadata),
                    source_url=_optional_str(metadata.get("source_url")),
                    artifact_type=_optional_str(metadata.get("artifact_type")),
                    language=_optional_str(metadata.get("language")),
                    connector_id=_optional_str(metadata.get("connector_id")),
                    connector_source_id=_optional_str(
                        metadata.get("connector_source_id")
                    ),
                    source_system=source_system,
                    created_at=_optional_str(metadata.get("created_at")),
                    start_line=_optional_int(metadata.get("start_line")),
                    start_page=_optional_int(metadata.get("start_page")),
                    project_ids=_decode_projects(metadata.get("project_ids")),
                )
            )

        return chunks

    def all_ids(self) -> frozenset[str]:
        raw_result = self._collection.get(include=[])
        return frozenset(str(chunk_id) for chunk_id in raw_result["ids"])

    def count(self) -> int:
        return self._collection.count()


def _chunks_from_get_result(raw_result: Any) -> list[Chunk]:
    ids = cast(list[str], raw_result["ids"])
    documents = cast(list[str], raw_result.get("documents") or [])
    metadatas = cast(list[Mapping[str, object]], raw_result.get("metadatas") or [])
    raw_embeddings = raw_result.get("embeddings")
    if raw_embeddings is None:
        embeddings: list[list[float]] = []
    elif hasattr(raw_embeddings, "tolist"):
        embeddings = cast(list[list[float]], raw_embeddings.tolist())
    else:
        embeddings = cast(list[list[float]], raw_embeddings)

    chunks: list[Chunk] = []

    for chunk_id, text, metadata, embedding in zip(
        ids,
        documents,
        metadatas,
        embeddings,
        strict=True,
    ):
        raw_position = metadata.get("position")
        position = (
            None
            if not isinstance(raw_position, (int, float))
            or raw_position == _NO_POSITION
            else int(raw_position)
        )

        kind_str = str(metadata.get("kind", "text"))
        if not is_chunk_kind(kind_str):
            raise ValueError(f"Unknown chunk kind {kind_str!r}")

        source_system = normalize_source_system(
            _optional_str(metadata.get("source_system"))
        )

        chunks.append(
            Chunk(
                id=str(chunk_id),
                artifact_id=str(metadata["artifact_id"]),
                filename=str(metadata["filename"]),
                position=position,
                kind=kind_str,
                text=str(text),
                embedding=[float(value) for value in embedding],
                source_role=_source_role_from_metadata(metadata),
                source_url=_optional_str(metadata.get("source_url")),
                artifact_type=_optional_str(metadata.get("artifact_type")),
                language=_optional_str(metadata.get("language")),
                connector_id=_optional_str(metadata.get("connector_id")),
                connector_source_id=_optional_str(metadata.get("connector_source_id")),
                source_system=source_system,
                created_at=_optional_str(metadata.get("created_at")),
                start_line=_optional_int(metadata.get("start_line")),
                start_page=_optional_int(metadata.get("start_page")),
                project_ids=_decode_projects(metadata.get("project_ids")),
            )
        )

    return chunks


def _encode_projects(project_ids: tuple[str, ...]) -> str:
    """Several project ids as one Chroma-storable scalar.

    Delimited on both ends (``|a|b|``) so that a substring check can never match
    half an id. Empty stays empty, which is what "belongs to no project" reads
    back as.
    """
    if not project_ids:
        return ""
    joined = _PROJECT_DELIMITER.join(project_ids)
    return f"{_PROJECT_DELIMITER}{joined}{_PROJECT_DELIMITER}"


def _decode_projects(value: object) -> tuple[str, ...]:
    """The inverse of :func:`_encode_projects`, tolerant of chunks written before it."""
    if not isinstance(value, str) or not value:
        return ()
    return tuple(part for part in value.split(_PROJECT_DELIMITER) if part)


def _source_role_from_metadata(metadata: Mapping[str, object]) -> SourceRole:
    raw_source_role = metadata.get("source_role")
    if raw_source_role in {"primary", "test"}:
        return cast(SourceRole, raw_source_role)

    return "primary"


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None

    return str(value)


def _optional_int(value: object) -> int | None:
    """Return the metadata value as an int, or None.

    Chroma metadata cannot store ``None`` directly, so absent optional ints
    (e.g. ``start_line``/``start_page``) are simply never written and read
    back as missing rather than via a sentinel like ``_NO_POSITION``.
    """
    return int(value) if isinstance(value, (int, float)) else None
