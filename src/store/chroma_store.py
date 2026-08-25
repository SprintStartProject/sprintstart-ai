import threading
from collections.abc import Iterator, Mapping
from typing import Any, cast

import chromadb
import chromadb.api
from chromadb.api.types import Metadata, PyEmbeddings, Where
from chromadb.config import Settings

from ingestion.source_role import SourceRole
from rag.filters import (
    PROJECT_IDS_METADATA_KEY,
    decode_project_ids,
    encode_project_ids,
    normalize_source_system,
    project_metadata_key,
    timestamp_from_iso,
    where_filter_for_chroma,
)
from rag.source_filter import SourceExclusions
from rag.types import Chunk, RetrievalFilters, ScoredChunk, is_chunk_kind

_NO_POSITION: int = -1
# Keep every Chroma ``get`` comfortably below SQLite's 32,766 bind-variable
# ceiling. A page includes ids, documents, and metadata, so using the backend
# ceiling itself would still be too large once Chroma hydrates those records.
_MAX_GET_PAGE: int = 10_000

_CLIENT_CACHE: dict[str, chromadb.api.ClientAPI] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def _get_persistent_client(path: str) -> chromadb.api.ClientAPI:
    client = _CLIENT_CACHE.get(path)
    if client is not None:
        return client

    # FastAPI resolves synchronous dependencies in a thread pool, so the first
    # chat and its asynchronous analytics fan-out can arrive here together.
    # Chroma's PersistentClient instances share process-wide, refcounted state;
    # constructing two for one path concurrently can make one failed init tear
    # down the other's system. Re-check inside the lock so exactly one thread
    # constructs and publishes the client.
    with _CLIENT_CACHE_LOCK:
        client = _CLIENT_CACHE.get(path)
        if client is not None:
            return client

        settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            allow_reset=True,
        )
        client = chromadb.PersistentClient(path=path, settings=settings)
        _CLIENT_CACHE[path] = client
        return client


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
            self._client = _get_persistent_client(path)
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

        metadatas: list[dict[str, str | int | float | bool]] = []
        for chunk in chunks:
            metadata: dict[str, str | int | float | bool] = {
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
                PROJECT_IDS_METADATA_KEY: encode_project_ids(chunk.project_ids),
            }
            # One boolean marker per project so Chroma can filter membership
            # server-side ($eq on a key documents of other projects don't have);
            # metadata values themselves cannot be lists.
            for project_id in chunk.project_ids:
                metadata[project_metadata_key(project_id)] = True
            if chunk.start_line is not None:
                metadata["start_line"] = chunk.start_line
            if chunk.start_page is not None:
                metadata["start_page"] = chunk.start_page
            metadatas.append(metadata)

        ids = [chunk.id for chunk in chunks]

        # Chroma's upsert *merges* metadata instead of replacing it, so a key
        # that is no longer written would survive a re-ingest — including the
        # ``project:<id>`` marker of a project the artifact was removed from,
        # which would keep it retrievable from that project forever. Deleting
        # first makes each chunk's metadata exactly what we write here.
        self._collection.delete(ids=ids)

        self._collection.upsert(
            ids=ids,
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
        exclude_roles: frozenset[SourceRole] = frozenset(),
        exclusions: SourceExclusions = SourceExclusions(),
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0 or top_k <= 0:
            return []

        # Every constraint goes into the ``where`` clause, which Chroma applies
        # before limiting to ``n_results``. Filtering the returned window
        # instead would let higher-ranked ineligible chunks push every eligible
        # one out of it.
        where_filter = where_filter_for_chroma(filters, exclude_roles, exclusions)

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
                    project_ids=decode_project_ids(
                        metadata.get(PROJECT_IDS_METADATA_KEY)
                    ),
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

    def all_chunks_without_embeddings(self) -> list[Chunk]:
        return list(self.iter_chunks_without_embeddings())

    def iter_chunks_without_embeddings(self) -> Iterator[Chunk]:
        """Yield the corpus through bounded, embedding-free Chroma reads."""
        offset = 0

        while True:
            page = self.list_chunks_without_embeddings(
                limit=_MAX_GET_PAGE,
                offset=offset,
            )
            if not page:
                return

            yield from page
            offset += len(page)

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
                    project_ids=decode_project_ids(
                        metadata.get(PROJECT_IDS_METADATA_KEY)
                    ),
                )
            )

        return chunks

    def all_ids(self) -> frozenset[str]:
        raw_result = self._collection.get(include=[])
        return frozenset(str(chunk_id) for chunk_id in raw_result["ids"])

    def retrieval_fingerprints(self) -> frozenset[str]:
        return frozenset(
            _retrieval_fingerprint(chunk_id, metadata)
            for chunk_id, metadata in self._iter_metadata_records()
        )

    def project_ids_for_artifact(self, artifact_id: str) -> frozenset[str]:
        return frozenset(
            project_id
            for _, metadata in self._iter_metadata_records(
                where={"artifact_id": artifact_id}
            )
            for project_id in decode_project_ids(metadata.get(PROJECT_IDS_METADATA_KEY))
        )

    def _iter_metadata_records(
        self,
        where: Where | None = None,
    ) -> Iterator[tuple[str, Mapping[str, object]]]:
        """Yield ids and metadata through bounded reads without hydrating text."""
        offset = 0

        while True:
            raw_result = self._collection.get(
                where=where,
                include=["metadatas"],
                limit=_MAX_GET_PAGE,
                offset=offset,
            )
            ids = raw_result["ids"]
            if not ids:
                return

            metadatas = cast(
                list[Mapping[str, object]],
                raw_result.get("metadatas") or [],
            )
            yield from zip(ids, metadatas, strict=True)
            offset += len(ids)

    def count(self) -> int:
        return self._collection.count()


# Metadata a cached BM25 index filters on. Chunk ids are content-hashed, so an
# id already stands for the chunk's text — but not for any of these, which the
# backend can change without touching the content (moving an artifact between
# projects, reclassifying it as test material, disabling its connector). They
# belong in the fingerprint or a stale index keeps filtering on the old values.
_FILTER_METADATA_KEYS = (
    PROJECT_IDS_METADATA_KEY,
    "source_role",
    "connector_id",
    "connector_source_id",
    "source_system",
    "created_at",
)


def _retrieval_fingerprint(chunk_id: str, metadata: Mapping[str, object]) -> str:
    parts = [chunk_id]
    parts.extend(str(metadata.get(key, "")) for key in _FILTER_METADATA_KEYS)
    return "\x00".join(parts)


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
                project_ids=decode_project_ids(metadata.get(PROJECT_IDS_METADATA_KEY)),
            )
        )

    return chunks


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
