"""Chroma records and the corpus types they stand for.

Chroma hands back parallel ``ids`` / ``documents`` / ``metadatas`` lists; these
functions turn them into :class:`Chunk` and :class:`ScoredChunk` and compute the
fingerprint a cached BM25 index keys on. They do no I/O.
"""

from collections.abc import Mapping
from typing import Any, cast

from corpus.filters import (
    PROJECT_IDS_METADATA_KEY,
    decode_project_ids,
    encode_project_ids,
    normalize_source_system,
    project_metadata_key,
    timestamp_from_iso,
)
from corpus.ids import ChunkId
from corpus.source_role import SourceRole
from corpus.types import Chunk, ChunkKind, ScoredChunk, is_chunk_kind

# Keeps every Chroma ``get`` comfortably below SQLite's 32,766 bind-variable
# ceiling. A page includes ids, documents, and metadata, so using the backend
# ceiling itself would still be too large once Chroma hydrates those records.
MAX_GET_PAGE: int = 10_000

# Stands in for ``Chunk.position is None``: Chroma metadata cannot store ``None``.
NO_POSITION: int = -1


def _position_from_metadata(metadata: Mapping[str, object]) -> int | None:
    raw_position = metadata.get("position")
    if not isinstance(raw_position, (int, float)) or raw_position == NO_POSITION:
        return None
    return int(raw_position)


def _kind_from_metadata(metadata: Mapping[str, object]) -> ChunkKind:
    kind_str = str(metadata.get("kind", "text"))
    if not is_chunk_kind(kind_str):
        raise ValueError(f"Unknown chunk kind {kind_str!r}")
    return kind_str


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


def retrieval_fingerprint(chunk: ChunkId, metadata: Mapping[str, object]) -> str:
    parts: list[str] = [chunk]
    parts.extend(str(metadata.get(key, "")) for key in _FILTER_METADATA_KEYS)
    return "\x00".join(parts)


def chunks_from_get_result(raw_result: Any) -> list[Chunk]:
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
        position = _position_from_metadata(metadata)

        kind_str = _kind_from_metadata(metadata)

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


def chunks_without_embeddings_from_get_result(raw_result: Any) -> list[Chunk]:
    ids = cast(list[str], raw_result["ids"])
    documents = cast(list[str], raw_result.get("documents") or [])
    metadatas = cast(list[Mapping[str, object]], raw_result.get("metadatas") or [])

    chunks: list[Chunk] = []
    for chunk_id, text, metadata in zip(ids, documents, metadatas, strict=True):
        position = _position_from_metadata(metadata)

        kind_str = _kind_from_metadata(metadata)

        chunks.append(
            Chunk(
                id=str(chunk_id),
                artifact_id=str(metadata["artifact_id"]),
                filename=str(metadata["filename"]),
                position=position,
                kind=kind_str,
                text=str(text),
                embedding=[],
                source_role=_source_role_from_metadata(metadata),
                source_url=_optional_str(metadata.get("source_url")),
                artifact_type=_optional_str(metadata.get("artifact_type")),
                language=_optional_str(metadata.get("language")),
                connector_id=_optional_str(metadata.get("connector_id")),
                connector_source_id=_optional_str(metadata.get("connector_source_id")),
                source_system=normalize_source_system(
                    _optional_str(metadata.get("source_system"))
                ),
                created_at=_optional_str(metadata.get("created_at")),
                start_line=_optional_int(metadata.get("start_line")),
                start_page=_optional_int(metadata.get("start_page")),
                project_ids=decode_project_ids(metadata.get(PROJECT_IDS_METADATA_KEY)),
            )
        )

    return chunks


def scored_chunk(
    chunk_id: ChunkId, text: str, metadata: Mapping[str, object], score: float
) -> ScoredChunk:
    position = _position_from_metadata(metadata)

    kind_str = _kind_from_metadata(metadata)

    return ScoredChunk(
        id=chunk_id,
        artifact_id=str(metadata["artifact_id"]),
        filename=str(metadata["filename"]),
        position=position,
        kind=kind_str,
        text=text,
        score=score,
        source_role=_source_role_from_metadata(metadata),
        source_url=_optional_str(metadata.get("source_url")),
        artifact_type=_optional_str(metadata.get("artifact_type")),
        language=_optional_str(metadata.get("language")),
        connector_id=_optional_str(metadata.get("connector_id")),
        connector_source_id=_optional_str(metadata.get("connector_source_id")),
        source_system=normalize_source_system(
            _optional_str(metadata.get("source_system"))
        ),
        created_at=_optional_str(metadata.get("created_at")),
        start_line=_optional_int(metadata.get("start_line")),
        start_page=_optional_int(metadata.get("start_page")),
        project_ids=decode_project_ids(metadata.get(PROJECT_IDS_METADATA_KEY)),
    )


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
    back as missing rather than via a sentinel like ``NO_POSITION``.
    """
    return int(value) if isinstance(value, (int, float)) else None


def visible_chunks(
    chunks: list[Chunk],
    revoked_artifact_ids: frozenset[str],
) -> list[Chunk]:
    return [chunk for chunk in chunks if chunk.artifact_id not in revoked_artifact_ids]


def chunk_metadata(chunk: Chunk) -> dict[str, str | int | float | bool]:
    """Metadata stored beside a chunk; ``chunks_from_get_result`` reads it back."""
    metadata: dict[str, str | int | float | bool] = {
        "artifact_id": chunk.artifact_id,
        "filename": chunk.filename,
        "position": chunk.position if chunk.position is not None else NO_POSITION,
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
    return metadata
