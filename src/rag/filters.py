from datetime import datetime
from typing import Any

from rag.types import (
    Chunk,
    RetrievalFilters,
    ScoredChunk,
    SourceSystem,
    is_source_system,
)


def normalize_source_system(value: str | None) -> SourceSystem | None:
    if value is None:
        return None

    normalized = value.upper()

    if is_source_system(normalized):
        return normalized

    return None


_PROJECT_DELIMITER = "|"
# Prefix of the per-project marker keys written into the chunk metadata. Chroma
# metadata values cannot be lists, so membership is encoded as one boolean key
# per project (``project:<uuid> = True``), which ``$eq`` can filter on
# server-side. ``project_ids`` (a delimited string) carries the same
# information in a form that can be read back into a ``Chunk``.
PROJECT_METADATA_PREFIX = "project:"
PROJECT_IDS_METADATA_KEY = "project_ids"


def project_metadata_key(project_id: str) -> str:
    return f"{PROJECT_METADATA_PREFIX}{project_id}"


def encode_project_ids(project_ids: tuple[str, ...] | list[str]) -> str:
    """Encode project ids as a delimited string, e.g. ``|uuid1|uuid2|``.

    Empty input encodes to the empty string so "no projects" never looks like a
    membership marker.
    """
    unique = list(dict.fromkeys(pid for pid in project_ids if pid))
    if not unique:
        return ""
    return _PROJECT_DELIMITER + _PROJECT_DELIMITER.join(unique) + _PROJECT_DELIMITER


def decode_project_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, str) or not value:
        return ()
    return tuple(part for part in value.split(_PROJECT_DELIMITER) if part)


def timestamp_from_iso(value: str | None) -> float:
    parsed = _parse_timestamp(value)
    return parsed or 0.0


def _parse_timestamp(value: str | None) -> float | None:
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def matches_retrieval_filters(
    chunk: Chunk | ScoredChunk,
    filters: RetrievalFilters | None,
) -> bool:
    if filters is None:
        return True

    if filters.project_id is not None:
        # Fail closed: a chunk that carries no project association at all (i.e.
        # ingested before the backend sent projectIds) is not visible to any
        # project. Re-sync via /ingest/sync to make it retrievable again.
        if filters.project_id not in chunk.project_ids:
            return False

    if filters.source_systems:
        if chunk.source_system not in filters.source_systems:
            return False

    has_time_filter = filters.time_from is not None or filters.time_to is not None

    if has_time_filter:
        chunk_timestamp = _parse_timestamp(chunk.created_at)

        if chunk_timestamp is None:
            return False

        if filters.time_from is not None:
            time_from = _parse_timestamp(filters.time_from)
            if time_from is not None and chunk_timestamp < time_from:
                return False

        if filters.time_to is not None:
            time_to = _parse_timestamp(filters.time_to)
            if time_to is not None and chunk_timestamp > time_to:
                return False

    return True


def where_filter_for_chroma(filters: RetrievalFilters | None) -> Any | None:
    if filters is None:
        return None

    conditions: list[dict[str, object]] = []

    if filters.project_id is not None:
        conditions.append({project_metadata_key(filters.project_id): {"$eq": True}})

    if filters.source_systems:
        conditions.append({"source_system": {"$in": filters.source_systems}})

    has_time_filter = filters.time_from is not None or filters.time_to is not None

    if has_time_filter:
        conditions.append({"created_at_ts": {"$gt": 0.0}})

    if filters.time_from is not None:
        conditions.append(
            {"created_at_ts": {"$gte": timestamp_from_iso(filters.time_from)}}
        )

    if filters.time_to is not None:
        conditions.append(
            {"created_at_ts": {"$lte": timestamp_from_iso(filters.time_to)}}
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}
