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
    """Whether ``chunk`` survives ``filters``.

    The project rule is the one worth stating. Material that belongs to **no**
    project is visible to every project, for the same reason a chunk with no
    ``connector_id`` is never excluded and a starter task with no track suits any
    role: absent scope is not the same as excluded scope. Everything ingested
    before this field existed has no project, and hiding all of it the moment a
    caller starts scoping would look like the corpus had emptied.
    """
    if filters is None:
        return True

    if filters.project_ids and chunk.project_ids:
        if not any(pid in filters.project_ids for pid in chunk.project_ids):
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
    """The part of ``filters`` the store itself can apply.

    ``project_ids`` is deliberately **not** pushed down. Chroma metadata values
    are scalars, so a chunk's several projects are stored as one delimited
    string, and there is no metadata operator that asks "contains this id" --
    equality would silently drop every chunk shared between two projects, which
    is the case the field exists for. It is applied in
    :func:`matches_retrieval_filters` instead, and
    :mod:`rag.hybrid` over-fetches when any filter is set so the post-filter has
    candidates to work with.
    """
    if filters is None:
        return None

    conditions: list[dict[str, object]] = []

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
