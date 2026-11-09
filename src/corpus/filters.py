from datetime import datetime

from corpus.types import (
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


def encoded_project_marker(project_id: str) -> str:
    """The substring an encoded membership list contains iff it holds ``project_id``.

    Lets a SQL ``LIKE`` match a single membership without re-deriving the
    encoding at the call site -- and without matching an id that merely shares
    a prefix, because both delimiters are part of the pattern.
    """
    return f"{_PROJECT_DELIMITER}{project_id}{_PROJECT_DELIMITER}"


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


def has_narrowing_filters(filters: RetrievalFilters | None) -> bool:
    """Whether ``filters`` narrows the corpus beyond the always-present scope.

    Only a caller-chosen narrowing counts. The project scope is on nearly every
    request and must never look like one: the buddy turn's canned reply below
    is guarded by it.
    """
    if filters is None:
        return False

    return bool(filters.source_systems) or (
        filters.time_from is not None or filters.time_to is not None
    )


# The reply a narrowed search gets when nothing under the narrowing matched:
# answering from no sources under a filter the reader chose themselves is the
# one thing not to do.
NO_FILTERED_RESULTS_MESSAGE = (
    "I could not find any matching sources for the selected filters, "
    "so I cannot answer this reliably."
)


def matches_retrieval_filters(
    chunk: Chunk | ScoredChunk,
    filters: RetrievalFilters | None,
) -> bool:
    """Whether ``chunk`` survives ``filters``.

    The project rule is the one worth stating, and it is deliberately
    fail-closed: material that belongs to **no** project is visible to **no**
    project. A chunk ingested before the backend sent ``projectIds`` carries no
    association, and stays unreachable until it is re-synced.
    """
    if filters is None:
        return True

    if filters.project_id is not None:
        # Fail closed: a chunk without any project association is visible to no project
        # until it is re-synced via /ingest/sync.
        if filters.project_id not in chunk.project_ids:
            return False

    if filters.project_ids is not None:
        # Fail closed for the same reason, and note that the empty set admits
        # nothing rather than everything: a caller that resolved a hire's
        # projects and got none has established they may see nothing, which is
        # not the same as not having asked.
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
