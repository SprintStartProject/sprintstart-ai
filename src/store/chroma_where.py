from typing import Any

from corpus.filters import project_metadata_key, timestamp_from_iso
from corpus.source_filter import SourceExclusions
from corpus.source_role import SourceRole
from corpus.types import RetrievalFilters


def where_filter_for_chroma(
    filters: RetrievalFilters | None,
    exclude_roles: frozenset[SourceRole] = frozenset(),
    exclusions: SourceExclusions = SourceExclusions(),
    revoked_artifact_ids: frozenset[str] = frozenset(),
) -> Any | None:
    """Translate every eligibility constraint into a Chroma ``where`` clause.

    Everything expressible here is applied *before* Chroma limits to
    ``n_results``, so ineligible chunks can never crowd eligible ones out of
    the result window. Constraints left out of this clause would have to be
    applied to an already-truncated list, which no amount of over-fetching
    makes correct.

    Chroma treats a document missing a metadata key as *matching* ``$ne``/
    ``$nin``, which is exactly the legacy-chunk semantics the in-Python
    predicates use: a chunk with no ``source_role``/``connector_id`` counts as
    ``primary`` and is never source-excluded.
    """
    conditions: list[dict[str, object]] = []

    if filters is not None:
        if filters.project_id is not None:
            conditions.append({project_metadata_key(filters.project_id): {"$eq": True}})

        if filters.project_ids is not None:
            # The per-project marker keys make membership in any of several
            # projects an $or of $eq, so this pushes down like every other
            # constraint rather than being left to the post-filter. Sorted so
            # the clause is stable for the same set.
            markers: list[dict[str, object]] = [
                {project_metadata_key(pid): {"$eq": True}}
                for pid in sorted(filters.project_ids)
            ]
            # An empty set admits nothing. Chroma has no "match nothing" literal,
            # so it is left to the in-Python predicate, which rejects it.
            if len(markers) == 1:
                conditions.append(markers[0])
            elif markers:
                conditions.append({"$or": markers})

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

    if revoked_artifact_ids:
        conditions.append({"artifact_id": {"$nin": sorted(revoked_artifact_ids)}})

    if exclude_roles:
        conditions.append({"source_role": {"$nin": sorted(exclude_roles)}})

    # A chunk with no connector is stored with ``connector_id: ""`` and is never
    # excluded, so the empty id must never reach a $nin list.
    excluded_connectors = sorted(c for c in exclusions.connectors if c)
    if excluded_connectors:
        conditions.append({"connector_id": {"$nin": excluded_connectors}})

    for connector_id, source_id in sorted(exclusions.sources):
        if not connector_id or not source_id:
            continue
        # NOT(connector_id == c AND connector_source_id == s), so a chunk from
        # the same connector but a different source stays eligible.
        conditions.append(
            {
                "$or": [
                    {"connector_id": {"$ne": connector_id}},
                    {"connector_source_id": {"$ne": source_id}},
                ]
            }
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {"$and": conditions}
