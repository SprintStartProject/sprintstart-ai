"""Project-membership revocation, shared by both ingest paths."""

import logging

from store.base import VectorStore

logger = logging.getLogger(__name__)


def revoke_removed_memberships(
    store: VectorStore,
    artifact_id: str,
    project_ids: tuple[str, ...],
) -> None:
    """Delete an artifact's indexed chunks if it lost any project membership.

    Re-ingesting replaces the chunks — but only once parsing and embedding have
    succeeded, and either can fail (or the process can die) in between. Until
    then the old chunks keep their ``project:<id>`` markers, so an artifact
    moved out of a project stays retrievable from it for as long as the retry
    takes. Deleting first costs availability (the artifact is unsearchable until
    the re-ingest completes) and buys confidentiality, which is the trade this
    service makes everywhere else.

    Membership is read back from the vector store rather than from the ingestion
    metadata, because the metadata records the last *requested* state: after a
    failed re-ingest the two disagree, and the vectors are what retrieval sees.

    Purely additive changes (a new project, unchanged content) delete nothing.
    """
    indexed = store.project_ids_for_artifact(artifact_id)
    removed = indexed - set(project_ids)
    if not removed:
        return

    logger.info(
        "Artifact %s left project(s) %s — dropping its chunks before re-ingest",
        artifact_id,
        ", ".join(sorted(removed)),
    )
    store.delete(artifact_id, exclude_ids=[])
