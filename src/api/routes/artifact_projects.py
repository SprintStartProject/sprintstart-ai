"""Project-membership maintenance for already-indexed artifacts.

Linking a source to another project, unlinking it, and deleting a project all
change *who may retrieve* an artifact, never the artifact itself. Retrieval is
fail-closed on the ``project:<id>`` markers in the chunk metadata (see
``rag.filters``), so a membership change that stops at the backend's database
leaves the corpus wrong in one of two directions: invisible where it should be
readable, or readable where it should be gone.

These routes rewrite that membership in place. No content is re-parsed and no
embedding is recomputed -- the existing vectors are carried over unchanged.
"""

import logging
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_ingestion_metadata_store, get_store
from api.schemas import (
    ArtifactProjectsRequest,
    ArtifactProjectsResponse,
    ArtifactProjectsSyncRequest,
    ArtifactProjectsSyncResponse,
    ProjectMembershipsDeletedResponse,
)
from ingestion.metadata_store import IngestionMetadataStore
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _rewrite_one(
    artifact: ArtifactProjectsRequest,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
) -> ArtifactProjectsResponse:
    project_ids = tuple(artifact.project_ids)

    try:
        chunk_count = store.set_project_ids_for_artifact(
            artifact.artifact_id,
            project_ids,
        )
        metadata_store.set_project_ids(
            artifact.artifact_id,
            project_ids,
            _utc_now(),
        )
    except Exception as exc:
        # Reported rather than raised: one unknown or broken artifact must not
        # abandon the rest of the batch half-applied, and the backend needs to
        # know which links did not take effect.
        logger.exception(
            "Failed to rewrite project membership of artifact %s",
            artifact.artifact_id,
        )
        return ArtifactProjectsResponse(
            artifact_id=artifact.artifact_id,
            chunk_count=0,
            status="failed",
            error_message=str(exc),
        )

    return ArtifactProjectsResponse(
        artifact_id=artifact.artifact_id,
        chunk_count=chunk_count,
    )


@router.post(
    "/artifacts/projects/sync",
    response_model=ArtifactProjectsSyncResponse,
    summary="Rewrite the project membership of indexed artifacts",
    description=(
        "Sets which projects each listed artifact belongs to, without "
        "re-ingesting or re-embedding it. Called by the backend when a source "
        "is linked to or unlinked from a project. Each entry carries the "
        "resulting membership, so the call is idempotent."
    ),
)
def sync_artifact_projects(
    body: ArtifactProjectsSyncRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    metadata_store: Annotated[
        IngestionMetadataStore,
        Depends(get_ingestion_metadata_store),
    ],
) -> ArtifactProjectsSyncResponse:
    results = [
        _rewrite_one(artifact, store, metadata_store) for artifact in body.artifacts
    ]

    failed = sum(1 for result in results if result.status == "failed")
    logger.info(
        "Project membership sync: %d artifact(s), %d failure(s)",
        len(results),
        failed,
    )

    return ArtifactProjectsSyncResponse(artifacts=results)


@router.delete(
    "/projects/{project_id}/memberships",
    response_model=ProjectMembershipsDeletedResponse,
    summary="Drop a deleted project from the whole corpus",
    description=(
        "Removes one project from every artifact and chunk that carries it. "
        "The artifacts themselves are kept -- they usually belong to other "
        "projects too -- they merely stop being retrievable from this one. "
        "Called by the backend when a project is deleted."
    ),
)
def delete_project_memberships(
    project_id: str,
    store: Annotated[VectorStore, Depends(get_store)],
    metadata_store: Annotated[
        IngestionMetadataStore,
        Depends(get_ingestion_metadata_store),
    ],
) -> ProjectMembershipsDeletedResponse:
    chunk_count = store.remove_project(project_id)
    artifact_count = metadata_store.remove_project(project_id, _utc_now())

    logger.info(
        "Purged project %s from %d chunk(s) across %d artifact(s)",
        project_id,
        chunk_count,
        artifact_count,
    )

    return ProjectMembershipsDeletedResponse(
        project_id=project_id,
        chunk_count=chunk_count,
        artifact_count=artifact_count,
    )
