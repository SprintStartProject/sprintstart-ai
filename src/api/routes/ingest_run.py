"""POST /api/v1/ingest/sync — batch ingest for completed GitHub ingestion runs."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    ArtifactDeindexResponse,
    ArtifactRunIngestRequest,
    ArtifactRunIngestResponse,
    RunArtifactsSyncRequest,
    RunArtifactsSyncResponse,
)
from corpus.ids import ArtifactId, SourceId
from ingestion.pipeline.steps import IngestDeps
from ingestion.pipeline.sync import (
    DeindexOutcome,
    SyncedArtifact,
    SyncedResult,
    deindex,
    ingest_all,
)
from llm.base import LLMClient
from store.base import VectorStore
from store.metadata_store import IngestionMetadataStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _synced_artifact(request: ArtifactRunIngestRequest) -> SyncedArtifact:
    return SyncedArtifact(
        artifact_id=ArtifactId(request.artifact_id),
        source_id=SourceId(request.source_id),
        artifact_type=request.artifact_type,
        project_ids=tuple(request.project_ids),
        source_system=request.source_system,
        source_url=request.source_url,
        title=request.title,
        body_text=request.body_text,
        mime=request.mime,
        language=request.language,
        source_created_at=request.source_created_at,
        source_updated_at=request.source_updated_at,
        state=request.state,
        has_assignee=request.has_assignee,
        labels=tuple(request.labels),
    )


def _ingest_response(result: SyncedResult) -> ArtifactRunIngestResponse:
    return ArtifactRunIngestResponse(
        artifact_id=result.artifact_id,
        chunk_count=result.chunk_count,
        status=result.status,
    )


def _deindex_response(outcome: DeindexOutcome) -> ArtifactDeindexResponse:
    if outcome.error is None:
        return ArtifactDeindexResponse(artifact_id=outcome.artifact_id)
    return ArtifactDeindexResponse(
        artifact_id=outcome.artifact_id,
        status="failed",
        error_message=outcome.error,
    )


@router.post(
    "/ingest/sync",
    response_model=RunArtifactsSyncResponse,
    summary="Batch ingest a completed GitHub ingestion run",
    description=(
        "Accepts a list of artifacts to index and a list of artifact IDs to remove "
        "from the vector store. Called by the backend after a GitHub ingestion run "
        "completes (COMPLETED or PARTIAL status). Deindexing runs first so a "
        "re-ingested artifact always reflects the latest content."
    ),
)
def ingest_run(
    body: RunArtifactsSyncRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    store: Annotated[VectorStore, Depends(get_store)],
    metadata_store: Annotated[
        IngestionMetadataStore,
        Depends(get_ingestion_metadata_store),
    ],
) -> RunArtifactsSyncResponse:
    logger.info(
        "Sync request received: %d to ingest, %d to deindex",
        len(body.artifacts_to_ingest),
        len(body.artifacts_to_deindex),
    )
    deps = IngestDeps(llm, store, metadata_store)

    deindexed = [
        _deindex_response(outcome)
        for outcome in deindex([ArtifactId(i) for i in body.artifacts_to_deindex], deps)
    ]
    results = [
        _ingest_response(result)
        for result in ingest_all(
            [_synced_artifact(a) for a in body.artifacts_to_ingest], deps
        )
    ]

    failed_deindexes = sum(1 for entry in deindexed if entry.status == "failed")
    logger.info(
        "Sync complete: %d ingested, %d deindexed (%d deindex failures)",
        len(results),
        len(deindexed) - failed_deindexes,
        failed_deindexes,
    )

    return RunArtifactsSyncResponse(artifacts=results, deindexed=deindexed)
