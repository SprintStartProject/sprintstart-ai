"""GET /api/v1/ingest/status -- read-only AI index state for many artifacts."""

from typing import Annotated

from fastapi import APIRouter, Depends, Query

from api.dependencies import get_ingestion_metadata_store
from api.schemas import (
    ArtifactIndexStatus,
    ArtifactIngestStatusResponse,
    IngestStatusResponse,
    StatusArtifactId,
)
from ingestion.metadata_store import (
    ArtifactRecord,
    IngestionMetadataStore,
    IngestionStatus,
)

router = APIRouter()

# The knowledge base asks once per visible page, and a page never shows more.
MAX_STATUS_IDS = 100

_STATUS_BY_RECORDED: dict[IngestionStatus, ArtifactIndexStatus] = {
    "completed": "indexed",
    "processing": "processing",
    "failed": "failed",
    "deindexed": "deindexed",
}


def _to_item(
    artifact_id: str,
    record: ArtifactRecord | None,
) -> ArtifactIngestStatusResponse:
    if record is None:
        return ArtifactIngestStatusResponse(artifact_id=artifact_id, status="unknown")

    return ArtifactIngestStatusResponse(
        artifact_id=artifact_id,
        # A status this service does not know how to name is reported as
        # unknown rather than guessed at: the chip must never claim more
        # than the record says.
        status=_STATUS_BY_RECORDED.get(record.status, "unknown"),
        updated_at=record.updated_at,
        chunk_count=record.chunk_count,
    )


@router.get(
    "/ingest/status",
    response_model=IngestStatusResponse,
    summary="Get the AI index status of artifacts",
    description=(
        "Read-only. Returns one entry per distinct requested artifact id, in "
        "request order: indexed, processing, failed, deindexed, or unknown "
        "for an id this service holds no record of. Pass ids as a repeated "
        f"query parameter, at most {MAX_STATUS_IDS} per request."
    ),
)
def get_ingest_status(
    metadata_store: Annotated[
        IngestionMetadataStore,
        Depends(get_ingestion_metadata_store),
    ],
    artifact_ids: Annotated[
        list[StatusArtifactId],
        Query(default_factory=list, max_length=MAX_STATUS_IDS),
    ],
) -> IngestStatusResponse:
    requested = list(dict.fromkeys(artifact_ids))
    records = metadata_store.get_artifacts(requested)
    return IngestStatusResponse(
        items=[
            _to_item(artifact_id, records.get(artifact_id)) for artifact_id in requested
        ]
    )
