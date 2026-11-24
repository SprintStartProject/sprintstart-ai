from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    IngestArtifactResponse,
    IngestChunkResponse,
    IngestRequest,
    IngestResponse,
    ValidationErrorResponse,
)
from corpus.ids import ArtifactId
from corpus.types import Chunk
from ingestion.pipeline.steps import IngestDeps, IngestError, IngestFailure
from ingestion.pipeline.upload import UploadedDocument, ingest_upload
from llm.base import LLMClient
from store.base import VectorStore
from store.metadata_store import ArtifactRecord, IngestionMetadataStore

router = APIRouter()

_STATUS_FOR_FAILURE: dict[IngestFailure, int] = {
    "invalid_content": 422,
    "too_large": 413,
    "llm_unavailable": 503,
    "store_failed": 500,
}


def _artifact_to_response(artifact: ArtifactRecord) -> IngestArtifactResponse:
    return IngestArtifactResponse(
        id=artifact.id,
        filename=artifact.filename,
        content_type=artifact.content_type,
        source_type=artifact.source_type,
        size_bytes=artifact.size_bytes,
        chunk_count=artifact.chunk_count,
        status=artifact.status,
        created_at=artifact.created_at,
        updated_at=artifact.updated_at,
        error_message=artifact.error_message,
    )


def _chunk_to_response(chunk: Chunk, chunk_index: int) -> IngestChunkResponse:
    return IngestChunkResponse(
        id=chunk.id,
        artifact_id=chunk.artifact_id,
        filename=chunk.filename,
        text=chunk.text,
        chunk_index=chunk_index,
        vector_store_id=chunk.id,
        kind=chunk.kind,
    )


def _response_from_records(
    artifact: ArtifactRecord,
    chunks: list[Chunk],
) -> IngestResponse:
    return IngestResponse(
        artifact_id=artifact.id,
        chunk_count=artifact.chunk_count,
        artifact=_artifact_to_response(artifact),
        chunks=[
            _chunk_to_response(chunk, chunk_index=index)
            for index, chunk in enumerate(chunks)
        ],
    )


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a document",
    description=(
        "Parses, chunks, and embeds a document, then stores it in the vector store. "
        "Re-ingesting the same artifact_id replaces the existing chunks. "
        "Supported types: plain text, Markdown, JSON, YAML, TOML, and images "
        "(.png, .jpg, .jpeg, .gif, .webp, .bmp — send as base64-encoded content). "
        "Image files are captioned via the configured vision model; if no vision model "
        "is available the request succeeds with chunk_count=0."
    ),
    responses={
        422: {
            "model": ValidationErrorResponse,
            "content": {
                "application/json": {"example": {"detail": "'question' is required"}}
            },
        },
        503: {
            "model": ValidationErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "detail": "LLM backend unreachable at 'http://localhost:11434'"
                    }
                }
            },
        },
    },
)
def ingest(
    body: IngestRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    store: Annotated[VectorStore, Depends(get_store)],
    metadata_store: Annotated[
        IngestionMetadataStore,
        Depends(get_ingestion_metadata_store),
    ],
) -> IngestResponse:
    document = UploadedDocument(
        artifact_id=ArtifactId(body.artifact_id),
        filename=body.filename,
        content=body.content,
        project_ids=tuple(body.project_ids),
        source_role=body.source_role,
        semantic_boundaries=body.semantic_boundaries,
        contextualize=body.contextualize,
    )
    try:
        result = ingest_upload(document, IngestDeps(llm, store, metadata_store))
    except IngestError as error:
        raise HTTPException(
            status_code=_STATUS_FOR_FAILURE[error.kind], detail=error.detail
        ) from error
    return _response_from_records(result.artifact, result.chunks)
