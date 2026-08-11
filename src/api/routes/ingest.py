import base64
import binascii
import logging
import mimetypes
import os
from dataclasses import replace
from datetime import UTC, datetime
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
from ingestion.mapper import to_chunk
from ingestion.membership import revoke_removed_memberships
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from ingestion.models import ParsedChunk
from ingestion.parser import parse
from ingestion.source_role import classify_source_role
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _content_type_for_filename(filename: str) -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


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
    request_time = _utc_now()
    content_bytes = body.content.encode("utf-8")

    existing = metadata_store.get_artifact(body.artifact_id)
    created_at = existing.created_at if existing is not None else request_time

    project_ids = tuple(dict.fromkeys(pid for pid in body.project_ids if pid))
    revoke_removed_memberships(store, body.artifact_id, project_ids)

    artifact = ArtifactRecord(
        id=body.artifact_id,
        filename=body.filename,
        content_type=_content_type_for_filename(body.filename),
        source_type="file",
        size_bytes=len(content_bytes),
        chunk_count=0,
        status="processing",
        created_at=created_at,
        updated_at=request_time,
        project_ids=project_ids,
    )
    metadata_store.save_artifact(artifact)

    max_length = int(os.getenv("INGEST_MAX_CONTENT_LENGTH", "500000"))
    if len(body.content) > max_length:
        detail = f"Content exceeds maximum length of {max_length} characters."
        store.delete(body.artifact_id, exclude_ids=[])
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=413, detail=detail)

    use_context_aware_chunking = body.semantic_boundaries or body.contextualize

    parsed_chunks = parse(
        body.filename,
        content_bytes,
        llm=llm if use_context_aware_chunking else None,
        semantic_boundaries=body.semantic_boundaries,
        contextualize=body.contextualize,
    )

    if not parsed_chunks:
        try:
            store.delete(body.artifact_id, exclude_ids=[])
        except Exception as exc:
            detail = f"Failed to delete existing vector chunks: {exc}"
            metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
            raise HTTPException(status_code=500, detail=detail) from exc
        completed_artifact = replace(
            artifact,
            chunk_count=0,
            status="completed",
            updated_at=_utc_now(),
        )
        metadata_store.save_completed_artifact(completed_artifact)
        return _response_from_records(completed_artifact, [])

    enriched: list[ParsedChunk] = []
    for chunk in parsed_chunks:
        if chunk.kind == "image":
            try:
                image_bytes = base64.b64decode(
                    "".join(chunk.content.split()), validate=True
                )
            except (binascii.Error, ValueError) as exc:
                detail = f"Image content for {body.filename!r} is not valid base64."
                metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
                raise HTTPException(status_code=422, detail=detail) from exc

            try:
                caption = llm.caption_image(image_bytes)
                enriched.append(
                    ParsedChunk(content=caption, kind="image", metadata=chunk.metadata)
                )
            except LLMUnavailableError:
                logger.warning(
                    "Vision model unavailable — skipping image chunk in %s",
                    body.filename,
                )
        else:
            enriched.append(chunk)

    if not enriched:
        try:
            store.delete(body.artifact_id, exclude_ids=[])
        except Exception as exc:
            detail = f"Failed to delete existing vector chunks: {exc}"
            metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
            raise HTTPException(status_code=500, detail=detail) from exc

        completed_artifact = replace(
            artifact,
            chunk_count=0,
            status="completed",
            updated_at=_utc_now(),
        )
        metadata_store.save_completed_artifact(completed_artifact)
        return _response_from_records(completed_artifact, [])

    source_role = body.source_role or classify_source_role(body.filename)
    try:
        embeddings = llm.embed_batch([chunk.content for chunk in enriched])
        chunks = [
            replace(
                to_chunk(
                    chunk,
                    body.artifact_id,
                    embedding,
                    source_role=source_role,
                    source_system="UPLOAD",
                    project_ids=body.project_ids,
                ),
                position=index,
            )
            for index, (chunk, embedding) in enumerate(
                zip(enriched, embeddings, strict=True)
            )
        ]
    except LLMUnavailableError as exc:
        detail = str(exc)
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=503, detail=detail) from exc

    try:
        store.add(chunks)
        store.delete(body.artifact_id, exclude_ids=[chunk.id for chunk in chunks])
    except Exception as exc:
        detail = f"Failed to store chunks in vector database: {exc}"
        metadata_store.mark_failed(body.artifact_id, detail, _utc_now())
        raise HTTPException(status_code=500, detail=detail) from exc

    completed_artifact = replace(
        artifact,
        chunk_count=len(chunks),
        status="completed",
        updated_at=_utc_now(),
    )

    metadata_store.save_completed_artifact(completed_artifact)

    logger.info(
        "Ingested artifact %s (%d chunks) from %s",
        body.artifact_id,
        len(chunks),
        body.filename,
    )

    return _response_from_records(completed_artifact, chunks)
