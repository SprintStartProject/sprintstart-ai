"""POST /api/v1/ingest/sync — batch ingest for completed GitHub ingestion runs."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    ArtifactRunIngestRequest,
    ArtifactRunIngestResponse,
    RunArtifactsSyncRequest,
    RunArtifactsSyncResponse,
)
from ingestion.mapper import to_chunk
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from ingestion.parser import parse
from ingestion.source_role import classify_source_role
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

# Each artifact already batches its own chunks into one embed_batch() call;
# this bounds how many artifacts are embedded concurrently across the batch,
# since /ingest/sync otherwise processes them one at a
# time and a few hundred artifacts can take minutes purely from sequential
# network round-trips to the embedding API. Kept modest to stay well under
# typical provider rate limits -- override via INGEST_CONCURRENCY.
_DEFAULT_INGEST_CONCURRENCY = 8


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _source_system_for(artifact: ArtifactRunIngestRequest) -> str:
    return artifact.source_system or "GITHUB"


def _filename_for(artifact: ArtifactRunIngestRequest) -> str:
    """Derive a filename from the artifact metadata.

    For FILE artifacts the relative path is embedded in sourceId as the last
    colon-separated segment, so we preserve it in full (including directory) so
    that citations remain unambiguous when multiple files share the same basename.
    All other types use .md.
    """
    if artifact.artifact_type == "FILE":
        # sourceId format: "github:owner/repo:FILE:src/main/App.kt"
        path_segment = artifact.source_id.rsplit(":", 1)[-1]
        return path_segment or f"{artifact.artifact_id}.txt"

    slug = artifact.source_id.rsplit(":", 1)[-1]
    type_prefix = artifact.artifact_type.lower().replace("_", "-")
    return f"{type_prefix}-{slug}.md"


def _connector_source_id_for(artifact: ArtifactRunIngestRequest) -> str | None:
    """Derive the connector-level source id (e.g. 'owner/repo') from sourceId.

    sourceId format: "github:owner/repo:TYPE:unique" — the connector-level
    source identity is always the second colon-separated segment, regardless
    of artifact type. This is distinct from the artifact-level sourceId
    itself, and from the enclosing connector id (artifact.source_system).
    """
    parts = artifact.source_id.split(":", 3)
    return parts[1] if len(parts) >= 2 else None


def _assemble_content(artifact: ArtifactRunIngestRequest) -> str:
    parts: list[str] = []
    if artifact.title:
        parts.append(f"# {artifact.title}")
    if artifact.body_text:
        parts.append(artifact.body_text)
    return "\n\n".join(parts)


def _ingest_one(
    artifact: ArtifactRunIngestRequest,
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
) -> ArtifactRunIngestResponse:
    request_time = _utc_now()
    filename = _filename_for(artifact)
    content = _assemble_content(artifact)
    content_bytes = content.encode("utf-8")

    max_length = int(os.getenv("INGEST_MAX_CONTENT_LENGTH", "500000"))

    existing = metadata_store.get_artifact(artifact.artifact_id)
    created_at = existing.created_at if existing is not None else request_time

    source_system = _source_system_for(artifact)

    record = ArtifactRecord(
        id=artifact.artifact_id,
        filename=filename,
        content_type=artifact.mime or "text/plain",
        source_type=source_system.lower(),
        size_bytes=len(content_bytes),
        chunk_count=0,
        status="processing",
        created_at=created_at,
        updated_at=request_time,
        source_id=artifact.source_id,
        source_url=artifact.source_url,
        artifact_type=artifact.artifact_type,
        language=artifact.language,
        state=artifact.state,
        has_assignee=artifact.has_assignee,
        labels=artifact.labels,
    )

    if len(content) > max_length:
        logger.warning(
            "Artifact %s exceeds max content length (%d > %d), skipping",
            artifact.artifact_id,
            len(content),
            max_length,
        )
        store.delete(artifact.artifact_id, exclude_ids=[])
        completed = replace(record, status="completed", updated_at=_utc_now())
        metadata_store.save_completed_artifact(completed)
        return ArtifactRunIngestResponse(
            artifact_id=artifact.artifact_id, chunk_count=0, status="completed"
        )

    metadata_store.save_artifact(record)

    if not content:
        store.delete(artifact.artifact_id, exclude_ids=[])
        completed = replace(record, status="completed", updated_at=_utc_now())
        metadata_store.save_completed_artifact(completed)
        return ArtifactRunIngestResponse(
            artifact_id=artifact.artifact_id, chunk_count=0, status="completed"
        )

    try:
        parsed_chunks = parse(filename, content_bytes)
    except Exception as exc:
        logger.warning(
            "Failed to parse artifact %s (%s): %s",
            artifact.artifact_id,
            filename,
            exc,
        )
        metadata_store.mark_failed(artifact.artifact_id, str(exc), _utc_now())
        return ArtifactRunIngestResponse(
            artifact_id=artifact.artifact_id, chunk_count=0, status="failed"
        )

    if not parsed_chunks:
        store.delete(artifact.artifact_id, exclude_ids=[])
        completed = replace(record, status="completed", updated_at=_utc_now())
        metadata_store.save_completed_artifact(completed)
        return ArtifactRunIngestResponse(
            artifact_id=artifact.artifact_id, chunk_count=0, status="completed"
        )

    source_role = classify_source_role(filename)

    try:
        embeddings = llm.embed_batch([chunk.content for chunk in parsed_chunks])
        chunks = [
            replace(
                to_chunk(
                    chunk,
                    artifact.artifact_id,
                    embedding,
                    source_role=source_role,
                    source_url=artifact.source_url,
                    artifact_type=artifact.artifact_type,
                    language=artifact.language,
                    source_created_at=artifact.source_created_at,
                    source_updated_at=artifact.source_updated_at,
                    source_system=source_system,
                    project_ids=tuple(artifact.project_ids),
                ),
                position=index,
                connector_id=source_system.lower(),
                connector_source_id=_connector_source_id_for(artifact),
            )
            for index, (chunk, embedding) in enumerate(
                zip(parsed_chunks, embeddings, strict=True)
            )
        ]
    except LLMUnavailableError as exc:
        # Mid-batch LLM outages must not sink the whole request:
        # record this artifact as failed and let the caller retry it later while
        # the rest of the batch still gets a chance to ingest.
        metadata_store.mark_failed(artifact.artifact_id, str(exc), _utc_now())
        return ArtifactRunIngestResponse(
            artifact_id=artifact.artifact_id, chunk_count=0, status="failed"
        )

    store.add(chunks)
    store.delete(artifact.artifact_id, exclude_ids=[chunk.id for chunk in chunks])

    completed = replace(
        record,
        chunk_count=len(chunks),
        status="completed",
        updated_at=_utc_now(),
    )
    metadata_store.save_completed_artifact(completed)

    return ArtifactRunIngestResponse(
        artifact_id=artifact.artifact_id, chunk_count=len(chunks), status="completed"
    )


def _ingest_one_safe(
    artifact: ArtifactRunIngestRequest,
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
) -> ArtifactRunIngestResponse:
    try:
        return _ingest_one(artifact, llm, store, metadata_store)
    except Exception as exc:
        logger.exception("Failed to ingest artifact %s", artifact.artifact_id)
        metadata_store.mark_failed(
            artifact.artifact_id,
            f"Unexpected error during batch ingest: {exc}",
            _utc_now(),
        )
        return ArtifactRunIngestResponse(
            artifact_id=artifact.artifact_id, chunk_count=0, status="failed"
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

    for artifact_id in body.artifacts_to_deindex:
        try:
            deleted_count = store.delete(artifact_id, exclude_ids=[])
            if deleted_count > 0:
                metadata_store.mark_deindexed(artifact_id, _utc_now())
        except Exception:
            logger.exception("Failed to deindex artifact %s", artifact_id)

    results: list[ArtifactRunIngestResponse] = []
    if body.artifacts_to_ingest:
        concurrency = min(
            int(os.getenv("INGEST_CONCURRENCY", str(_DEFAULT_INGEST_CONCURRENCY))),
            len(body.artifacts_to_ingest),
        )

        def _ingest(artifact: ArtifactRunIngestRequest) -> ArtifactRunIngestResponse:
            return _ingest_one_safe(artifact, llm, store, metadata_store)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(_ingest, body.artifacts_to_ingest))

    logger.info(
        "Sync complete: %d ingested, %d deindexed",
        len(results),
        len(body.artifacts_to_deindex),
    )

    return RunArtifactsSyncResponse(artifacts=results)
