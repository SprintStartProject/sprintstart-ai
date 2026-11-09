"""Ingest of one uploaded document."""

import logging
import mimetypes
import os
from dataclasses import dataclass, replace

from corpus.ids import ArtifactId
from corpus.source_role import SourceRole, classify_source_role
from corpus.types import Chunk
from ingestion.mapper import to_chunk
from ingestion.models import ParsedChunk
from ingestion.parsing.parser import parse
from ingestion.pipeline.steps import (
    ArtifactRef,
    IngestDeps,
    IngestError,
    caption_images,
    clear_chunks,
    complete,
    embed_chunks,
    open_scope,
    replace_chunks,
    utc_now,
)
from llm.errors import LLMUnavailableError
from store.metadata_store import ArtifactRecord

logger = logging.getLogger(__name__)

_DEFAULT_MAX_CONTENT_LENGTH = 500000


@dataclass(frozen=True)
class UploadedDocument:
    artifact_id: ArtifactId
    filename: str
    content: str
    project_ids: tuple[str, ...] = ()
    source_role: SourceRole | None = None
    semantic_boundaries: bool = False
    contextualize: bool = False


@dataclass(frozen=True)
class IngestedUpload:
    artifact: ArtifactRecord
    chunks: list[Chunk]


def _content_type_for(filename: str) -> str:
    content_type, _ = mimetypes.guess_type(filename)
    return content_type or "application/octet-stream"


def ingest_upload(doc: UploadedDocument, deps: IngestDeps) -> IngestedUpload:
    """Parse, embed and store an uploaded document, replacing its earlier chunks.

    Raises ``IngestError`` after recording the failure on the artifact.
    """
    request_time = utc_now()
    created_at, project_ids = open_scope(
        deps, doc.artifact_id, doc.project_ids, request_time
    )
    record = ArtifactRecord(
        id=doc.artifact_id,
        filename=doc.filename,
        content_type=_content_type_for(doc.filename),
        source_type="file",
        size_bytes=len(doc.content.encode("utf-8")),
        chunk_count=0,
        status="processing",
        created_at=created_at,
        updated_at=request_time,
        project_ids=project_ids,
    )
    deps.metadata_store.save_artifact(record)
    try:
        return _ingest(doc, deps, record)
    except IngestError as error:
        deps.metadata_store.mark_failed(doc.artifact_id, error.detail, utc_now())
        raise


def _ingest(
    doc: UploadedDocument, deps: IngestDeps, record: ArtifactRecord
) -> IngestedUpload:
    max_length = int(
        os.getenv("INGEST_MAX_CONTENT_LENGTH", str(_DEFAULT_MAX_CONTENT_LENGTH))
    )
    if len(doc.content) > max_length:
        clear_chunks(deps.store, doc.artifact_id)
        raise IngestError(
            "too_large", f"Content exceeds maximum length of {max_length} characters."
        )

    parsed = _parse(doc, deps)
    if not parsed:
        return _finish_empty(doc, deps, record)

    enriched = caption_images(
        deps.llm,
        parsed,
        ArtifactRef(doc.artifact_id, doc.filename),
        skip_when_unavailable=True,
    )
    if not enriched:
        return _finish_empty(doc, deps, record)

    chunks = _embed(doc, deps, enriched)
    try:
        replace_chunks(deps.store, doc.artifact_id, chunks)
    except Exception as exc:
        raise IngestError(
            "store_failed", f"Failed to store chunks in vector database: {exc}"
        ) from exc

    completed = complete(deps, record, len(chunks))
    logger.info(
        "Ingested artifact %s (%d chunks) from %s",
        doc.artifact_id,
        len(chunks),
        doc.filename,
    )
    return IngestedUpload(completed, chunks)


def _parse(doc: UploadedDocument, deps: IngestDeps) -> list[ParsedChunk]:
    use_llm = doc.semantic_boundaries or doc.contextualize
    return parse(
        doc.filename,
        doc.content.encode("utf-8"),
        llm=deps.llm if use_llm else None,
        semantic_boundaries=doc.semantic_boundaries,
        contextualize=doc.contextualize,
    )


def _embed(
    doc: UploadedDocument, deps: IngestDeps, enriched: list[ParsedChunk]
) -> list[Chunk]:
    source_role = doc.source_role or classify_source_role(doc.filename)

    def build(chunk: ParsedChunk, embedding: list[float], index: int) -> Chunk:
        return replace(
            to_chunk(
                chunk,
                doc.artifact_id,
                embedding,
                source_role=source_role,
                source_system="UPLOAD",
                project_ids=doc.project_ids,
            ),
            position=index,
        )

    try:
        return embed_chunks(deps.llm, enriched, build)
    except LLMUnavailableError as exc:
        raise IngestError("llm_unavailable", str(exc)) from exc


def _finish_empty(
    doc: UploadedDocument, deps: IngestDeps, record: ArtifactRecord
) -> IngestedUpload:
    try:
        clear_chunks(deps.store, doc.artifact_id)
    except Exception as exc:
        raise IngestError(
            "store_failed", f"Failed to delete existing vector chunks: {exc}"
        ) from exc
    return IngestedUpload(complete(deps, record, 0), [])
