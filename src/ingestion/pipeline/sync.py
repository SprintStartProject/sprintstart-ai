"""Ingest of the artifacts a completed ingestion run hands over."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Literal

from corpus.ids import ArtifactId, SourceId
from corpus.source_role import classify_source_role
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
    decode_base64,
    embed_chunks,
    open_scope,
    replace_chunks,
    utc_now,
)
from llm.errors import LLMUnavailableError
from store.metadata_store import ArtifactRecord

logger = logging.getLogger(__name__)

# Each artifact already batches its own chunks into one embed_batch() call; this
# bounds how many artifacts are embedded concurrently across the batch, since
# processing them one at a time makes a few hundred artifacts take minutes purely
# from sequential round-trips to the embedding API. Kept modest to stay well under
# typical provider rate limits; override with INGEST_CONCURRENCY.
_DEFAULT_INGEST_CONCURRENCY = 8
_DEFAULT_MAX_CONTENT_LENGTH = 500000
_DEFAULT_MAX_BINARY_BYTES = 10 * 1024 * 1024  # 10 MB upload limit


@dataclass(frozen=True)
class SyncedArtifact:
    artifact_id: ArtifactId
    source_id: SourceId
    artifact_type: str
    project_ids: tuple[str, ...] = ()
    source_system: str | None = None
    source_url: str | None = None
    title: str | None = None
    body_text: str | None = None
    mime: str | None = None
    language: str | None = None
    source_created_at: str | None = None
    source_updated_at: str | None = None
    state: str | None = None
    has_assignee: bool | None = None
    labels: tuple[str, ...] = ()


@dataclass(frozen=True)
class SyncedResult:
    artifact_id: ArtifactId
    chunk_count: int
    status: Literal["completed", "failed"]


@dataclass(frozen=True)
class DeindexOutcome:
    artifact_id: ArtifactId
    error: str | None = None


def source_system_for(artifact: SyncedArtifact) -> str:
    return artifact.source_system or "GITHUB"


def is_binary(artifact: SyncedArtifact) -> bool:
    mime = artifact.mime
    return mime is not None and (mime.startswith("image/") or mime == "application/pdf")


def filename_for(artifact: SyncedArtifact) -> str:
    """Derive a filename from the artifact metadata.

    For FILE artifacts from uploads (sourceSystem="UPLOAD"), the original filename
    is preserved in artifact.title, while sourceId is a raw upload UUID without an
    extension. Suffix-based dispatch in parse() needs the extension, so we prefer
    title.

    For FILE artifacts from GitHub, the relative path is embedded in sourceId as
    the last colon-separated segment, so we preserve it in full (including
    directory) so that citations remain unambiguous when multiple files share the
    same basename. All other types use .md.
    """
    if artifact.artifact_type == "FILE":
        if source_system_for(artifact).upper() == "UPLOAD" and artifact.title:
            return artifact.title

        # sourceId format: "github:owner/repo:FILE:src/main/App.kt"
        path_segment = artifact.source_id.rsplit(":", 1)[-1]
        return path_segment or f"{artifact.artifact_id}.txt"

    slug = artifact.source_id.rsplit(":", 1)[-1]
    type_prefix = artifact.artifact_type.lower().replace("_", "-")
    return f"{type_prefix}-{slug}.md"


def connector_source_id_for(artifact: SyncedArtifact) -> str | None:
    """Derive the connector-level source id (e.g. 'owner/repo') from sourceId.

    sourceId format: "github:owner/repo:TYPE:unique" — the connector-level
    source identity is always the second colon-separated segment, regardless
    of artifact type. This is distinct from the artifact-level sourceId
    itself, and from the enclosing connector id (artifact.source_system).
    """
    parts = artifact.source_id.split(":", 3)
    return parts[1] if len(parts) >= 2 else None


def assemble_content(artifact: SyncedArtifact) -> str:
    if is_binary(artifact):
        return artifact.body_text or ""
    parts: list[str] = []
    if artifact.title:
        parts.append(f"# {artifact.title}")
    if artifact.body_text:
        parts.append(artifact.body_text)
    return "\n\n".join(parts)


def ingest_all(artifacts: list[SyncedArtifact], deps: IngestDeps) -> list[SyncedResult]:
    """Ingest the artifacts concurrently, keeping the order they were given in."""
    if not artifacts:
        return []
    concurrency = min(
        int(os.getenv("INGEST_CONCURRENCY", str(_DEFAULT_INGEST_CONCURRENCY))),
        len(artifacts),
    )

    def ingest(artifact: SyncedArtifact) -> SyncedResult:
        return ingest_synced_safe(artifact, deps)

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        return list(executor.map(ingest, artifacts))


def deindex(artifact_ids: list[ArtifactId], deps: IngestDeps) -> list[DeindexOutcome]:
    outcomes: list[DeindexOutcome] = []
    for artifact_id in artifact_ids:
        try:
            deleted_count = deps.store.delete(artifact_id, exclude_ids=[])
            if deleted_count > 0:
                deps.metadata_store.mark_deindexed(artifact_id, utc_now())
        except Exception as exc:
            # VectorStore.delete leaves a durable tombstone on failure, so
            # retrieval stays fail-closed while the backend schedules a retry.
            logger.exception("Failed to deindex artifact %s", artifact_id)
            outcomes.append(DeindexOutcome(artifact_id, str(exc)))
        else:
            outcomes.append(DeindexOutcome(artifact_id))
    return outcomes


def ingest_synced_safe(artifact: SyncedArtifact, deps: IngestDeps) -> SyncedResult:
    """Like ``ingest_synced``, and a failure nobody foresaw only fails this artifact."""
    try:
        return ingest_synced(artifact, deps)
    except Exception as exc:
        logger.exception("Failed to ingest artifact %s", artifact.artifact_id)
        deps.metadata_store.mark_failed(
            artifact.artifact_id,
            f"Unexpected error during batch ingest: {exc}",
            utc_now(),
        )
        return SyncedResult(artifact.artifact_id, 0, "failed")


def ingest_synced(artifact: SyncedArtifact, deps: IngestDeps) -> SyncedResult:
    """Ingest one synced artifact; a failure is recorded and reported, not raised."""
    try:
        return _ingest(artifact, deps)
    except IngestError as error:
        if error.log:
            logger.warning(error.log)
        deps.metadata_store.mark_failed(artifact.artifact_id, error.detail, utc_now())
        return SyncedResult(artifact.artifact_id, 0, "failed")


def _ingest(artifact: SyncedArtifact, deps: IngestDeps) -> SyncedResult:
    filename = filename_for(artifact)
    content = assemble_content(artifact)
    decoded_size = _decoded_size(artifact, content)
    record = _open_record(artifact, filename, _size_of(content, decoded_size), deps)

    reason = _oversize_reason(artifact, content, decoded_size)
    if reason is not None:
        logger.warning(reason)
        return _finish_empty(artifact, deps, record)

    deps.metadata_store.save_artifact(record)
    if not content:
        return _finish_empty(artifact, deps, record)

    parsed = _parse(artifact, filename, content)
    if not parsed:
        return _finish_empty(artifact, deps, record)

    enriched = caption_images(
        deps.llm,
        parsed,
        ArtifactRef(artifact.artifact_id, filename),
        skip_when_unavailable=False,
    )
    if not enriched:
        return _finish_empty(artifact, deps, record)

    chunks = _embed(artifact, filename, deps, enriched)
    replace_chunks(deps.store, artifact.artifact_id, chunks)
    complete(deps, record, len(chunks))
    return SyncedResult(artifact.artifact_id, len(chunks), "completed")


def _decoded_size(artifact: SyncedArtifact, content: str) -> int | None:
    if not (is_binary(artifact) and content):
        return None
    try:
        return len(decode_base64(content))
    except ValueError:
        return None


def _size_of(content: str, decoded_size: int | None) -> int:
    return decoded_size if decoded_size is not None else len(content.encode("utf-8"))


def _open_record(
    artifact: SyncedArtifact, filename: str, size_bytes: int, deps: IngestDeps
) -> ArtifactRecord:
    request_time = utc_now()
    created_at, project_ids = open_scope(
        deps, artifact.artifact_id, artifact.project_ids, request_time
    )
    return ArtifactRecord(
        id=artifact.artifact_id,
        filename=filename,
        content_type=artifact.mime or "text/plain",
        source_type=source_system_for(artifact).lower(),
        size_bytes=size_bytes,
        chunk_count=0,
        status="processing",
        created_at=created_at,
        updated_at=request_time,
        source_id=artifact.source_id,
        source_url=artifact.source_url,
        artifact_type=artifact.artifact_type,
        language=artifact.language,
        project_ids=project_ids,
        state=artifact.state,
        has_assignee=artifact.has_assignee,
        labels=list(artifact.labels),
    )


def _oversize_reason(
    artifact: SyncedArtifact, content: str, decoded_size: int | None
) -> str | None:
    if is_binary(artifact):
        limit = int(
            os.getenv("INGEST_MAX_BINARY_BYTES", str(_DEFAULT_MAX_BINARY_BYTES))
        )
        if decoded_size is None or decoded_size <= limit:
            return None
        return (
            f"Artifact {artifact.artifact_id} exceeds max binary size "
            f"({decoded_size} > {limit} bytes), skipping"
        )
    max_length = int(
        os.getenv("INGEST_MAX_CONTENT_LENGTH", str(_DEFAULT_MAX_CONTENT_LENGTH))
    )
    if len(content) <= max_length:
        return None
    return (
        f"Artifact {artifact.artifact_id} exceeds max content length "
        f"({len(content)} > {max_length}), skipping"
    )


def _parse(artifact: SyncedArtifact, filename: str, content: str) -> list[ParsedChunk]:
    content_bytes = content.encode("utf-8")
    if artifact.mime == "application/pdf":
        content_bytes = _decode_pdf(artifact, filename, content)
    try:
        return parse(filename, content_bytes)
    except Exception as exc:
        raise IngestError(
            "invalid_content",
            str(exc),
            log=f"Failed to parse artifact {artifact.artifact_id} ({filename}): {exc}",
        ) from exc


def _decode_pdf(artifact: SyncedArtifact, filename: str, content: str) -> bytes:
    try:
        return decode_base64(content)
    except ValueError as exc:
        raise IngestError(
            "invalid_content",
            f"PDF content for {filename!r} is not valid base64.",
            log=(
                f"PDF content for artifact {artifact.artifact_id} ({filename}) "
                f"is not valid base64: {exc}"
            ),
        ) from exc


def _embed(
    artifact: SyncedArtifact,
    filename: str,
    deps: IngestDeps,
    enriched: list[ParsedChunk],
) -> list[Chunk]:
    source_system = source_system_for(artifact)
    source_role = classify_source_role(filename)

    def build(chunk: ParsedChunk, embedding: list[float], index: int) -> Chunk:
        return replace(
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
                project_ids=artifact.project_ids,
            ),
            position=index,
            connector_id=source_system.lower(),
            connector_source_id=connector_source_id_for(artifact),
        )

    try:
        return embed_chunks(deps.llm, enriched, build)
    except LLMUnavailableError as exc:
        raise IngestError("llm_unavailable", str(exc)) from exc


def _finish_empty(
    artifact: SyncedArtifact, deps: IngestDeps, record: ArtifactRecord
) -> SyncedResult:
    clear_chunks(deps.store, artifact.artifact_id)
    complete(deps, record, 0)
    return SyncedResult(artifact.artifact_id, 0, "completed")
