"""The steps the two ingest flows share: an uploaded document and a synced artifact."""

import base64
import logging
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Literal

from corpus.ids import ArtifactId
from corpus.types import Chunk
from ingestion.membership import revoke_removed_memberships
from ingestion.models import ParsedChunk
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore
from store.metadata_store import ArtifactRecord, IngestionMetadataStore

logger = logging.getLogger(__name__)

IngestFailure = Literal[
    "invalid_content", "too_large", "llm_unavailable", "store_failed"
]


class IngestError(Exception):
    """An ingest that ended in failure.

    ``kind`` says what went wrong, ``detail`` is what the caller reports and what
    the artifact record keeps, and ``log`` is the warning to write, if any.
    """

    def __init__(
        self, kind: IngestFailure, detail: str, log: str | None = None
    ) -> None:
        super().__init__(detail)
        self.kind: IngestFailure = kind
        self.detail = detail
        self.log = log


@dataclass(frozen=True)
class IngestDeps:
    llm: LLMClient
    store: VectorStore
    metadata_store: IngestionMetadataStore


@dataclass(frozen=True)
class ArtifactRef:
    id: str
    filename: str


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def decode_base64(text: str) -> bytes:
    """Decode strict base64, tolerating whitespace; raises ``ValueError`` otherwise."""
    return base64.b64decode("".join(text.split()), validate=True)


def open_scope(
    deps: IngestDeps,
    artifact_id: ArtifactId,
    project_ids: Iterable[str],
    request_time: str,
) -> tuple[str, tuple[str, ...]]:
    """When the artifact was first ingested and the projects it now belongs to.

    Projects the artifact has left lose access to its chunks right away, before
    anything is embedded.
    """
    existing = deps.metadata_store.get_artifact(artifact_id)
    created_at = existing.created_at if existing is not None else request_time
    projects = tuple(dict.fromkeys(pid for pid in project_ids if pid))
    revoke_removed_memberships(deps.store, artifact_id, projects)
    return created_at, projects


def clear_chunks(store: VectorStore, artifact_id: ArtifactId) -> None:
    store.delete(artifact_id, exclude_ids=[])


def complete(
    deps: IngestDeps, record: ArtifactRecord, chunk_count: int
) -> ArtifactRecord:
    done = replace(
        record, chunk_count=chunk_count, status="completed", updated_at=utc_now()
    )
    deps.metadata_store.save_completed_artifact(done)
    return done


def caption_images(
    llm: LLMClient,
    parsed: list[ParsedChunk],
    ref: ArtifactRef,
    *,
    skip_when_unavailable: bool,
) -> list[ParsedChunk]:
    """Replace each image chunk by its caption.

    Without a vision model an image is dropped when ``skip_when_unavailable`` is
    set and fails the ingest otherwise.
    """
    enriched: list[ParsedChunk] = []
    for chunk in parsed:
        if chunk.kind != "image":
            enriched.append(chunk)
            continue
        captioned = _caption(llm, chunk, ref, skip_when_unavailable)
        if captioned is not None:
            enriched.append(captioned)
    return enriched


def _caption(
    llm: LLMClient, chunk: ParsedChunk, ref: ArtifactRef, skip: bool
) -> ParsedChunk | None:
    try:
        image_bytes = decode_base64(chunk.content)
    except ValueError as exc:
        raise IngestError(
            "invalid_content",
            f"Image content for {ref.filename!r} is not valid base64.",
            log=(
                f"Image content for artifact {ref.id} ({ref.filename}) "
                f"is not valid base64: {exc}"
            ),
        ) from exc

    try:
        caption = llm.caption_image(image_bytes)
    except LLMUnavailableError as exc:
        if skip:
            logger.warning(
                "Vision model unavailable — skipping image chunk in %s", ref.filename
            )
            return None
        raise IngestError(
            "llm_unavailable",
            str(exc),
            log=(
                f"Vision model unavailable for artifact {ref.id} "
                f"({ref.filename}): {exc}"
            ),
        ) from exc
    return ParsedChunk(content=caption, kind="image", metadata=chunk.metadata)


def embed_chunks(
    llm: LLMClient,
    parsed: list[ParsedChunk],
    build: Callable[[ParsedChunk, list[float], int], Chunk],
) -> list[Chunk]:
    """Embed the chunks in one call and build a ``Chunk`` from each result."""
    embeddings = llm.embed_batch([chunk.content for chunk in parsed])
    return [
        build(chunk, embedding, index)
        for index, (chunk, embedding) in enumerate(zip(parsed, embeddings, strict=True))
    ]


def replace_chunks(
    store: VectorStore, artifact_id: ArtifactId, chunks: list[Chunk]
) -> None:
    """Store the new chunks, then drop the artifact's chunks from earlier runs."""
    store.add(chunks)
    store.delete(artifact_id, exclude_ids=[chunk.id for chunk in chunks])
