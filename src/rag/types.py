from dataclasses import dataclass
from typing import Literal, TypeGuard

from ingestion.source_role import SourceRole

ChunkKind = Literal["text", "code", "pdf", "image"]
SourceSystem = Literal["GITHUB", "JIRA", "UPLOAD"]


def is_chunk_kind(value: str) -> TypeGuard[ChunkKind]:
    return value in ("text", "code", "pdf", "image")


def is_source_system(value: str) -> TypeGuard[SourceSystem]:
    return value in ("GITHUB", "JIRA", "UPLOAD")


@dataclass(frozen=True)
class RetrievalFilters:
    # The project the retrieval is scoped to. Chunks that don't belong to it —
    # including chunks with no project at all — are never returned. Retrieval
    # is deliberately fail-closed here: a missing project association makes
    # content invisible rather than visible to everyone (see ``rag.filters``).
    project_id: str | None = None
    #: The same rule as :attr:`project_id`, for a caller scoped to *several*
    #: projects at once -- a hire on two teams is the ordinary case. A chunk is
    #: eligible if it belongs to any of them, and, exactly as with
    #: ``project_id``, a chunk belonging to no project at all is eligible for
    #: none of them.
    #:
    #: Set both and a chunk must satisfy both, which is rarely what a caller
    #: wants; pass whichever matches the caller's cardinality, not both.
    project_ids: frozenset[str] | None = None
    source_systems: list[SourceSystem] | None = None
    time_from: str | None = None
    time_to: str | None = None


@dataclass(frozen=True)
class Chunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    embedding: list[float]
    kind: ChunkKind = "text"
    position: int | None = None
    source_role: SourceRole = "primary"
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None
    connector_id: str | None = None
    connector_source_id: str | None = None
    source_system: SourceSystem | None = None
    created_at: str | None = None
    # Projects the owning artifact belongs to. The backend owns this mapping
    # (``artifact_projects``) and sends it on ingest; an empty tuple means the
    # chunk predates project separation and is unreachable by project-scoped
    # retrieval until it is re-synced.
    project_ids: tuple[str, ...] = ()
    # 1-based line the chunk starts on in the source file. Only meaningful for
    # "text"/"code" chunks; PDFs track the source page instead (``start_page``).
    start_line: int | None = None
    # 1-based PDF page the chunk was extracted from. Only meaningful for "pdf"
    # chunks; text/code chunks track the source line instead (``start_line``).
    start_page: int | None = None


@dataclass(frozen=True)
class ScoredChunk:
    id: str
    artifact_id: str
    filename: str
    text: str
    score: float
    kind: ChunkKind = "text"
    position: int | None = None
    source_role: SourceRole = "primary"
    source_url: str | None = None
    artifact_type: str | None = None
    language: str | None = None
    connector_id: str | None = None
    connector_source_id: str | None = None
    source_system: SourceSystem | None = None
    created_at: str | None = None
    start_line: int | None = None
    start_page: int | None = None
    #: See :attr:`Chunk.project_ids`. Carried through retrieval so a citation can
    #: say where the material belongs.
    project_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    artifact_id: str
    start_line: int | None = None
    start_page: int | None = None
