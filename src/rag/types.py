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
    source_systems: list[SourceSystem] | None = None
    time_from: str | None = None
    time_to: str | None = None
    #: Only material belonging to one of these projects, plus material belonging
    #: to no project at all.
    #:
    #: A set rather than one id because the caller is usually a *person*, and a
    #: person can be on several projects — narrowing to one of them would hide
    #: their own material, and narrowing to none would show them everybody's.
    #: ``None`` (and an empty set) narrow nothing, so a single-project deployment
    #: is unaffected. See :func:`rag.filters.matches_retrieval_filters` for why
    #: unscoped material stays visible rather than disappearing.
    project_ids: frozenset[str] | None = None


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
    # 1-based line the chunk starts on in the source file. Only meaningful for
    # "text"/"code" chunks; PDFs track the source page instead (``start_page``).
    start_line: int | None = None
    # 1-based PDF page the chunk was extracted from. Only meaningful for "pdf"
    # chunks; text/code chunks track the source line instead (``start_line``).
    start_page: int | None = None
    # Which projects this material belongs to. A tuple rather than a single id
    # because one artifact genuinely can serve several projects -- a shared
    # repository is the ordinary case, not an edge one. Empty means "not scoped
    # to any project", which is treated as belonging to all of them.
    project_ids: tuple[str, ...] = ()


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
    #: See :attr:`Chunk.project_ids`. Carried through retrieval so a consumer can
    #: filter after the fact and a citation can say where the material belongs.
    project_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class Citation:
    artifact_id: str
    start_line: int | None = None
    start_page: int | None = None
