import hashlib

from ingestion.models import ParsedChunk
from ingestion.source_role import SourceRole
from rag.filters import normalize_source_system
from rag.types import Chunk, SourceSystem


def _deterministic_id(artifact_id: str, content: str, position: int) -> str:
    digest = hashlib.sha256(f"{artifact_id}:{position}:{content}".encode()).hexdigest()
    return f"chunk-{digest}"


def _optional_str(value: object) -> str | None:
    if value is None or value == "":
        return None

    return str(value)


def _source_timestamp_for(
    parsed: ParsedChunk,
    source_created_at: str | None,
    source_updated_at: str | None,
) -> str | None:
    return (
        source_updated_at
        or source_created_at
        or _optional_str(parsed.metadata.get("source_updated_at"))
        or _optional_str(parsed.metadata.get("source_created_at"))
        or _optional_str(parsed.metadata.get("updated_at"))
        or _optional_str(parsed.metadata.get("created_at"))
        or _optional_str(parsed.metadata.get("indexed_at"))
    )


def _optional_int(metadata: dict[str, str], key: str) -> int | None:
    raw = metadata.get(key)
    return int(raw) if raw is not None else None


def to_chunk(
    parsed: ParsedChunk,
    artifact_id: str,
    embedding: list[float],
    source_role: SourceRole = "primary",
    source_url: str | None = None,
    artifact_type: str | None = None,
    language: str | None = None,
    source_created_at: str | None = None,
    source_updated_at: str | None = None,
    source_system: SourceSystem | str | None = None,
    project_ids: tuple[str, ...] = (),
) -> Chunk:
    position = int(parsed.metadata.get("chunk_index", "0"))
    effective_source_system = normalize_source_system(
        str(source_system) if source_system is not None else None
    )

    if effective_source_system is None:
        effective_source_system = normalize_source_system(
            _optional_str(parsed.metadata.get("source_system"))
        )

    return Chunk(
        id=_deterministic_id(artifact_id, parsed.content, position),
        artifact_id=artifact_id,
        filename=str(parsed.metadata["filename"]),
        text=parsed.content,
        embedding=embedding,
        kind=parsed.kind,
        position=position,
        source_role=source_role,
        source_url=source_url or _optional_str(parsed.metadata.get("source_url")),
        artifact_type=artifact_type
        or _optional_str(parsed.metadata.get("artifact_type")),
        language=language or _optional_str(parsed.metadata.get("language")),
        source_system=effective_source_system,
        created_at=_source_timestamp_for(
            parsed,
            source_created_at=source_created_at,
            source_updated_at=source_updated_at,
        ),
        start_line=_optional_int(parsed.metadata, "start_line"),
        start_page=_optional_int(parsed.metadata, "page_number"),
        project_ids=project_ids,
    )
