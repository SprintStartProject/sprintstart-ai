"""The ParsedChunk a chunker emits, with the metadata every chunk carries."""

from pathlib import Path

from ingestion.models import ChunkKind, ParsedChunk
from ingestion.utils import build_metadata


def to_parsed_chunk(
    chunk_content: str,
    kind: ChunkKind,
    filename: str,
    chunk_index: int,
    total_chunks_amount: int,
    start_line: int | None = None,
    has_context_block: bool = False,
    context_block_range: tuple[int, int] | None = None,
    has_overlap: bool = False,
    overlap_range: tuple[int, int] | None = None,
):
    """Create a ParsedChunk with standard metadata.

    Args:
        chunk_content (str):
            Content of the chunk.

        kind (ChunkKind):
            Chunk type ("text", "code", "pdf", or "image").

        filename (str):
            Name of the source file.

        chunk_index (int):
            Zero-based index of the chunk within the source file.

        total_chunks_amount (int):
            Total number of chunks produced for the source file.

        start_line (int, optional):
            1-based line the chunk starts on in the source file, if known.

        has_context_block (bool, optional):
            Whether a situating context block was prepended to the chunk
            content. Defaults to ``False``.

        context_block_range (tuple[int, int] | None, optional):
            ``(start, end)`` character range of the context block within
            ``chunk_content``, if ``has_context_block`` is ``True``.
            Defaults to ``None``.

        has_overlap (bool, optional):
            Whether the chunk contains character/paragraph overlap carried
            over from a neighboring chunk. Defaults to ``False``.

        overlap_range (tuple[int, int] | None, optional):
            ``(start, end)`` character range of the overlap within
            ``chunk_content``, if ``has_overlap`` is ``True``. Defaults to
            ``None``.

    Returns:
        ParsedChunk:
            Chunk instance with content, type and metadata.
    """
    metadata = {
        **build_metadata(Path(filename)),
        "chunk_index": str(chunk_index),
        "total_chunks": str(total_chunks_amount),
        "has_context_block": str(has_context_block).lower(),
        "context_block_range": (
            f"{context_block_range[0]}:{context_block_range[1]}"
            if context_block_range is not None
            else ""
        ),
        "has_overlap": str(has_overlap).lower(),
        "overlap_range": (
            f"{overlap_range[0]}:{overlap_range[1]}"
            if overlap_range is not None
            else ""
        ),
    }
    if start_line is not None:
        metadata["start_line"] = str(start_line)

    return ParsedChunk(
        content=chunk_content,
        kind=kind,
        metadata=metadata,
    )
