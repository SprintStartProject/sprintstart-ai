"""Fitting LLM-planned chunks into the size budget as ``ParsedChunk``s."""

from ingestion.chunking.parsed_chunk import to_parsed_chunk
from ingestion.models import ParsedChunk

Piece = tuple[str, tuple[int, int] | None, bool, tuple[int, int] | None]
"""A chunk: content, context-block range, has-overlap flag, overlap range."""

# A context block is a brief situating sentence. Blocks shorter than
# this threshold carry negligible situating value (noise), while blocks
# exceeding this fraction of chunk_size would crowd out the chunk body.
_CONTEXT_MIN_CHARS = 15
_CONTEXT_MAX_FRACTION = 0.25


def sanitize_context_block(raw: str | None, chunk_size: int) -> str | None:
    """Validate and trim an LLM-provided context block.

    Drops blocks that are blank or too short to carry meaningful situating
    context (fewer than ~3 words), and truncates blocks that exceed the
    context budget fraction (25% of chunk_size by default) so adequate
    space remains for the chunk body.
    """
    if raw is None:
        return None
    text = raw.strip()
    min_len = min(_CONTEXT_MIN_CHARS, max(1, chunk_size // 4))
    if not text or len(text) < min_len:
        return None
    max_len = max(int(chunk_size * _CONTEXT_MAX_FRACTION), min_len)
    return text[:max_len] if len(text) > max_len else text


def hard_split_with_context(
    context_block: str,
    body: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Piece]:
    """Split an oversized ``context_block + body`` chunk into sub-chunks.

    The context block is re-prepended to every resulting sub-chunk (it is
    small and self-contained, so duplicating it preserves the situating
    context each sub-chunk would otherwise lose). The remaining budget
    (``chunk_size`` minus the context block) is then hard-split the same
    way :func:`ingestion.chunking.chunker.group_paragraphs_into_chunks` hard-splits
    oversized paragraphs — overlapping character windows of ``chunk_size -
    chunk_overlap`` stride.

    Args:
        context_block (str):
            The situating context block text, already including its
            trailing separator (empty string if there is none).

        body (str):
            The chunk's own content, without the context block.

        chunk_size (int):
            Maximum size of each resulting sub-chunk, in characters,
            including the re-prepended context block.

        chunk_overlap (int):
            Overlap between consecutive sub-chunk bodies.

    Returns:
        list[Piece]:
            One tuple per sub-chunk: ``(content, context_block_range,
            has_overlap, overlap_range)``.
    """
    results: list[Piece] = []
    context_len = len(context_block)
    body_budget = max(chunk_size - context_len, 1)
    body_overlap = min(chunk_overlap, body_budget - 1) if body_budget > 1 else 0

    start = 0
    first = True
    while start < len(body) or first:
        body_slice = body[start : start + body_budget]
        content = context_block + body_slice

        context_range = (0, context_len) if context_len else None
        actual_overlap = min(body_overlap, len(body_slice))
        has_overlap = not first and actual_overlap > 0
        overlap_range = (
            (context_len, context_len + actual_overlap) if has_overlap else None
        )

        results.append((content, context_range, has_overlap, overlap_range))

        if start + body_budget >= len(body):
            break
        start += body_budget - body_overlap
        first = False

    return results


def fit_chunks(
    raw_chunks: list[str],
    context_blocks: dict[str, str],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Piece]:
    """Prepend each chunk's context block and hard-split those over ``chunk_size``."""
    pieces: list[Piece] = []
    for order_index, chunk_body in enumerate(raw_chunks):
        context_text = sanitize_context_block(
            context_blocks.get(str(order_index)), chunk_size
        )
        context_block = f"{context_text}\n\n" if context_text else ""
        combined = context_block + chunk_body

        if len(combined) <= chunk_size:
            context_range = (0, len(context_block)) if context_block else None
            pieces.append((combined, context_range, False, None))
            continue

        pieces.extend(
            hard_split_with_context(
                context_block, chunk_body, chunk_size, chunk_overlap
            )
        )
    return pieces


def to_parsed_chunks_from_pieces(
    filename: str, pieces: list[Piece]
) -> list[ParsedChunk]:
    total_chunks_amount = len(pieces)
    return [
        to_parsed_chunk(
            content,
            "text",
            filename,
            chunk_index,
            total_chunks_amount,
            has_context_block=context_range is not None,
            context_block_range=context_range,
            has_overlap=has_overlap,
            overlap_range=overlap_range,
        )
        for chunk_index, (
            content,
            context_range,
            has_overlap,
            overlap_range,
        ) in enumerate(pieces)
    ]
