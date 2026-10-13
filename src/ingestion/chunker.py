import os
from pathlib import Path

from ingestion.models import ChunkKind, ParsedChunk
from ingestion.utils import build_metadata

try:
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "512"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "64"))
except ValueError as err:
    raise ValueError("CHUNK_SIZE and CHUNK_OVERLAP must be an integer") from err

if chunk_overlap >= chunk_size:
    raise ValueError("chunk_overlap must be smaller than chunk_size")


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


def _paragraphs_with_start_lines(text: str) -> list[tuple[str, int]]:
    """Split text into non-empty paragraphs, keeping each one's start line.

    Mirrors ``text.split("\\n\\n")`` semantics (paragraphs separated by a
    blank line, stripped, empty ones dropped) while additionally tracking
    the 1-based line each surviving paragraph starts on in ``text``. This is
    the single source of truth for paragraph splitting: :func:`
    split_into_paragraphs` is defined in terms of it so both stay in sync.
    """
    paragraphs: list[tuple[str, int]] = []
    offset = 0
    for part in text.split("\n\n"):
        stripped = part.strip()
        if stripped:
            # ``split("\n\n")`` only breaks on exactly two newlines, so runs of
            # 3+ newlines (extra blank lines) leave a leading "\n" attached to
            # the *next* segment (e.g. "A\n\n\nB".split("\n\n") ==
            # ["A", "\nB"]). Without accounting for it, the start offset would
            # be off by however many extra blank lines preceded the paragraph.
            leading_whitespace = len(part) - len(part.lstrip())
            start_offset = offset + leading_whitespace
            # ``count("\n", 0, start_offset)`` is the number of newlines before
            # the paragraph, i.e. its 0-based line index; +1 makes it 1-based.
            line_no = text.count("\n", 0, start_offset) + 1
            paragraphs.append((stripped, line_no))
        offset += len(part) + 2  # +2 accounts for the removed "\n\n" delimiter
    return paragraphs


def split_into_paragraphs(text: str) -> list[str]:
    """Split text into non-empty, whitespace-stripped paragraphs.

    Paragraphs are separated by double newlines (``\\n\\n``).

    Args:
        text (str):
            Text content to split.

    Returns:
        list[str]:
            Only Paragraphs in document order.
    """
    return [paragraph for paragraph, _ in _paragraphs_with_start_lines(text)]


def _flush_paragraph_bucket(
    chunks: list[str],
    bucket: list[str],
) -> None:
    if bucket:
        chunks.append("\n\n".join(bucket))
        bucket.clear()


def _flush_paragraph_bucket_with_lines(
    chunks: list[tuple[str, int]],
    bucket: list[tuple[str, int]],
) -> None:
    if bucket:
        chunks.append(("\n\n".join(p for p, _ in bucket), bucket[0][1]))
        bucket.clear()


def group_paragraphs_into_chunks(
    paragraphs: list[str],
    chunk_size: int = chunk_size,
    chunk_overlap: int = chunk_overlap,
) -> list[str]:
    """Accumulate paragraphs into character-limited chunks.

    Paragraphs are accumulated into chunks without exceeding ``chunk_size``.
    When adding another paragraph would exceed ``chunk_size``, the current
    chunk is finalized and the next paragraph starts a fresh chunk. Paragraphs
    that exceed ``chunk_size`` on their own are hard-split into overlapping
    character chunks using ``chunk_overlap``.

    This grouping logic is shared by :func:`chunk_text` and the LLM-based
    context-aware chunker (when it is asked to only contextualize
    already-fixed chunk boundaries rather than choose semantic boundaries
    itself).

    Args:
        paragraphs (list[str]):
            Paragraphs in document order, e.g. from
            :func:`split_into_paragraphs`.

        chunk_size (int, optional):
            Maximum chunk size in characters.
            Defaults to the value configured via ``CHUNK_SIZE``.

        chunk_overlap (int, optional):
            Overlap used when hard-splitting oversized paragraphs.
            Defaults to the value configured via ``CHUNK_OVERLAP``.

    Returns:
        list[str]:
            Chunk contents (paragraphs joined by ``\\n\\n``).
    """
    chunks: list[str] = []
    paragraph_bucket: list[str] = []

    for paragraph in paragraphs:
        paragraph_length: int = len(paragraph)

        # hard split by character when paragraph itself exceeds chunk_size
        if paragraph_length > chunk_size:
            _flush_paragraph_bucket(chunks, paragraph_bucket)

            offset: int = 0
            while offset < len(paragraph):
                chunks.append(paragraph[offset : offset + chunk_size])
                offset += chunk_size - chunk_overlap

            continue

        # Would adding the paragraph exceed chunk_size?
        candidate_length: int = (
            len("\n\n".join(paragraph_bucket + [paragraph]))
            if paragraph_bucket
            else paragraph_length
        )

        # handle when paragraph_bucket + paragraph exceeds chunk size
        if paragraph_bucket and (candidate_length > chunk_size):
            _flush_paragraph_bucket(chunks, paragraph_bucket)

        # append paragraph to bucket
        paragraph_bucket.append(paragraph)

    # if there is some content left, flush it
    _flush_paragraph_bucket(chunks, paragraph_bucket)

    return chunks


def _group_paragraphs_into_chunks_with_lines(
    paragraphs: list[tuple[str, int]],
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, int]]:
    """Line-tracking counterpart of :func:`group_paragraphs_into_chunks`.

    Mirrors that function's grouping/hard-split decisions exactly (same
    ``chunk_size``/``chunk_overlap`` semantics), but operates on
    ``(paragraph, start_line)`` pairs so each resulting chunk can report the
    1-based line it starts on. Kept separate from
    :func:`group_paragraphs_into_chunks` (which the LLM-based context-aware
    chunker also calls with plain strings) to avoid changing that function's
    public contract.

    Args:
        paragraphs (list[tuple[str, int]]):
            Paragraphs with their 1-based start line, e.g. from
            :func:`_paragraphs_with_start_lines`.

        chunk_size (int):
            Maximum chunk size in characters.

        chunk_overlap (int):
            Overlap used when hard-splitting oversized paragraphs.

    Returns:
        list[tuple[str, int]]:
            One ``(chunk_content, start_line)`` pair per resulting chunk.
    """
    chunks: list[tuple[str, int]] = []
    paragraph_bucket: list[tuple[str, int]] = []

    for paragraph, line in paragraphs:
        paragraph_length: int = len(paragraph)

        # hard split by character when paragraph itself exceeds chunk_size
        if paragraph_length > chunk_size:
            _flush_paragraph_bucket_with_lines(chunks, paragraph_bucket)

            offset: int = 0
            while offset < len(paragraph):
                lines_before_piece = paragraph.count("\n", 0, offset)
                piece_start_line = line + lines_before_piece
                chunks.append(
                    (
                        paragraph[offset : offset + chunk_size],
                        piece_start_line,
                    )
                )
                offset += chunk_size - chunk_overlap

            continue

        # Would adding the paragraph exceed chunk_size?
        candidate_length: int = (
            len("\n\n".join([p for p, _ in paragraph_bucket] + [paragraph]))
            if paragraph_bucket
            else paragraph_length
        )

        # handle when paragraph_bucket + paragraph exceeds chunk size
        if paragraph_bucket and (candidate_length > chunk_size):
            _flush_paragraph_bucket_with_lines(chunks, paragraph_bucket)

        # append paragraph to bucket
        paragraph_bucket.append((paragraph, line))

    # if there is some content left, flush it
    _flush_paragraph_bucket_with_lines(chunks, paragraph_bucket)

    return chunks


def chunk_text(
    filename: str,
    text: str,
    chunk_size: int = chunk_size,
    chunk_overlap: int = chunk_overlap,
) -> list[ParsedChunk]:
    """Split text into paragraph-aware chunks.

    The function preserves paragraph boundaries by splitting on
    double newlines (``\\n\\n``). Paragraphs are accumulated until
    adding another paragraph would exceed the configured chunk size.

    When a chunk boundary is reached, the current chunk is finalized and the
    next paragraph starts a fresh chunk. Paragraphs that exceed ``chunk_size``
    on their own are split into overlapping character chunks by the given
    chunk_overlap.

    Args:
        filename (str):
            Name of the source file.

        text (str):
            Text content to split.

        chunk_size (int, optional):
            Maximum chunk size in characters.
            Defaults to the value configured via ``CHUNK_SIZE``.

        chunk_overlap (int, optional):
            Overlap used when hard-splitting oversized paragraphs.
            Defaults to the value configured via ``CHUNK_OVERLAP``.

    Returns:
        list[ParsedChunk]:
            Paragraph-aware text chunks with metadata.
    """
    paragraphs_with_lines = _paragraphs_with_start_lines(text)
    chunks = _group_paragraphs_into_chunks_with_lines(
        paragraphs_with_lines, chunk_size, chunk_overlap
    )

    total_chunks_amount = len(chunks)

    return [
        to_parsed_chunk(
            chunk_content,
            "text",
            filename,
            chunk_index,
            total_chunks_amount,
            start_line=start_line,
        )
        for chunk_index, (chunk_content, start_line) in enumerate(chunks)
    ]


def chunk_code(
    filename: str,
    code: str,
    chunk_size: int = chunk_size,
    start_line: int | None = None,
) -> list[ParsedChunk]:
    """Split large code blocks into smaller code chunks.

    The function preserves line boundaries and creates sequential
    code chunks that do not exceed the configured chunk size.

    Args:
        filename (str):
            Name of the source file.

        code (str):
            Source code content.

        chunk_size (int, optional):
            Maximum size of each chunk in characters.
            Defaults to the value configured via CHUNK_SIZE (fallback 512).

        start_line (int, optional):
            1-based line the given ``code`` starts on in the source file, if
            known. Every chunk produced from ``code`` is tagged with this
            same value, so it points at the enclosing symbol's definition
            rather than at each individual split's own offset.

    Returns:
        list[ParsedChunk]:
            Sequential code chunks with metadata and chunk indices.
    """
    lines: list[str] = code.splitlines()
    chunks_content: list[str] = []
    current_chunk_content: list[str] = []
    current_chunk_content_length = 0
    for line in lines:
        if (
            current_chunk_content_length + len(line) > chunk_size
            and current_chunk_content
        ):
            chunks_content.append("\n".join(current_chunk_content))
            current_chunk_content = []
            current_chunk_content_length = 0

        current_chunk_content.append(line)
        current_chunk_content_length += len(line) + 1  # + 1, because each added line
        # brings a line break (/n) with it

    if current_chunk_content:
        chunks_content.append("\n".join(current_chunk_content))

    total_chunks_amount: int = len(chunks_content)

    return [
        to_parsed_chunk(
            chunk_content,
            "code",
            filename,
            chunk_index,
            total_chunks_amount,
            start_line=start_line,
        )
        for chunk_index, chunk_content in enumerate(chunks_content)
    ]
