"""LLM-based context-aware chunking strategy for text content.

This is an opt-in, additive alternative to the paragraph-boundary
``chunk_text`` strategy in :mod:`ingestion.chunker`. Instead of splitting on
character length alone, an LLM is given the full text (split into
paragraphs) and asked to choose semantically coherent chunk boundaries and,
optionally, a short situating "context block" for chunks that would benefit from one.

Both behaviors share a single LLM call but are independently toggleable:

- ``semantic_boundaries``: let the LLM choose paragraph groupings instead of
  the character-length accumulation used by ``chunk_text``.
- ``contextualize``: let the LLM flag chunks that need a short context block
  and prepend it to their content.

If either the text is too large for the LLM, or the LLM output cannot be
parsed, the whole call falls back to :func:`ingestion.chunker.chunk_text`.

The LLM is asked to reason over small integer *indices*, not character
offsets — LLMs are unreliable at counting characters, but reliably reproduce
small integers that are put in front of each segment in the prompt.

Note on ``semantic_boundaries=False``: the caller has already grouped the
text into its final chunks (e.g. via ``chunk_text``'s paragraph-accumulation
logic) *before* calling this module, and passes those finished chunks in as
``segments``. This way the numbering the LLM sees lines up exactly with the
chunk order index that ``context_blocks`` must key off — there is no
separate "boundaries" step to reconcile. When ``semantic_boundaries=True``,
``segments`` are the raw paragraphs instead, and the LLM's own ``boundaries``
determine the resulting chunk grouping (and thus what order index each
chunk has).
"""

import json
import logging
import os

from pydantic import BaseModel, Field, ValidationError

from ingestion.chunker import (
    chunk_overlap as default_chunk_overlap,
)
from ingestion.chunker import (
    chunk_size as default_chunk_size,
)
from ingestion.chunker import (
    chunk_text,
    group_paragraphs_into_chunks,
    split_into_paragraphs,
    to_parsed_chunk,
)
from ingestion.models import ParsedChunk
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

try:
    # There is no tokenizer in this project (backend is configurable across
    # Ollama/OpenAI/Anthropic), so this is a character-count proxy for a
    # token-count limit rather than an exact one. ~4 chars/token is a common
    # rough estimate for English prose; the default is deliberately
    # conservative to leave headroom for the prompt scaffolding itself.
    max_chars: int = int(os.getenv("CONTEXT_AWARE_CHUNKING_MAX_CHARS", "24000"))
except ValueError as err:
    raise ValueError("CONTEXT_AWARE_CHUNKING_MAX_CHARS must be an integer") from err

# A context block is a brief situating sentence. Blocks shorter than
# this threshold carry negligible situating value (noise), while blocks
# exceeding this fraction of chunk_size would crowd out the chunk body.
_CONTEXT_MIN_CHARS = 15
_CONTEXT_MAX_FRACTION = 0.25


def _sanitize_context_block(raw: str | None, chunk_size: int) -> str | None:
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


def exceeds_llm_limit(text: str, max_chars: int = max_chars) -> bool:
    """Check whether ``text`` is too large to hand to the LLM as-is.

    This is a cheap character-count proxy for a token-count limit — used to
    decide upfront whether to attempt context-aware chunking at all, before
    spending an LLM call. Callers should fall back to
    :func:`ingestion.chunker.chunk_text` when this returns ``True``.

    Args:
        text (str):
            The full text that would be sent to the LLM.

        max_chars (int, optional):
            Maximum character count considered safe to send.
            Defaults to the value configured via
            ``CONTEXT_AWARE_CHUNKING_MAX_CHARS``.

    Returns:
        bool:
            ``True`` if ``text`` exceeds ``max_chars``.
    """
    return len(text) > max_chars


class ContextAwareChunkingError(Exception):
    """Raised when the LLM output cannot be parsed/validated.

    Callers should catch this and fall back to :func:`chunker.chunk_text`.
    """


class _Payload(BaseModel):
    # Segment indices where a new chunk starts (only meaningful when the LLM
    # was asked to choose boundaries itself). Must be sorted, start at 0, and
    # contain no duplicates or out-of-range indices.
    boundaries: list[int] = Field(default_factory=list[int])
    # Chunk *order* index (0-based, as string) -> short situating sentence.
    # Chunks missing from this dict need no context block.
    context_blocks: dict[str, str] = Field(default_factory=dict)


def _build_prompt(
    segments: list[str],
    semantic_boundaries: bool,
    contextualize: bool,
) -> list[Message]:
    """Build the chat prompt for the context-aware chunking LLM call.

    Args:
        segments (list[str]):
            The text to analyze, in document order. When
            ``semantic_boundaries`` is ``True`` these are raw paragraphs
            that the LLM groups into chunks itself via ``boundaries``. When
            ``semantic_boundaries`` is ``False`` these are already the
            caller's finished chunks — the LLM only fills
            ``context_blocks`` for them, keyed by their position here.

        semantic_boundaries (bool):
            Whether the LLM should choose chunk boundaries itself. When
            ``False``, the caller has already fixed the boundaries and the
            LLM is only asked to fill ``context_blocks`` for those chunks.

        contextualize (bool):
            Whether the LLM should propose context blocks at all. When
            ``False``, the LLM is instructed to always leave
            ``context_blocks`` empty.

    Returns:
        list[Message]:
            System + user messages ready to pass to ``LLMClient.generate``.
    """
    unit_name = "paragraph" if semantic_boundaries else "chunk"
    numbered_segments = "\n\n".join(
        f"[{index}] {segment}" for index, segment in enumerate(segments)
    )

    instructions: list[str] = []
    if semantic_boundaries:
        instructions.append(
            "1. 'boundaries': a sorted list of paragraph indices where a new "
            "chunk should start, based on topic shifts, section boundaries, "
            "or other semantic breaks. It must start with 0 and contain no "
            "duplicates or indices outside the given paragraph range."
        )
    else:
        instructions.append(
            "1. 'boundaries': always return an empty list — chunk "
            "boundaries are fixed by the caller and are not part of this "
            "task."
        )

    if contextualize:
        chunk_reference = (
            "the resulting chunk's order index (0-based, counted over the "
            "chunks formed by your own 'boundaries' — NOT the paragraph "
            "index)"
            if semantic_boundaries
            else "the chunk's index as given above"
        )
        instructions.append(
            "2. 'context_blocks': a mapping from " + chunk_reference + " (as "
            "a string) to a short (one to two sentence) situating sentence "
            "that gives the chunk context it would otherwise lack (e.g. "
            "what document/section it is part of, what it is about). Only "
            "include chunks that genuinely benefit from this — "
            "self-contained chunks should be omitted from the mapping "
            "entirely."
        )
    else:
        instructions.append(
            "2. 'context_blocks': always return an empty object — "
            "contextualization is not part of this task."
        )

    system = (
        f"You analyze a document that has been split into numbered "
        f"{unit_name}s, in order to prepare it for chunking in a retrieval "
        "system.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences) with "
        "exactly two keys:\n" + "\n".join(instructions) + "\n\n"
        'JSON schema: {"boundaries": [int], "context_blocks": '
        '{"<chunk_order_index>": str}}'
    )
    user = f"Numbered {unit_name}s:\n\n{numbered_segments}"

    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _request_chunking_plan(
    segments: list[str],
    llm: LLMClient,
    semantic_boundaries: bool,
    contextualize: bool,
) -> _Payload:
    """Call the LLM and parse its response into a validated ``_Payload``.

    Raises:
        ContextAwareChunkingError:
            If the LLM output is not valid/parseable JSON matching the
            expected schema.
    """
    messages = _build_prompt(segments, semantic_boundaries, contextualize)
    raw = llm.generate(messages)

    try:
        payload = _Payload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise ContextAwareChunkingError(
            f"invalid context-aware chunking output: {exc}"
        ) from exc

    num_segments = len(segments)
    if any(
        index < 0 or index >= num_segments for index in payload.boundaries
    ) or payload.boundaries != sorted(set(payload.boundaries)):
        raise ContextAwareChunkingError(
            "LLM returned invalid boundaries: "
            f"{payload.boundaries!r} for {num_segments} segments"
        )
    if payload.boundaries and payload.boundaries[0] != 0:
        raise ContextAwareChunkingError(
            f"LLM boundaries must start at 0, got {payload.boundaries!r}"
        )

    return payload


def _hard_split_with_context(
    context_block: str,
    body: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[tuple[str, tuple[int, int] | None, bool, tuple[int, int] | None]]:
    """Split an oversized ``context_block + body`` chunk into sub-chunks.

    The context block is re-prepended to every resulting sub-chunk (it is
    small and self-contained, so duplicating it preserves the situating
    context each sub-chunk would otherwise lose). The remaining budget
    (``chunk_size`` minus the context block) is then hard-split the same
    way :func:`ingestion.chunker.group_paragraphs_into_chunks` hard-splits
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
        list[tuple[str, tuple[int, int] | None, bool, tuple[int, int] | None]]:
            One tuple per sub-chunk: ``(content, context_block_range,
            has_overlap, overlap_range)``.
    """
    results: list[tuple[str, tuple[int, int] | None, bool, tuple[int, int] | None]] = []
    context_len = len(context_block)
    # Final safety: context block is pre-validated by _sanitize_context_block,
    # but ensure at least 1 character of body budget so the sliding window
    # stride is strictly positive and can never stall.
    body_budget = max(chunk_size - context_len, 1)
    body_overlap = min(chunk_overlap, body_budget - 1) if body_budget > 1 else 0

    start = 0
    first = True
    while start < len(body) or first:
        body_slice = body[start : start + body_budget]
        content = context_block + body_slice

        context_range = (0, context_len) if context_len else None
        # The carried overlap is a prefix of this piece's body, located
        # directly after the prepended context block. Clamp the overlap length
        # to the actual slice length so the range never exceeds content.
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


def chunk_text_context_aware(
    filename: str,
    text: str,
    llm: LLMClient,
    semantic_boundaries: bool = True,
    contextualize: bool = True,
    chunk_size: int = default_chunk_size,
    chunk_overlap: int = default_chunk_overlap,
) -> list[ParsedChunk]:
    """Split text into chunks using an LLM for semantic boundaries and/or
    selective contextualization.

    This is an opt-in alternative to :func:`ingestion.chunker.chunk_text`.
    Falls back to it when:

    - both ``semantic_boundaries`` and ``contextualize`` are ``False``
      (nothing for the LLM to do),
    - ``text`` exceeds the configured LLM size limit (see
      :func:`exceeds_llm_limit`),
    - the LLM backend is unreachable, or
    - the LLM output cannot be parsed/validated.

    Args:
        filename (str):
            Name of the source file.

        text (str):
            Text content to split.

        llm (LLMClient):
            Client used to request the chunking plan.

        semantic_boundaries (bool, optional):
            Let the LLM choose chunk boundaries based on semantic coherence
            instead of ``chunk_text``'s character-length accumulation.
            Defaults to ``True``.

        contextualize (bool, optional):
            Let the LLM flag chunks that would benefit from a short
            situating context block and prepend it to their content.
            Defaults to ``True``.

        chunk_size (int, optional):
            Maximum chunk size in characters, including any prepended
            context block. Defaults to the value configured via
            ``CHUNK_SIZE``.

        chunk_overlap (int, optional):
            Overlap used when hard-splitting oversized chunks.
            Defaults to the value configured via ``CHUNK_OVERLAP``.

    Returns:
        list[ParsedChunk]:
            Context-aware text chunks with metadata, or the
            ``chunk_text`` fallback result.
    """
    if not semantic_boundaries and not contextualize:
        return chunk_text(filename, text, chunk_size, chunk_overlap)

    if exceeds_llm_limit(text):
        logger.warning(
            "Text for %s exceeds context-aware chunking size limit "
            "(%d chars) — falling back to chunk_text",
            filename,
            len(text),
        )
        return chunk_text(filename, text, chunk_size, chunk_overlap)

    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        return chunk_text(filename, text, chunk_size, chunk_overlap)

    segments = (
        paragraphs
        if semantic_boundaries
        else group_paragraphs_into_chunks(paragraphs, chunk_size, chunk_overlap)
    )

    try:
        payload = _request_chunking_plan(
            segments, llm, semantic_boundaries, contextualize
        )
    except (ContextAwareChunkingError, LLMUnavailableError) as exc:
        logger.warning(
            "Context-aware chunking failed for %s (%s) — falling back to chunk_text",
            filename,
            exc,
        )
        return chunk_text(filename, text, chunk_size, chunk_overlap)

    if semantic_boundaries:
        boundaries = payload.boundaries or [0]
        raw_chunks = [
            "\n\n".join(paragraphs[start:end])
            for start, end in zip(
                boundaries, [*boundaries[1:], len(paragraphs)], strict=True
            )
        ]
    else:
        raw_chunks = segments

    final_chunks: list[
        tuple[str, tuple[int, int] | None, bool, tuple[int, int] | None]
    ] = []
    for order_index, chunk_body in enumerate(raw_chunks):
        context_text = (
            payload.context_blocks.get(str(order_index)) if contextualize else None
        )
        # Validate and bound context block length before prefixing
        context_text = _sanitize_context_block(context_text, chunk_size)
        context_block = f"{context_text}\n\n" if context_text else ""
        combined = context_block + chunk_body

        if len(combined) <= chunk_size:
            context_range = (0, len(context_block)) if context_block else None
            final_chunks.append((combined, context_range, False, None))
            continue

        final_chunks.extend(
            _hard_split_with_context(
                context_block, chunk_body, chunk_size, chunk_overlap
            )
        )

    total_chunks_amount = len(final_chunks)

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
        ) in enumerate(final_chunks)
    ]
