from ingestion.chunking.chunker import chunk_text
from ingestion.chunking.context_aware_chunker import chunk_text_context_aware
from ingestion.models import ParsedChunk
from llm.base import LLMClient


def parse_text(
    filename: str,
    content: bytes,
    llm: LLMClient | None = None,
    semantic_boundaries: bool = False,
    contextualize: bool = False,
) -> list[ParsedChunk]:
    """Parse plain text-based files into paragraph-aware text chunks.

    When ``llm`` is provided, the LLM-based context-aware chunking strategy
    (see :func:`ingestion.chunking.context_aware_chunker.chunk_text_context_aware`)
    is used instead of the plain :func:`ingestion.chunking.chunker.chunk_text`. This
    is opt-in: without an ``llm``, behavior is unchanged regardless of the
    ``semantic_boundaries``/``contextualize`` flags, since there is no LLM
    to call.

    Args:
        filename (str): Name of the source file.
        content (bytes): Raw file content as bytes.

        llm (LLMClient | None, optional):
            Client used for context-aware chunking. When ``None`` (the
            default), plain paragraph/character-based chunking is used.
            Defaults to ``None``.

        semantic_boundaries (bool, optional):
            Let the LLM choose chunk boundaries based on semantic
            coherence. Only relevant when ``llm`` is provided. Defaults to
            ``False``.

        contextualize (bool, optional):
            Let the LLM prepend a short situating context block to chunks
            that would benefit from one. Only relevant when ``llm`` is
            provided. Defaults to ``False``.

    Returns:
        list[ParsedChunk]: A list of ParsedChunk objects, each with kind='text'.
    """
    text = content.decode(encoding="utf-8", errors="replace")

    use_context_aware_chunking = semantic_boundaries or contextualize
    if not use_context_aware_chunking or llm is None:
        return chunk_text(filename, text)

    return chunk_text_context_aware(
        filename,
        text,
        llm,
        semantic_boundaries=semantic_boundaries,
        contextualize=contextualize,
    )
