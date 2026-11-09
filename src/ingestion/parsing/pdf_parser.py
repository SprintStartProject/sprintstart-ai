import io
import logging
from collections.abc import Callable, Iterator

from pypdf import PdfReader

from ingestion.chunking.chunker import chunk_text
from ingestion.chunking.context_aware_chunker import chunk_text_context_aware
from ingestion.models import ParsedChunk
from llm.base import LLMClient

logger = logging.getLogger(__name__)


def parse_pdf(
    filename: str,
    content: bytes,
    llm: LLMClient | None = None,
    semantic_boundaries: bool = False,
    contextualize: bool = False,
) -> list[ParsedChunk]:
    """
    Parse a PDF file into structured text chunks.

    Each PDF page is processed independently. Extracted text is split
    into fixed-size chunks using `chunk_text`,
    or, when ``llm`` is provided, using the LLM-based context-aware chunker
    (see :func:`ingestion.chunking.context_aware_chunker.chunk_text_context_aware`).

    Each resulting chunk is enriched with PDF-specific metadata.

    Behavior:
        - Each page is processed separately
        - Pages without extractable text are skipped with a warning
        - Each page is split into multiple chunks if needed
        - Each chunk is labeled with kind="pdf"
        - Page-level metadata is attached to each chunk

    Metadata added:
        - page_number: The 1-based index of the PDF page the chunk belongs to
        - chunk_index: Sequential index of the chunk within the current page
        - global_pdf_chunk_index: Sequential global index of the chunk across the
          entire PDF document

    Note on context-aware chunking scope: the LLM is invoked once **per
    page** (not once for the whole document), since pages are already
    processed independently here. This means semantic boundaries and
    context blocks are chosen with only a single page's text in view,
    not the full document — context blocks are therefore page-scoped
    rather than document-scoped. Widening this to whole-document context
    (e.g. by threading a document-level summary into each page's call) is
    a possible future refinement, not implemented here.

    Args:
        filename (str):
            Name of the source PDF file.

        content (bytes):
            Raw binary content of the uploaded PDF file.

        llm (LLMClient | None, optional):
            Client used for context-aware chunking. When ``None`` (the
            default), plain paragraph/character-based chunking is used for
            every page. Defaults to ``None``.

        semantic_boundaries (bool, optional):
            Let the LLM choose chunk boundaries per page based on semantic
            coherence. Only relevant when ``llm`` is provided. Defaults to
            ``False``.

        contextualize (bool, optional):
            Let the LLM prepend a short situating context block to chunks
            that would benefit from one. Only relevant when ``llm`` is
            provided. Defaults to ``False``.

    Returns:
        list[ParsedChunk]:
            A flat list of all extracted and chunked PDF content,
            enriched with page and chunk metadata.
    """

    chunk_page = _page_chunker(llm, semantic_boundaries, contextualize)
    pdf_chunks: list[ParsedChunk] = []
    for page_number, text in _page_texts(filename, content):
        for chunk in chunk_page(filename, text):
            chunk.kind = "pdf"
            chunk.metadata["global_pdf_chunk_index"] = str(len(pdf_chunks))
            chunk.metadata["page_number"] = str(page_number)
            pdf_chunks.append(chunk)

    return pdf_chunks


def _page_texts(filename: str, content: bytes) -> Iterator[tuple[int, str]]:
    """Yield ``(page_number, text)`` for every page with extractable text.

    An unreadable PDF yields nothing; a page whose text cannot be extracted is skipped.
    """
    try:
        reader = PdfReader(io.BytesIO(content))  # in memory processing
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", filename, e)
        return

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text()
        except Exception as e:
            logger.warning(
                "Failed to extract text from page %s in %s: %s",
                page_number,
                filename,
                e,
            )
            continue

        if not text or not text.strip():
            logger.warning("Skipping empty page %s in %s", page_number, filename)
            continue

        yield page_number, text


def _page_chunker(
    llm: LLMClient | None, semantic_boundaries: bool, contextualize: bool
) -> Callable[[str, str], list[ParsedChunk]]:
    """Pick the chunker for a page: the LLM one only when it has work to do."""
    if llm is None or not (semantic_boundaries or contextualize):
        return chunk_text
    client = llm
    return lambda filename, text: chunk_text_context_aware(
        filename,
        text,
        client,
        semantic_boundaries=semantic_boundaries,
        contextualize=contextualize,
    )
