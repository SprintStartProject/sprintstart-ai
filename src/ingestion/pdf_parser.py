import io
import logging

from pypdf import PdfReader

from ingestion.chunker import chunk_text
from ingestion.context_aware_chunker import chunk_text_context_aware
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
    (see :func:`ingestion.context_aware_chunker.chunk_text_context_aware`).

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

    try:
        reader = PdfReader(io.BytesIO(content))  # in memory processing
    except Exception as e:
        logger.error("Failed to read PDF %s: %s", filename, e)
        return []

    pdf_chunks: list[ParsedChunk] = []
    global_pdf_chunk_index: int = 0
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

        use_context_aware_chunking = semantic_boundaries or contextualize
        chunks: list[ParsedChunk] = (
            chunk_text(filename, text)
            if not use_context_aware_chunking or llm is None
            else chunk_text_context_aware(
                filename,
                text,
                llm,
                semantic_boundaries=semantic_boundaries,
                contextualize=contextualize,
            )
        )
        for chunk in chunks:
            chunk.kind = "pdf"
            chunk.metadata["global_pdf_chunk_index"] = str(global_pdf_chunk_index)
            chunk.metadata["page_number"] = str(page_number)
            pdf_chunks.append(chunk)
            global_pdf_chunk_index += 1

    return pdf_chunks
