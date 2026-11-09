import logging
from pathlib import Path

from ingestion.code_parser import parse_code
from ingestion.image_parser import parse_image
from ingestion.language_utils import TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES
from ingestion.models import ParsedChunk
from ingestion.pdf_parser import parse_pdf
from ingestion.text_parser import parse_text
from llm.base import LLMClient

logger = logging.getLogger(__name__)

CODE_EXTENSIONS = set(TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES.keys())

PDF_EXTENSION = {".pdf"}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}


def parse(
    filename: str,
    content: bytes,
    llm: LLMClient | None = None,
    semantic_boundaries: bool = False,
    contextualize: bool = False,
) -> list[ParsedChunk]:
    """Parse a file into structured chunks based on its file extension.

    The parser acts as a dispatcher and forwards the file content
    to the appropriate parser implementation.

    Supported file types:
    - Source code files (see ``CODE_EXTENSIONS``) -> parsed into code chunks
    - PDF files (.pdf) -> extracted text chunk
    - Image files (.png, .jpg, .jpeg, .gif, .webp, .bmp) -> image chunk
    - Any other extension (.txt, .json, .md, ...) -> single plain-text chunk

    There is no "unsupported" type: an unknown extension falls back to the
    plain-text parser rather than raising.

    ``llm``, ``semantic_boundaries``, and ``contextualize`` only affect the
    plain-text and PDF paths (see :func:`ingestion.text_parser.parse_text`
    and :func:`ingestion.pdf_parser.parse_pdf`); code and image parsing are
    unaffected. Passing no ``llm`` preserves the exact prior behavior.

    Args:
        filename (str):  Name of the uploaded file including extension.
        content (bytes): Raw file content as bytes.

        llm (LLMClient | None, optional):
            Client used for context-aware chunking of text/PDF content.
            When ``None`` (the default), plain paragraph/character-based
            chunking is used. Defaults to ``None``.

        semantic_boundaries (bool, optional):
            Let the LLM choose chunk boundaries based on semantic
            coherence. Only relevant when ``llm`` is provided. Defaults to
            ``False``.

        contextualize (bool, optional):
            Let the LLM prepend a short situating context block to chunks
            that would benefit from one. Only relevant when ``llm`` is
            provided. Defaults to ``False``.

    Returns:
        list[ParsedChunk]: A list of ParsedChunk objects extracted from the file.
    """

    # dispatcher organizes wich parser to use

    file_suffix = Path(filename).suffix

    if file_suffix in CODE_EXTENSIONS:
        return parse_code(filename, content)

    elif file_suffix in PDF_EXTENSION:
        return parse_pdf(
            filename,
            content,
            llm=llm,
            semantic_boundaries=semantic_boundaries,
            contextualize=contextualize,
        )

    elif file_suffix in IMAGE_EXTENSIONS:
        return parse_image(filename, content)

    else:
        return parse_text(
            filename,
            content,
            llm=llm,
            semantic_boundaries=semantic_boundaries,
            contextualize=contextualize,
        )
