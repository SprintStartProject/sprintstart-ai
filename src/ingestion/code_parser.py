import logging
import re
from pathlib import Path

from ingestion.chunker import chunk_code
from ingestion.language_utils import PATTERNS
from ingestion.models import ParsedChunk
from ingestion.text_parser import parse_text
from ingestion.tree_sitter_parser import parse_with_tree_sitter

logger = logging.getLogger(__name__)


def parse_code(filename: str, content: bytes) -> list[ParsedChunk]:
    """Parse source code into semantic code chunks.

    The parser first attempts AST-based parsing via tree-sitter to
    extract top-level symbols (e.g. functions, classes, interfaces)
    together with the file preamble such as imports or module headers.

    Each extracted symbol is converted into one or more code chunks when surpassing the
    max chunk size.
    Symbol metadata is attached to every chunk:

    - symbol_name (e.g. the function name)
    - symbol_kind (e.g. function_definition)

    If tree-sitter parsing fails, the parser falls back to the
    legacy regex-based parser. If no symbols can be identified,
    the file is parsed as plain text.

    Args:
        filename (str):
            Name of the source file including extension.

        content (bytes):
            Raw file content as bytes.

    Returns:
        list[ParsedChunk]:
            Parsed code chunks enriched with symbol metadata when
            tree-sitter extraction succeeds. Falls back to regex-
            based code chunks or text chunks when necessary.
    """

    try:
        symbols, preamble = parse_with_tree_sitter(filename, content)
    except Exception as exc:
        logger.warning(
            "Tree-sitter parsing failed for %s. Falling back to regex parser.",
            filename,
            exc_info=exc,
        )
        return fallback_regex_parser(filename, content)

    # Tree-sitter ran cleanly but found no top-level symbols
    if not symbols:
        return parse_text(filename, content)

    chunks: list[ParsedChunk] = []

    for symbol in symbols:
        full_content = f"{preamble}\n\n{symbol.content}" if preamble else symbol.content
        # Turn the symbol's byte offset into a 1-based line: count the
        # newlines that come before it in the original bytes (+1 for 1-based).
        # This is the source line citations should point at — the definition
        # itself, not the preamble we prepend above nor a split's own offset.
        symbol_start_line = content.count(b"\n", 0, symbol.start_byte) + 1

        code_chunks = chunk_code(filename, full_content, start_line=symbol_start_line)

        for chunk in code_chunks:
            chunk.metadata = {
                **chunk.metadata,
                "symbol_name": symbol.name or "",
                "symbol_kind": symbol.kind,
            }

        chunks.extend(code_chunks)

    return _reindex_chunks(chunks)


def _reindex_chunks(chunks: list[ParsedChunk]) -> list[ParsedChunk]:
    """Re-number ``chunk_index`` globally across a file's chunks.

    Each symbol is chunked independently, so the per-symbol ``chunk_index``
    restarts at 0. The deterministic chunk ID is derived from content + position,
    so two identical-content chunks at index 0 (e.g. duplicated boilerplate
    definitions) would collide and silently overwrite one another on upsert.
    Re-indexing across all chunks keeps every chunk's position unique.
    """
    total = len(chunks)
    return [
        ParsedChunk(
            content=chunk.content,
            kind=chunk.kind,
            metadata={
                **chunk.metadata,
                "chunk_index": str(i),
                "total_chunks": str(total),
            },
        )
        for i, chunk in enumerate(chunks)
    ]


def fallback_regex_parser(filename: str, content: bytes) -> list[ParsedChunk]:
    """Parse source code files into code-aware chunks.

    The parser detects top-level definitions such as functions and
    classes by matching with a regex pattern and creates self-contained chunks
    by prepending the file
    preamble (e.g. imports or module headers).

    If the language is unsupported or no top-level definitions are
    found, the parser falls back to the plain text parser.

    Args:
        filename (str):
            Name of the source file including extension.

        content (bytes):
            Raw file content as bytes.

    Returns:
        list[ParsedChunk]:
            Parsed code chunks with kind="code" or text chunks when
            fallback parsing is used.
    """

    file_suffix = Path(filename).suffix
    # Falls back to the text chunker if language is not supported
    if file_suffix not in PATTERNS.keys():
        return parse_text(filename, content)

    pattern: re.Pattern[str] = PATTERNS[file_suffix]
    lines = content.decode(encoding="utf-8", errors="replace").splitlines()

    boundaries: list[int] = compute_boundaries(lines, pattern)

    if not boundaries:
        # Falls back to the text chunker if no top level content could be found
        return parse_text(filename, content)

    preamble_end: int = boundaries[0]
    preamble: str = "\n".join(lines[:preamble_end]).strip()

    parsed_chunks: list[ParsedChunk] = []
    for start_line, end_line in zip(
        boundaries, boundaries[1:] + [len(lines)], strict=False
    ):
        content_body: str = "\n".join(lines[start_line:end_line]).strip()

        if not content_body:
            continue

        chunk_content: str = (
            f"{preamble}\n\n{content_body}" if preamble else content_body
        )
        # `start_line` is 0-based (a list index); citations use 1-based lines.
        for chunk in chunk_code(filename, chunk_content, start_line=start_line + 1):
            parsed_chunks.append(chunk)

    return _reindex_chunks(parsed_chunks)


def compute_boundaries(lines: list[str], pattern: re.Pattern[str]) -> list[int]:
    """Compute top-level definition boundaries in source code.

    The function scans source code line-by-line and returns the
    indices of lines that start new top-level definitions.

    Only definitions without indentation are considered to avoid
    matching nested functions or classes.

    Args:
        lines (list[str]):
            Source code split into lines.

        pattern (re.Pattern[str]):
            Language-specific regex used to detect definitions.

    Returns:
        list[int]:
            Line indices representing the start of top-level
            definitions.
    """
    boundaries: list[int] = []
    for line_count, line in enumerate(lines):
        # filter for top level content without indentation
        if pattern.match(line) and line == line.lstrip():
            boundaries.append(line_count)
    return boundaries
