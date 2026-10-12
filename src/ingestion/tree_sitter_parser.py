import threading
from dataclasses import dataclass
from pathlib import Path

from tree_sitter_language_pack import (
    Node,
    Parser,
    Tree,
    detect_language_from_extension,
    get_parser,
)

from ingestion.language_utils import TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES


@dataclass
class CodeSymbol:
    name: str | None
    kind: str
    content: str
    start_byte: int
    end_byte: int


# tree-sitter Parser objects are unsendable (pyo3): a parser may only be used on
# the thread that created it. Cache per-thread so parsers created on a FastAPI
# threadpool worker are never reused on another thread.
_thread_local_parsers = threading.local()


def get_cached_parser(language: str) -> Parser:
    cache: dict[str, Parser] = getattr(_thread_local_parsers, "parsers", None) or {}
    parser = cache.get(language)
    if parser is None:
        parser = get_parser(language)
        cache[language] = parser
        _thread_local_parsers.parsers = cache
    return parser


def parse_with_tree_sitter(
    filename: str, content: bytes
) -> tuple[list[CodeSymbol], str]:
    """
    Parse source code using tree-sitter.

    Extracts top-level symbols and the file preamble.

    Args:
        filename:
            Source filename including extension.

        content:
            Raw file bytes.

    Returns:
        Tuple consisting of:

        - list[CodeSymbol]
        - preamble string

    Raises:
        UnsupportedLanguageError:
            No supported grammar exists.

        ParseTreeError:
            Tree-sitter parsing failed.
    """
    symbols: list[CodeSymbol] = []
    preamble_parts: list[str] = []
    seen_symbol = False

    extension: str = Path(filename).suffix.lower()

    if extension not in TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES.keys():
        raise UnsupportedLanguageError(
            f"No tree-sitter support for extension '{extension}'."
        )
    normalized_extension: str = extension.lstrip(".")

    code_language: str | None = detect_language_from_extension(normalized_extension)

    if code_language is None:
        raise UnsupportedLanguageError(
            f"Could not detect language for '{normalized_extension}'."
        )
    try:
        parser: Parser = get_cached_parser(code_language)
        tree: Tree | None = parser.parse_bytes(content)
    except Exception as exc:
        raise ParseTreeError(f"Failed creating parse tree for '{filename}'.") from exc

    if tree is None:
        raise ParseTreeError(f"Tree-sitter returned no parse tree for '{filename}'.")
    root_node: Node = tree.root_node()

    for i in range(root_node.child_count()):
        child_node = root_node.child(i)
        if child_node is None:
            continue
        start = child_node.start_byte()
        end = child_node.end_byte()
        if child_node.kind() in TOP_LEVEL_NODES_FOR_SUPPORTED_LANGUAGES[extension]:
            seen_symbol = True
            symbols.append(
                CodeSymbol(
                    _extract_name(child_node, content),
                    child_node.kind(),
                    _slice(content, start, end),
                    start,
                    end,
                )
            )
        elif not seen_symbol:
            preamble_parts.append(_slice(content, start, end))
        else:
            # Top-level statements that appear *after* the first symbol (module-level
            # code, re-exports, ``if __name__ == "__main__"`` guards) are not symbols
            # but must stay searchable, so emit them as module kind
            trailing: str = _slice(content, start, end).strip()
            if trailing:
                symbols.append(
                    CodeSymbol(
                        None,
                        "module",
                        trailing,
                        start,
                        end,
                    )
                )

    preamble = "\n".join(preamble_parts).strip()
    return symbols, preamble


def _slice(content: bytes, start_byte: int, end_byte: int) -> str:
    """Decode a tree-sitter byte span.

    tree-sitter reports ``start_byte``/``end_byte`` as offsets into the raw
    bytes, so the span must be sliced from ``content`` (bytes) and decoded —
    slicing a decoded ``str`` with byte offsets corrupts any non-ASCII source.
    """
    return content[start_byte:end_byte].decode("utf-8", errors="replace")


def _extract_name(
    node: Node | None,
    content: bytes,
) -> str | None:
    """
    Extract the first identifier found in a subtree.
    """

    if node is None:
        return None

    if node.kind() == "identifier":
        return _slice(content, node.start_byte(), node.end_byte())

    for i in range(node.child_count()):
        child = node.child(i)

        result = _extract_name(child, content)

        if result is not None:
            return result

    return None


class TreeSitterParserError(Exception):
    """Base exception for tree-sitter parsing."""


class UnsupportedLanguageError(TreeSitterParserError):
    """Raised when no tree-sitter grammar exists."""


class ParseTreeError(TreeSitterParserError):
    """Raised when tree-sitter parsing failed."""
