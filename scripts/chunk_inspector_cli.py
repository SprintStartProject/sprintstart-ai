from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cli_colors import C
from rich.align import Align
from rich.console import Console
from rich.table import Table

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

from api.dependencies import get_llm  # noqa: E402
from ingestion.models import ParsedChunk  # noqa: E402
from ingestion.parser import parse  # noqa: E402

console = Console()


def main() -> int:
    logging.basicConfig(level=logging.WARNING)

    args = _parse_args()

    path = Path(args.filepath)

    if not path.exists():
        print(C.red(f"[error] file does not exist: {path}"))
        return 1

    try:
        filename, content = _load_file(path)
        use_llm = args.semantic_boundaries or args.contextualize
        llm = None
        if use_llm:
            try:
                llm = get_llm()
            except Exception as exc:
                print(
                    C.yellow(
                        f"[warn] LLM client unavailable ({exc}) — "
                        "falling back to deterministic chunker"
                    )
                )
                llm = None

        chunks = parse(
            filename,
            content,
            llm=llm,
            semantic_boundaries=args.semantic_boundaries,
            contextualize=args.contextualize,
        )

        if not chunks:
            print(C.yellow("[warn] no chunks generated"))
            return 0

        report = _analyze_chunks(chunks)

        _render(chunks, report, args)

    except Exception as exc:
        print(C.red(f"[error] failed to process file: {exc}"))
        return 1

    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect ingestion chunking behavior")

    parser.add_argument(
        "filepath",
        help="Path to file to inspect",
    )

    parser.add_argument(
        "--semantic-boundaries",
        action="store_true",
        help="Let LLM choose chunk boundaries based on semantic coherence",
    )

    parser.add_argument(
        "--contextualize",
        action="store_true",
        help="Let LLM prepend situating context blocks to chunks",
    )

    group = parser.add_mutually_exclusive_group()

    parser.add_argument(
        "--show-overlap",
        action="store_true",
        help="Show overlap between adjacent chunks",
    )

    group.add_argument(
        "--raw",
        action="store_true",
        help="Print only chunk content (no metadata)",
    )

    group.add_argument(
        "--json",
        action="store_true",
        help="Output chunks as JSON",
    )

    return parser.parse_args()


def _load_file(path: Path) -> tuple[str, bytes]:
    """
    Loads a file and returns:
    - filename (str)
    - raw content (bytes)
    """
    filename: str = path.name
    content: bytes = path.read_bytes()

    return filename, content


@dataclass
class ChunkReport:
    total: int
    avg_size: float
    min_size: int
    max_size: int


def _analyze_chunks(chunks: list[ParsedChunk]) -> ChunkReport:
    """
    Compute summary statistics for chunks.
    """

    if not chunks:
        return ChunkReport(
            total=0,
            avg_size=0.0,
            min_size=0,
            max_size=0,
        )

    total_chunks: int = len(chunks)
    min_size, max_size, total_size = len(chunks[0].content), 0, 0

    for chunk in chunks:
        chunk_size = len(chunk.content)

        if chunk_size < min_size:
            min_size = chunk_size

        if chunk_size > max_size:
            max_size = chunk_size

        total_size += chunk_size

    return ChunkReport(
        total=total_chunks,
        avg_size=total_size / total_chunks,
        min_size=min_size,
        max_size=max_size,
    )


def _render(
    chunks: list[ParsedChunk],
    report: ChunkReport,
    args: argparse.Namespace,
) -> None:
    """
    Handles all output modes:
    - pretty
    - raw
    - json
    """
    if args.raw:
        console.rule(title="Raw Chunks", characters="=")
        print_chunks(chunks, pretty=False, show_overlap=args.show_overlap)

    elif args.json:
        data = [chunk_to_dict(chunk) for chunk in chunks]
        print(json.dumps(data, indent=2, ensure_ascii=False))

    else:
        console.rule(title="Detailed Chunks", characters="=")
        print_chunks(chunks, pretty=True, show_overlap=args.show_overlap)

    console.rule(title="Overall Report", characters="=")
    report_table = Table(expand=False)
    report_table.add_column("Total Chunks", justify="center")
    report_table.add_column("Avg. Chunk Size", justify="center")
    report_table.add_column("Min Chunk Size", justify="center")
    report_table.add_column("Max Chunk Size", justify="center")
    report_table.add_row(
        f" {report.total}",
        f" {report.avg_size}",
        f" {report.min_size}",
        f" {report.max_size}",
    )

    console.print(Align.center(report_table))
    print("")


def print_chunks(
    chunks: list[ParsedChunk], pretty: bool = True, show_overlap: bool = False
) -> None:
    """Print all generated chunks to the terminal.

    Depending on the selected mode, the function prints chunk metadata,
    content, and optionally highlights overlap regions shared with
    adjacent chunks.

    Args:
        chunks (list[ParsedChunk]):
            Chunks produced by the ingestion pipeline.

        pretty (bool, optional):
            Whether to render chunk metadata tables in addition to the
            chunk content. Defaults to True.

        show_overlap (bool, optional):
            Whether overlap regions should be highlighted in the chunk
            content output. Defaults to False.
    """

    for chunk in chunks:
        print("")
        isPdf: bool = chunk.kind == "pdf"
        isCode: bool = chunk.kind == "code"
        if isPdf:
            console.rule(
                title=f"Chunk {chunk.metadata['chunk_index']} - Page {chunk.metadata['page_number']}",  # noqa: E501
                characters="=",
            )
        elif isCode:
            console.rule(
                title=f"Chunk {chunk.metadata['chunk_index']} - {chunk.metadata.get('symbol_kind', '-')}: {chunk.metadata.get('symbol_name', '-')}",  # noqa: E501
                characters="=",
            )
        else:
            console.rule(title=f"Chunk {chunk.metadata['chunk_index']}", characters="=")

        has_context = chunk.metadata.get("has_context_block") == "true"
        context_range_str = chunk.metadata.get("context_block_range", "")
        has_overlap = chunk.metadata.get("has_overlap") == "true"
        overlap_range_str = chunk.metadata.get("overlap_range", "")

        if pretty:
            metadata_table = Table(expand=False)
            metadata_table.add_column("chunk kind", justify="center")
            metadata_table.add_column("filename", justify="center")
            metadata_table.add_column("type", justify="center")
            if isPdf:
                metadata_table.add_column("global-index", justify="center")
            elif isCode:
                metadata_table.add_column("symbol-kind", justify="center")
                metadata_table.add_column("symbol-name", justify="center")
            metadata_table.add_column("chars", justify="center")
            metadata_table.add_column("context", justify="center")
            metadata_table.add_column("overlap", justify="center")
            row = [
                chunk.kind,
                chunk.metadata["filename"],
                chunk.metadata["type"],
            ]

            if isPdf:
                row.append(str(chunk.metadata.get("global_pdf_chunk_index", "-")))

            if isCode:
                row.append(chunk.metadata.get("symbol_kind", "-"))
                row.append(chunk.metadata.get("symbol_name", "-"))

            context_display = (
                chunk.metadata.get("context_block_range", "yes") if has_context else "-"
            )
            overlap_display = (
                chunk.metadata.get("overlap_range", "yes") if has_overlap else "-"
            )
            row.extend(
                [
                    str(len(chunk.content)),
                    context_display,
                    overlap_display,
                ]
            )

            metadata_table.add_row(*row)
            console.print(Align.center(metadata_table))
        print("")

        if has_context and context_range_str:
            ctx_start, ctx_end = map(int, context_range_str.split(":"))
            ctx_text = chunk.content[ctx_start:ctx_end]
            body_text = chunk.content[ctx_end:]
            print(C.cyan(f"[context] {ctx_text.strip()}"))
            print(C.bold("body:"))
            if show_overlap and has_overlap and overlap_range_str:
                ov_s, ov_e = map(int, overlap_range_str.split(":"))
                rel_s = max(0, ov_s - ctx_end)
                rel_e = max(0, ov_e - ctx_end)
                print(C.green(body_text[rel_s:rel_e]), end="")
                print(C.dim(body_text[rel_e:]))
            else:
                print(C.dim(body_text))
        else:
            print(C.bold("content:"))
            if show_overlap and has_overlap and overlap_range_str:
                ov_s, ov_e = map(int, overlap_range_str.split(":"))
                print(chunk.content[:ov_s], end="")
                print(C.green(chunk.content[ov_s:ov_e]), end="")
                print(C.dim(chunk.content[ov_e:]))
            else:
                print(C.dim(chunk.content))

        print("")


def chunk_to_dict(chunk: ParsedChunk) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "filename": chunk.metadata.get("filename", ""),
        "type": chunk.metadata.get("type", ""),
        "chunk_index": chunk.metadata.get("chunk_index", ""),
        "total_chunks": chunk.metadata.get("total_chunks", ""),
    }

    if "symbol_name" in chunk.metadata:
        metadata["symbol_name"] = chunk.metadata["symbol_name"]
    if "symbol_kind" in chunk.metadata:
        metadata["symbol_kind"] = chunk.metadata["symbol_kind"]

    if "page_number" in chunk.metadata:
        metadata["page_number"] = chunk.metadata["page_number"]
    if "global_pdf_chunk_index" in chunk.metadata:
        metadata["global_pdf_chunk_index"] = chunk.metadata["global_pdf_chunk_index"]

    if "has_context_block" in chunk.metadata:
        metadata["has_context_block"] = chunk.metadata["has_context_block"]
    if "context_block_range" in chunk.metadata:
        metadata["context_block_range"] = chunk.metadata["context_block_range"]
    if "has_overlap" in chunk.metadata:
        metadata["has_overlap"] = chunk.metadata["has_overlap"]
    if "overlap_range" in chunk.metadata:
        metadata["overlap_range"] = chunk.metadata["overlap_range"]

    return {
        "content": chunk.content,
        "kind": chunk.kind,
        "metadata": metadata,
    }


if __name__ == "__main__":
    raise SystemExit(main())
