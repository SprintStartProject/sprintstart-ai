import re

from pydantic import BaseModel

from agents.tools.base import Tool, ToolResult
from rag.filters import matches_retrieval_filters
from rag.source_filter import SourceExclusions, is_excluded
from rag.types import RetrievalFilters, ScoredChunk
from store.base import VectorStore

_FETCH_SCORE = 0.9
_EXTENSION_RE = re.compile(r"\.[a-z0-9]+$")


def _stem(filename: str) -> str:
    return _EXTENSION_RE.sub("", filename.lower())


def _matches(filename: str, query: str, query_has_ext: bool) -> bool:
    name = filename.lower()
    if name == query:
        return True
    return not query_has_ext and _stem(name) == query


class FetchFileArgs(BaseModel):
    filename: str


class FetchFileTool(Tool[FetchFileArgs]):
    name = "fetch_file"
    description = (
        "Return all indexed chunks from a specific file. "
        "Use when the relevant filename is already known."
    )
    args_model = FetchFileArgs

    def __init__(
        self,
        store: VectorStore,
        *,
        exclusions: SourceExclusions = SourceExclusions(),
        filters: RetrievalFilters | None = None,
    ) -> None:
        self._store = store
        self._exclusions = exclusions
        self._filters = filters

    def run(self, args: FetchFileArgs) -> ToolResult:
        query = args.filename.strip().lower()
        query_has_ext = bool(_EXTENSION_RE.search(query))
        results = [
            ScoredChunk(
                id=chunk.id,
                artifact_id=chunk.artifact_id,
                filename=chunk.filename,
                text=chunk.text,
                score=_FETCH_SCORE,
                position=chunk.position,
                kind=chunk.kind,
                start_line=chunk.start_line,
                start_page=chunk.start_page,
                project_ids=chunk.project_ids,
            )
            # Same as GrepTool: a whole-corpus in-memory scan, so the project
            # scope has to be enforced here rather than in the store's query.
            for chunk in self._store.all_chunks_without_embeddings()
            if matches_retrieval_filters(chunk, self._filters)
            and not is_excluded(chunk, self._exclusions)
            and _matches(chunk.filename, query, query_has_ext)
        ]
        return ToolResult(
            summary=f"fetch_file({args.filename!r}): {len(results)} chunk(s).",
            chunks=results,
        )
