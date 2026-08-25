from pydantic import BaseModel, field_validator

from agents.tools.base import Tool, ToolResult
from rag.filters import matches_retrieval_filters
from rag.source_filter import SourceExclusions, is_excluded
from rag.types import RetrievalFilters, ScoredChunk
from store.base import VectorStore

_GREP_SCORE = 1.0


class GrepArgs(BaseModel):
    patterns: list[str]

    @field_validator("patterns", mode="before")
    @classmethod
    def coerce_to_list(cls, v: object) -> object:
        return [v] if isinstance(v, str) else v


class GrepTool(Tool[GrepArgs]):
    name = "grep"
    description = (
        "Exact (case-insensitive) substring search for identifiers or string "
        "literals. Use when you know a function name, symbol or exact phrase."
    )
    args_model = GrepArgs

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

    def run(self, args: GrepArgs) -> ToolResult:
        needles = [p.lower() for p in args.patterns]
        results = [
            ScoredChunk(
                id=chunk.id,
                artifact_id=chunk.artifact_id,
                filename=chunk.filename,
                text=chunk.text,
                score=_GREP_SCORE,
                position=chunk.position,
                kind=chunk.kind,
                start_line=chunk.start_line,
                start_page=chunk.start_page,
                project_ids=chunk.project_ids,
            )
            # This tool scans the whole corpus in memory rather than going
            # through the vector store's query path, so the retrieval filters
            # (notably the project scope) have to be applied here explicitly.
            for chunk in self._store.iter_chunks_without_embeddings()
            if matches_retrieval_filters(chunk, self._filters)
            and not is_excluded(chunk, self._exclusions)
            and any(needle in chunk.text.lower() for needle in needles)
        ]
        return ToolResult(
            summary=f"grep({args.patterns}): {len(results)} match(es).",
            chunks=results,
        )
