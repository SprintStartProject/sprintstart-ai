from pydantic import BaseModel

from agents.tools.base import Tool, ToolResult
from llm.base import LLMClient
from rag.retriever import retrieve
from rag.source_filter import SourceExclusions
from rag.types import RetrievalFilters
from store.base import VectorStore

_DEFAULT_TOP_K = 5
_DEFAULT_MIN_SCORE = 0.3


class RetrieveArgs(BaseModel):
    query: str


class RetrieveTool(Tool[RetrieveArgs]):
    name = "retrieve"
    description = (
        "Semantic + keyword search over the knowledge base. "
        "Use for conceptual or open-ended questions."
    )
    args_model = RetrieveArgs

    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        *,
        top_k: int = _DEFAULT_TOP_K,
        min_score: float = _DEFAULT_MIN_SCORE,
        exclusions: SourceExclusions = SourceExclusions(),
        filters: RetrievalFilters | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._top_k = top_k
        self._min_score = min_score
        self._exclusions = exclusions
        self._filters = filters

    def run(self, args: RetrieveArgs) -> ToolResult:
        chunks = retrieve(
            args.query,
            self._llm,
            self._store,
            self._top_k,
            self._min_score,
            self._exclusions,
            self._filters,
        )
        return ToolResult(
            summary=f"retrieve({args.query!r}): {len(chunks)} chunk(s).",
            chunks=chunks,
        )
