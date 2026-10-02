from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from llm.base import ToolCall, ToolSpec
from rag.types import ScoredChunk

CapabilityKind = Literal["agent", "tool"]

# Tool calls asked for in the same step run together, up to this many. The cap
# is low because each `retrieve` already forks a pair of threads internally
# (`rag.hybrid`), so the real thread count is about double this.
MAX_PARALLEL_TOOLS = 4


@dataclass(frozen=True)
class Invocation:
    kind: CapabilityKind
    name: str


@dataclass(frozen=True)
class ToolResult:
    """What a tool found.

    ``summary`` is the one-line fallback shown to the model when there is
    nothing to show — an error, or a search that matched nothing. When
    ``chunks`` is non-empty the model is given the chunks themselves instead;
    see ``agents.tools.evidence.format_evidence``.
    """

    summary: str
    chunks: list[ScoredChunk] = field(default_factory=list[ScoredChunk])
    match_chunk_ids: frozenset[str] | None = None
    #: True for a call that never ran -- an unknown tool, or arguments that did
    #: not validate. ``summary`` is then the error itself, which every caller
    #: has to show: a model told "no matches" retries the search, a model told
    #: its arguments were wrong fixes them.
    is_error: bool = False

    @property
    def match_count(self) -> int:
        """Return the number of direct matches, excluding context-only chunks."""
        if self.match_chunk_ids is None:
            return len(self.chunks)
        return len(self.match_chunk_ids)

    def matched_chunks(
        self, chunks: list[ScoredChunk] | None = None
    ) -> list[ScoredChunk]:
        """Return direct matches from *chunks* while preserving their order.

        Tools that do not distinguish direct matches from context retain the
        legacy behavior where every returned chunk is considered a match.
        """
        candidates = self.chunks if chunks is None else chunks
        if self.match_chunk_ids is None:
            return candidates
        return [chunk for chunk in candidates if chunk.id in self.match_chunk_ids]

    @classmethod
    def empty(cls, summary: str) -> "ToolResult":
        """A call that ran and had nothing to report; ``summary`` says what."""
        return cls(summary=summary, chunks=[])

    @classmethod
    def failed(cls, summary: str) -> "ToolResult":
        """A call that never ran; ``summary`` is the error the model must see."""
        return cls(summary=summary, chunks=[], is_error=True)


class Tool[ArgsT: BaseModel](ABC):
    name: str
    description: str
    args_model: type[ArgsT]
    kind: CapabilityKind = "tool"

    def execute(self, raw_args: dict[str, object]) -> ToolResult:
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError:
            return ToolResult.failed(f"Invalid arguments for tool {self.name!r}.")
        return self.run(args)

    def tool_spec(self) -> ToolSpec:
        return ToolSpec(
            name=self.name,
            description=self.description,
            parameters=self.args_model.model_json_schema(),
        )

    @abstractmethod
    def run(self, args: ArgsT) -> ToolResult: ...


AnyTool = Tool[Any]


class ToolRegistry:
    def __init__(self, tools: Iterator[AnyTool] | list[AnyTool]) -> None:
        self._tools: dict[str, AnyTool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: AnyTool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name!r}")
        self._tools[tool.name] = tool

    def get(self, name: str) -> AnyTool | None:
        return self._tools.get(name)

    def names(self) -> frozenset[str]:
        return frozenset(self._tools)

    def execute(self, name: str, raw_args: dict[str, object]) -> ToolResult:
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult.failed(f"Unknown tool: {name!r}.")
        return tool.execute(raw_args)

    def specs(self) -> list[ToolSpec]:
        return [tool.tool_spec() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)


def run_tool_calls(
    registry: ToolRegistry, calls: Sequence[ToolCall]
) -> list[ToolResult]:
    """Run a step's tool calls, together when there is more than one.

    Shared by every agent loop (the chat agent and the onboarding buddy), so
    the two cannot drift on how a step's calls are executed. Both prompts ask
    the model to request every search it needs in one step, which only pays off
    if the calls overlap: each search is a network round-trip (an embedding
    call) plus a corpus scan, so running three serially costs three times what
    running them together does. The cap lives here for the same reason -- one
    constant, one rationale.

    Safe because the tools only read -- the process-wide BM25 index guards its
    rebuild with a lock (``rag.hybrid.BM25IndexCache``), the store's queries
    are reads, and the SDK HTTP clients are thread-safe. Results come back in
    call order (``map``), so what the model and the caller see never depends on
    which search happened to finish first.
    """

    def run_one(call: ToolCall) -> ToolResult:
        return registry.execute(call.name, call.arguments)

    if len(calls) == 1:
        return [run_one(calls[0])]
    workers = min(len(calls), MAX_PARALLEL_TOOLS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, calls))
