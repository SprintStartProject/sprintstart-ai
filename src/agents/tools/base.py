from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ValidationError

from llm.base import ToolSpec
from rag.types import ScoredChunk

CapabilityKind = Literal["agent", "tool"]


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
    see ``agents.chat_agent._format_evidence``.
    """

    summary: str
    chunks: list[ScoredChunk] = field(default_factory=list[ScoredChunk])

    @classmethod
    def empty(cls, summary: str) -> "ToolResult":
        return cls(summary=summary, chunks=[])


class Tool[ArgsT: BaseModel](ABC):
    name: str
    description: str
    args_model: type[ArgsT]
    kind: CapabilityKind = "tool"

    def execute(self, raw_args: dict[str, object]) -> ToolResult:
        try:
            args = self.args_model.model_validate(raw_args)
        except ValidationError:
            return ToolResult.empty(f"Invalid arguments for tool {self.name!r}.")
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
            return ToolResult.empty(f"Unknown tool: {name!r}.")
        return tool.execute(raw_args)

    def specs(self) -> list[ToolSpec]:
        return [tool.tool_spec() for tool in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)
