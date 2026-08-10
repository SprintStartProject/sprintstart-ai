from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import NotRequired, Protocol, TypedDict


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: dict[str, object]


class Message(TypedDict):
    role: str
    content: str
    tool_calls: NotRequired[list[ToolCall]]
    tool_call_id: NotRequired[str]
    name: NotRequired[str]


class ToolSpec(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


@dataclass(frozen=True)
class ChatResult:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])


class LLMClient(Protocol):
    @property
    def model_name(self) -> str | None:
        """Identifier of the chat model, for provenance/audit; ``None`` if N/A."""
        ...

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult: ...
    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str: ...
    def stream(self, messages: list[Message]) -> Iterator[str]: ...
    def embed(self, text: str) -> list[float]: ...
    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...
    def caption_image(self, image_bytes: bytes) -> str: ...
