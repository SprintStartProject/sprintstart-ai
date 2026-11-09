"""What a Buddy turn yields: the stream events and the turn's outcome."""

from collections.abc import Iterator
from dataclasses import dataclass, field

from corpus.types import Citation
from llm.base import (
    Message,
    ToolCall,
)


@dataclass
class AgentTurnResult:
    """The outcome of one agent turn.

    ``final`` distinguishes "here is the answer" (``text`` is it) from "I need the
    backend to run these tools first" (``pending_tool_calls`` are them). ``messages``
    is always the full running conversation the caller must carry back verbatim next
    turn -- it already includes any search steps run here and the tool-use turn the
    pending calls belong to.
    """

    final: bool
    text: str
    messages: list[Message]
    pending_tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    citations: list[Citation] = field(default_factory=list[Citation])
    #: The model's reasoning, one entry per call that returned any. Display-only:
    #: it is never written into ``messages`` as text the caller carries back.
    reasoning: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class AgentReasoning:
    """A fragment of the model's reasoning, as it arrives."""

    text: str


@dataclass(frozen=True)
class AgentToken:
    """A fragment of the answer text, as it arrives."""

    text: str


@dataclass(frozen=True)
class AgentToolUse:
    """A search this loop is about to run itself (never a backend tool)."""

    name: str
    arguments: dict[str, object]


type AgentStreamEvent = AgentReasoning | AgentToken | AgentToolUse | AgentTurnResult

# Between the text of one hop and the text of the next. A hop that wrote something and
# then searched leaves its text without a trailing break, and the next hop would run
# straight on from it ("...let me check.The deploy runs on...").
_HOP_SEPARATOR = "\n\n"


class HopText:
    """Keeps the answer text of successive hops readable as one stream."""

    def __init__(self) -> None:
        self._emitted = False
        self._owed = False

    def begin_hop(self) -> None:
        self._owed = self._emitted

    def token(self, text: str) -> Iterator[AgentToken]:
        if not text:
            return
        if self._owed:
            self._owed = False
            yield AgentToken(_HOP_SEPARATOR)
        self._emitted = True
        yield AgentToken(text)
