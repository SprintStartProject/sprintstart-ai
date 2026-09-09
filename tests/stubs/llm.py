from collections.abc import Callable, Iterator, Mapping, Sequence

from llm.base import (
    ChatResult,
    LLMChatStreamEvent,
    LLMStreamEvent,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolSpec,
)

Turn = Sequence[tuple[str, Mapping[str, object]]]


class StubLLMClient:
    def __init__(
        self,
        generate_response: str = "stub answer",
        embedding: list[float] | None = None,
        caption: str = "stub caption",
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self.generate_response = generate_response
        self.embedding = embedding or [0.0] * 768
        self.caption = caption
        self.embed_fn = embed_fn

    @property
    def model_name(self) -> str | None:
        return "stub-model"

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        return ChatResult(text=self.generate_response, tool_calls=[])

    def chat_stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Iterator[LLMChatStreamEvent]:
        # Buffer parity: the stub's chat result is constant, so replaying it as
        # one terminal event keeps every inherited-by-default stub usable as a
        # streaming client without each test overriding the method. The text
        # delta mirrors a real provider streaming its answer during the turn;
        # the terminal ChatResult.text repeats it by contract.
        yield TextDelta(self.generate_response)
        yield self.chat(messages, tools)

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        return self.generate_response

    def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
        yield TextDelta(self.generate_response)

    def embed(self, text: str) -> list[float]:
        if self.embed_fn is not None:
            return self.embed_fn(text)
        return self.embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    def caption_image(self, image_bytes: bytes) -> str:
        return self.caption


class ScriptedLLMClient:
    """
    Drives the tool-calling loop deterministically.

    Each `chat` call pops the next scripted turn and returns those tool calls. Once
    the script is exhausted it returns no tool calls (the agent stops gathering).
    `stream` always yields the fixed answer.
    """

    def __init__(
        self,
        turns: Sequence[Turn],
        *,
        answer: str = "final answer",
        embedding: list[float] | None = None,
        stream_events: Sequence[LLMStreamEvent] | None = None,
        reasoning: str | None = None,
        reasoning_details: list[dict[str, object]] | None = None,
    ) -> None:
        self._turns: list[Turn] = list(turns)
        self.answer = answer
        self.embedding = embedding or [0.0] * 768
        self._stream_events = list(stream_events) if stream_events is not None else None
        self.reasoning = reasoning
        self.reasoning_details = reasoning_details or []
        self.chat_calls: list[list[Message]] = []
        self.stream_calls: list[list[Message]] = []
        self.chat_stream_calls: list[list[Message]] = []
        # Optional hook: tests exercising the streaming decision path assign a
        # callable returning the event sequence for a turn. When unset,
        # chat_stream replays the scripted chat() turn as a single terminal
        # event, so legacy scripted tests keep passing unchanged.
        self.stream_turns: (
            Callable[[list[Message]], Sequence[LLMChatStreamEvent]] | None
        ) = None

    @property
    def model_name(self) -> str | None:
        return "scripted-model"

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        self.chat_calls.append(messages)
        turn: Turn = self._turns.pop(0) if self._turns else []
        calls = [
            ToolCall(id=f"call_{i}", name=name, arguments=dict(args))
            for i, (name, args) in enumerate(turn)
        ]
        return ChatResult(
            text="" if calls else self.answer,
            tool_calls=calls,
            reasoning=self.reasoning if calls else None,
            reasoning_details=self.reasoning_details if calls else [],
        )

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        return self.answer

    def chat_stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Iterator[LLMChatStreamEvent]:
        # Recording parity with chat(): the agent moved its decision call onto
        # this method, so legacy assertions on chat_calls must keep counting
        # decision turns made via the streaming path.
        self.chat_calls.append(messages)
        self.chat_stream_calls.append(messages)
        if self.stream_turns is not None:
            yield from self.stream_turns(messages)
            return
        turn: Turn = self._turns.pop(0) if self._turns else []
        calls = [
            ToolCall(id=f"call_{i}", name=name, arguments=dict(args))
            for i, (name, args) in enumerate(turn)
        ]
        if calls and self.reasoning:
            yield ReasoningDelta(self.reasoning)
        elif not calls:
            # A real streaming provider emits answer deltas during the turn;
            # the terminal ChatResult.text mirrors them, never replaces them.
            yield TextDelta(self.answer)
        yield ChatResult(
            text="" if calls else self.answer,
            tool_calls=calls,
            reasoning=self.reasoning if calls else None,
            reasoning_details=self.reasoning_details if calls else [],
        )

    def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
        self.stream_calls.append(messages)
        yield from self._stream_events or [TextDelta(self.answer)]

    def embed(self, text: str) -> list[float]:
        return self.embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embedding for _ in texts]

    def caption_image(self, image_bytes: bytes) -> str:
        return "stub caption"
