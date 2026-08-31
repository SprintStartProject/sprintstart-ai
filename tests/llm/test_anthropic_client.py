from types import SimpleNamespace
from typing import Any, cast

import pytest

from llm.anthropic_client import (
    AnthropicClient,
    _to_anthropic_messages,  # pyright: ignore[reportPrivateUsage]
)
from llm.base import Message, ReasoningDelta, TextDelta, ToolCall

_EPHEMERAL = {"type": "ephemeral"}


class _FakeStream:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events

    def __enter__(self) -> "_FakeStream":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._events)


class _FakeMessages:
    def __init__(self, events: list[SimpleNamespace]) -> None:
        self._events = events
        self.stream_kwargs: dict[str, object] = {}

    def stream(self, **kwargs: object) -> _FakeStream:
        self.stream_kwargs = kwargs
        return _FakeStream(self._events)


def _delta(delta_type: str, **values: object) -> SimpleNamespace:
    return SimpleNamespace(
        type="content_block_delta",
        delta=SimpleNamespace(type=delta_type, **values),
    )


def test_stream_enables_thinking_and_separates_reasoning_from_text() -> None:
    client = AnthropicClient(
        api_key="test",
        chat_model="claude-haiku-4-5",
        thinking_budget_tokens=1024,
    )
    messages = _FakeMessages(
        [
            _delta("thinking_delta", thinking="Checking evidence."),
            _delta("thinking_delta", thinking=""),
            _delta("text_delta", text="Answer."),
        ]
    )
    client.client.messages = cast("Any", messages)

    events = list(client.stream([Message(role="user", content="Hello")]))

    assert events == [ReasoningDelta("Checking evidence."), TextDelta("Answer.")]
    assert messages.stream_kwargs["thinking"] == {
        "type": "enabled",
        "budget_tokens": 1024,
    }


def test_stream_defaults_to_thinking_disabled() -> None:
    from anthropic import Omit

    client = AnthropicClient(
        api_key="test",
        chat_model="claude-haiku-4-5",
    )
    messages = _FakeMessages([_delta("text_delta", text="Answer.")])
    client.client.messages = cast("Any", messages)

    assert list(client.stream([Message(role="user", content="Hello")])) == [
        TextDelta("Answer.")
    ]
    assert isinstance(messages.stream_kwargs["thinking"], Omit)


@pytest.mark.parametrize("budget", [1, 1023, 4096])
def test_invalid_thinking_budget_is_rejected(budget: int) -> None:
    with pytest.raises(ValueError):
        AnthropicClient(
            api_key="test",
            chat_model="claude-haiku-4-5",
            max_tokens=4096,
            thinking_budget_tokens=budget,
        )


def _blocks(content: object) -> list[dict[str, Any]]:
    """The content blocks of a converted message, as plain dicts.

    The SDK's params are ``TypedDict``s whose optional keys can't be read
    without a guard, which is noise in assertions about exactly those keys.
    """
    assert isinstance(content, list)
    return cast("list[dict[str, Any]]", content)


def _system_blocks(system: object) -> list[dict[str, Any]]:
    assert isinstance(system, list)
    return cast("list[dict[str, Any]]", system)


def test_system_prompt_carries_a_cache_breakpoint() -> None:
    """Tools render before the system prompt, so one breakpoint on the system
    block covers the whole fixed preamble every chat request shares."""
    system, _ = _to_anthropic_messages(
        [
            Message(role="system", content="You are SprintStart's assistant."),
            Message(role="user", content="hi"),
        ]
    )

    blocks = _system_blocks(system)
    assert len(blocks) == 1
    assert blocks[0]["text"] == "You are SprintStart's assistant."
    assert blocks[0]["cache_control"] == _EPHEMERAL


def test_system_parts_are_joined_into_one_cached_block() -> None:
    system, _ = _to_anthropic_messages(
        [
            Message(role="system", content="First."),
            Message(role="system", content="Second."),
            Message(role="user", content="hi"),
        ]
    )

    assert [block["text"] for block in _system_blocks(system)] == ["First.\n\nSecond."]


def test_absent_system_prompt_is_omitted_not_an_empty_block() -> None:
    from anthropic import Omit

    system, _ = _to_anthropic_messages([Message(role="user", content="hi")])

    assert isinstance(system, Omit)


def test_final_user_message_carries_a_cache_breakpoint() -> None:
    _, converted = _to_anthropic_messages([Message(role="user", content="hi")])

    blocks = _blocks(converted[-1]["content"])
    assert blocks == [{"type": "text", "text": "hi", "cache_control": _EPHEMERAL}]


def test_user_content_keeps_one_shape_whatever_its_position() -> None:
    """A user turn must serialise identically whether or not it is last.

    Caching matches on exact bytes, so a turn rendered as a bare string in one
    request and a block list in the next breaks the prefix match and silently
    never hits — which is the entire value of the breakpoint below it.
    """
    _, last = _to_anthropic_messages([Message(role="user", content="what broke?")])
    _, not_last = _to_anthropic_messages(
        [
            Message(role="user", content="what broke?"),
            Message(role="assistant", content="looking"),
        ]
    )

    marked = _blocks(last[0]["content"])[0]
    plain = _blocks(not_last[0]["content"])[0]

    assert plain == {"type": "text", "text": "what broke?"}
    assert {k: v for k, v in marked.items() if k != "cache_control"} == plain


def test_breakpoint_lands_on_the_tool_result_of_the_answer_call() -> None:
    """The chat agent's answer call ends on tool results; the breakpoint there
    means the next request re-reads everything up to and including them."""
    _, converted = _to_anthropic_messages(
        [
            Message(role="system", content="sys"),
            Message(role="user", content="what broke?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="c1", name="retrieve", arguments={})],
            ),
            Message(role="tool", content="a chunk of source text", tool_call_id="c1"),
        ]
    )

    blocks = _blocks(converted[-1]["content"])
    assert blocks[-1]["type"] == "tool_result"
    assert blocks[-1]["cache_control"] == _EPHEMERAL


def test_only_the_last_block_is_marked() -> None:
    """More than four breakpoints is an API error, so exactly one lands in the
    message list however many blocks it holds."""
    _, converted = _to_anthropic_messages(
        [
            Message(role="user", content="what broke?"),
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(id="c1", name="retrieve", arguments={}),
                    ToolCall(id="c2", name="grep", arguments={}),
                ],
            ),
            Message(role="tool", content="first", tool_call_id="c1"),
            Message(role="tool", content="second", tool_call_id="c2"),
        ]
    )

    marked = [
        block
        for message in converted
        for block in _blocks(message["content"])
        if block.get("cache_control")
    ]
    assert len(marked) == 1
    assert marked[0]["tool_use_id"] == "c2"


def test_a_turn_extends_the_previous_turns_cached_prefix() -> None:
    """Appending tool results leaves the rendered prefix byte-identical.

    That is what lets the next deciding call read what this turn wrote. Note
    this covers the *messages* half only: `chat` sends tool definitions and
    `stream` does not, and tools render first, so the answer call of a turn
    cannot read its own deciding call's entry (see `_mark_last_block`).
    """
    decide: list[Message] = [
        Message(role="system", content="sys"),
        Message(role="user", content="what broke?"),
    ]
    answer: list[Message] = [
        *decide,
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="c1", name="retrieve", arguments={})],
        ),
        Message(role="tool", content="a chunk", tool_call_id="c1"),
    ]

    decide_system, decide_msgs = _to_anthropic_messages(decide)
    answer_system, answer_msgs = _to_anthropic_messages(answer)

    # Identical prefix apart from the breakpoint marker itself, which is
    # metadata rather than rendered content.
    assert decide_system == answer_system
    decided = _blocks(decide_msgs[0]["content"])[0]
    answered = _blocks(answer_msgs[0]["content"])[0]
    assert {k: v for k, v in decided.items() if k != "cache_control"} == answered
