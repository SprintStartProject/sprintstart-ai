"""Streaming tool-decision tests: `chat_stream` across all three LLM clients.

These pin the contract the chat agent now relies on: reasoning and answer
deltas arrive live during a tool-decision turn, tool calls are assembled from
fragments (or recovered from leaked markup), and the stream always ends with a
terminal `ChatResult` that mirrors what was streamed.
"""

import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pytest

from llm.anthropic_client import AnthropicClient
from llm.base import (
    ChatResult,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
)
from llm.ollama_client import OllamaClient
from llm.openai_client import OpenAIClient
from llm.tool_call_recovery import GuardedTextCollector, recover_tool_calls
from tests.llm.test_anthropic_client import (  # pyright: ignore[reportPrivateUsage]
    _delta,
    _FakeMessages,
)
from tests.llm.test_ollama_client import (  # pyright: ignore[reportPrivateUsage]
    _TOOL_SPEC,
    _FakeOllamaClient,
    _make_client,
)
from tests.llm.test_openai_client import (  # pyright: ignore[reportPrivateUsage]
    make_client as make_openai_client,
)
from tests.llm.test_tool_call_recovery import (  # pyright: ignore[reportPrivateUsage]
    _LEAKED,
)

_USER = [Message(role="user", content="What were the blockers?")]


def _sse(chunks: list[dict[str, Any]]) -> str:
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _chunk(delta: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "1",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
    }


class TestOpenAIChatStream:
    def test_streams_reasoning_then_assembles_fragmented_tool_calls(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is True
            assert body["tools"], "tool catalogue must be forwarded"

            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(
                    [
                        _chunk({"reasoning": "I should search the retro."}),
                        _chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "retrieve",
                                            "arguments": '{"que',
                                        },
                                    }
                                ]
                            }
                        ),
                        _chunk(
                            {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'ry": "blockers"}'},
                                    }
                                ]
                            }
                        ),
                    ]
                ),
            )

        client = cast("OpenAIClient", make_openai_client(handler))
        events = list(client.chat_stream(_USER, tools=[_TOOL_SPEC]))

        assert events[0] == ReasoningDelta("I should search the retro.")
        terminal = events[-1]
        assert isinstance(terminal, ChatResult)
        assert terminal.tool_calls == [
            ToolCall(id="call_1", name="retrieve", arguments={"query": "blockers"})
        ]
        assert all(not isinstance(e, TextDelta) for e in events)

    def test_plain_answer_streams_deltas_and_result_mirrors_them(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["stream"] is True
            assert "tools" not in body
            assert "max_tokens" not in body

            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse([_chunk({"content": "Hel"}), _chunk({"content": "lo"})]),
            )

        client = cast("OpenAIClient", make_openai_client(handler))
        events = list(client.chat_stream(_USER))

        assert events == [
            TextDelta("Hel"),
            TextDelta("lo"),
            ChatResult(text="Hello", tool_calls=[]),
        ]

    def test_tool_mode_sends_reasoning_budget_params(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            assert body["max_tokens"]
            assert body["reasoning"]["max_tokens"] == 1024
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse([_chunk({"content": "done"})]),
            )

        client = cast(
            "OpenAIClient",
            make_openai_client(handler, max_tokens=4096, reasoning_max_tokens=1024),
        )
        events = list(client.chat_stream(_USER, tools=[_TOOL_SPEC]))

        assert events[-1] == ChatResult(text="done", tool_calls=[])

    def test_leaked_markup_is_recovered_not_shown(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse([_chunk({"content": _LEAKED})]),
            )

        client = cast("OpenAIClient", make_openai_client(handler))
        events = list(client.chat_stream(_USER, tools=[_TOOL_SPEC]))

        assert all(not isinstance(e, TextDelta) for e in events)
        terminal = events[-1]
        assert isinstance(terminal, ChatResult)
        expected_calls, cleaned = recover_tool_calls(_LEAKED)
        assert len(terminal.tool_calls) == len(expected_calls)
        for got, want in zip(terminal.tool_calls, expected_calls, strict=True):
            assert got.name == want.name
            assert got.arguments == want.arguments
        assert terminal.text == cleaned
        assert "DSML" not in terminal.text


class TestAnthropicChatStream:
    def test_streams_thinking_text_and_tool_input(self) -> None:
        client = AnthropicClient(api_key="test", chat_model="claude-haiku-4-5")
        messages = _FakeMessages(
            [
                _delta("thinking_delta", thinking="Planning the search."),
                _delta("text_delta", text="Answer."),
                SimpleNamespace(
                    type="content_block_start",
                    index=1,
                    content_block=SimpleNamespace(
                        type="tool_use", id="toolu_1", name="retrieve"
                    ),
                ),
                SimpleNamespace(
                    type="content_block_delta",
                    index=1,
                    delta=SimpleNamespace(
                        type="input_json_delta", partial_json='{"query": "blockers"}'
                    ),
                ),
                SimpleNamespace(type="content_block_stop", index=1),
                SimpleNamespace(type="message_stop"),
            ]
        )
        client.client.messages = cast("Any", messages)

        events = list(client.chat_stream(_USER, tools=[_TOOL_SPEC]))

        assert events == [
            ReasoningDelta("Planning the search."),
            TextDelta("Answer."),
            ChatResult(
                text="Answer.",
                tool_calls=[
                    ToolCall(
                        id="toolu_1", name="retrieve", arguments={"query": "blockers"}
                    )
                ],
            ),
        ]

    def test_tools_disable_thinking_on_the_decision_turn(self) -> None:
        from anthropic import Omit

        client = AnthropicClient(
            api_key="test",
            chat_model="claude-haiku-4-5",
            thinking_budget_tokens=1024,
        )
        messages = _FakeMessages([SimpleNamespace(type="message_stop")])
        client.client.messages = cast("Any", messages)

        events = list(client.chat_stream(_USER, tools=[_TOOL_SPEC]))

        assert isinstance(messages.stream_kwargs["thinking"], Omit)
        assert messages.stream_kwargs["tools"]
        assert events[-1] == ChatResult(text="", tool_calls=[])


class TestOllamaChatStream:
    def test_streams_thinking_and_content(self) -> None:
        fake = _FakeOllamaClient(
            stream_tokens=["Hello"],
            stream_thinking=["hmm"],
        )
        client = cast("OllamaClient", _make_client(inner_client=fake))

        events = list(client.chat_stream(_USER, tools=[_TOOL_SPEC]))

        assert events == [
            ReasoningDelta("hmm"),
            TextDelta("Hello"),
            ChatResult(text="Hello", tool_calls=[]),
        ]
        assert fake.last_tools is not None

    def test_structured_tool_calls_reach_the_terminal_result(self) -> None:
        fake = _FakeOllamaClient(
            stream_tokens=[""],
            tool_calls=[("retrieve", {"query": "x"})],
        )
        client = cast("OllamaClient", _make_client(inner_client=fake))

        events = list(client.chat_stream(_USER, tools=[_TOOL_SPEC]))

        terminal = events[-1]
        assert isinstance(terminal, ChatResult)
        assert len(terminal.tool_calls) == 1
        assert terminal.tool_calls[0].name == "retrieve"
        assert terminal.tool_calls[0].arguments == {"query": "x"}


class TestGuardedTextCollector:
    def test_leaked_markup_is_held_then_recovered(self) -> None:
        collector = GuardedTextCollector()

        visible = collector.add_text(_LEAKED)

        assert visible == ""
        recovered, cleaned, _ = collector.finish()
        assert recovered
        assert "DSML" not in cleaned

    def test_clean_text_releases_its_tail_at_flush(self) -> None:
        collector = GuardedTextCollector()

        first = collector.add_text("Hello <to")
        tail = collector.flush_visible()
        recovered, cleaned, _ = collector.finish()

        assert first == "Hello"
        assert tail == " <to"
        assert recovered == []
        assert first + tail == "Hello <to"
        assert "DSML" not in cleaned

    def test_plain_text_streams_untouched(self) -> None:
        collector = GuardedTextCollector()

        assert collector.add_text("plain answer") == "plain answer"
        assert collector.flush_visible() == ""
        recovered, cleaned, _ = collector.finish()
        assert recovered == []
        assert cleaned == "plain answer"


@pytest.mark.parametrize("empty", ["", "   "])
def test_collector_ignores_whitespace_only_reasoning(empty: str) -> None:
    collector = GuardedTextCollector()

    collector.add_reasoning(empty)

    assert collector.finish() == ([], "", "")
