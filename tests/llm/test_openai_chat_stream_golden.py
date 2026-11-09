"""The OpenAI client's streaming tool turn is pinned request and event for event.

``chat_stream`` folds reasoning fragments, answer text, leaked tool markup and
fragmented tool calls into one event sequence, and every branch is a way a provider
cuts a reply up. Each scenario is replayed at several chunk sizes with and without
tools and a reasoning budget, and the request body plus the events are compared to a
checked-in copy. After a deliberate change rerun with ``UPDATE_PROMPT_GOLDENS=1`` and
review the diff.
"""

import json
import os
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import httpx

from llm.base import Message, ToolSpec
from llm.errors import LLMUnavailableError
from llm.openai_client import OpenAIClient
from tests.llm.test_openai_client import make_client
from tests.llm.test_tool_call_recovery import _LEAKED

_GOLDEN = Path(__file__).parent / "openai_chat_stream_events.txt"
_USER = [Message(role="user", content="What were the blockers?")]
_TOOLS = [
    ToolSpec(
        name="retrieve",
        description="Search the docs.",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
]
_Delta = dict[str, Any]
_Handler = Callable[[httpx.Request], httpx.Response]
_Factory = Callable[[list[dict[str, Any]]], _Handler]


@dataclass(frozen=True)
class _Config:
    tools: bool
    reasoning: int | None
    max_tokens: int | None


# Recovered tool calls get a random id; everything else in the output is fixed.
_GENERATED_ID = re.compile(r"call_[0-9a-f]{32}")


def _text(content: str, size: int) -> list[_Delta]:
    return [{"content": content[i : i + size]} for i in range(0, len(content), size)]


def _call(index: int, **fields: Any) -> _Delta:
    function = {k: v for k, v in fields.items() if k in ("name", "arguments")}
    call: _Delta = {"index": index}
    if "id" in fields:
        call["id"] = fields["id"]
    if function:
        call["function"] = function
    return {"tool_calls": [call]}


def _scenarios(size: int) -> dict[str, list[_Delta]]:
    return {
        "plain": _text("Blockers were the review queue.", size),
        "reasoning-then-text": [
            {"reasoning": "think "},
            {"reasoning_content": "more"},
            *_text("Answer.", size),
        ],
        "whitespace-reasoning-ignored": [{"reasoning": "   "}, *_text("ok", size)],
        "reasoning-details": [
            {
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "a", "index": 0}
                ]
            },
            {
                "reasoning_details": [
                    {"type": "reasoning.text", "text": "b", "index": 0}
                ]
            },
            {"reasoning_details": [{"type": "reasoning.summary", "summary": "s"}]},
            *_text("Done.", size),
        ],
        "fragmented-call": [
            _call(0, id="call_1", name="retrieve", arguments='{"que'),
            _call(0, arguments='ry": "blockers"}'),
        ],
        "call-with-content": [
            *_text("Let me look. ", size),
            _call(0, id="c9", name="retrieve", arguments='{"query": "x"}'),
        ],
        "two-calls-out-of-order": [
            _call(1, id="b", name="second", arguments="{}"),
            _call(0, id="a", name="first", arguments='{"k": 1}'),
        ],
        "call-without-id": [_call(2, name="retrieve", arguments='{"query": "q"}')],
        "call-bad-arguments": [_call(0, id="x", name="retrieve", arguments="{oops")],
        "call-no-arguments": [_call(0, id="x", name="retrieve")],
        "leaked-markup": _text(_LEAKED, size),
        "leaked-markup-after-text": _text("Checking.\n\n" + _LEAKED, size),
        "marker-prefix-prose": _text("Use <｜｜DSML as a label.", size),
        "structured-wins-over-markup": [
            *_text(_LEAKED, size),
            _call(0, id="s", name="retrieve", arguments="{}"),
        ],
        "empty-and-bare-deltas": [{}, {"content": ""}, {"role": "assistant"}],
        "nothing": [],
    }


def _sse(deltas: list[_Delta], usage_chunk: bool) -> str:
    chunks: list[dict[str, Any]] = [
        {
            "id": "1",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
        }
        for delta in deltas
    ]
    if usage_chunk:
        chunks.append(
            {
                "id": "1",
                "object": "chat.completion.chunk",
                "choices": [],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            }
        )
    return "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"


def _record(handler_for: _Factory, config: _Config) -> str:
    bodies: list[dict[str, Any]] = []
    client: OpenAIClient = make_client(
        handler_for(bodies),
        max_tokens=config.max_tokens,
        reasoning_max_tokens=config.reasoning,
    )
    try:
        events = list(client.chat_stream(_USER, tools=_TOOLS if config.tools else None))
    except (LLMUnavailableError, httpx.HTTPError) as exc:
        outcome: list[Any] = [f"{type(exc).__name__}: {exc}"]
    else:
        outcome = [
            {type(e).__name__: asdict(e) if is_dataclass(e) else repr(e)}
            for e in events
        ]
    dumped = json.dumps({"request": bodies, "events": outcome}, sort_keys=True)
    return _GENERATED_ID.sub("call_<generated>", dumped)


def _serving(deltas: list[_Delta], usage_chunk: bool) -> _Factory:
    def handler_for(
        bodies: list[dict[str, Any]],
    ) -> _Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=_sse(deltas, usage_chunk),
            )

        return handler

    return handler_for


def _failing(
    status: int,
) -> _Factory:
    def handler_for(
        bodies: list[dict[str, Any]],
    ) -> _Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(status, json={"error": {"message": "boom"}})

        return handler

    return handler_for


def _dropping(
    deltas: list[_Delta],
) -> _Factory:
    def stream() -> Iterator[bytes]:
        yield _sse(deltas, False).split("data: [DONE]")[0].encode()
        raise httpx.ReadError("connection dropped")

    def handler_for(
        bodies: list[dict[str, Any]],
    ) -> _Handler:
        def handler(request: httpx.Request) -> httpx.Response:
            bodies.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=stream()
            )

        return handler

    return handler_for


def _lines() -> str:
    lines: list[str] = []
    configs = [
        ("tools", _Config(True, None, None)),
        ("tools-reasoning", _Config(True, 512, 2048)),
        ("tools-reasoning-no-max", _Config(True, 512, None)),
        ("no-tools-reasoning", _Config(False, 512, 2048)),
        ("no-tools", _Config(False, None, 4096)),
    ]
    for size in (1, 3, 7, 1000):
        for name, deltas in _scenarios(size).items():
            for label, config in configs:
                for usage in (False, True):
                    if usage and size != 3:
                        continue
                    run = _record(_serving(deltas, usage), config)
                    lines.append(f"{name} size={size} {label} usage={usage} {run}\n")
    base = _Config(True, None, None)
    for status in (401, 429, 500):
        lines.append(f"http-{status} {_record(_failing(status), base)}\n")
    dropped = _record(_dropping(_text("Hello there", 3)), base)
    lines.append(f"dropped-midway {dropped}\n")
    return "".join(lines)


def test_every_scenario_matches_the_golden() -> None:
    actual = _lines()
    if os.environ.get("UPDATE_PROMPT_GOLDENS") == "1":
        _GOLDEN.write_text(actual, encoding="utf-8", newline="\n")
    assert actual == _GOLDEN.read_text(encoding="utf-8")
