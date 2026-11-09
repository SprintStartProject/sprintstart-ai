"""Exact wire shapes the three provider modules produce.

Each assertion compares a whole payload, because a field that drifts in these shapes
is rejected by the provider, or silently loses a tool call, long after the code that
broke it ran.
"""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from anthropic import Omit
from openai.types.chat import ChatCompletionMessage

from llm.anthropic_wire import (
    detect_image_mime_type as anthropic_mime,
)
from llm.anthropic_wire import (
    mark_last_block,
    to_anthropic_messages,
    to_anthropic_tools,
    user_message,
)
from llm.base import Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from llm.ollama_wire import from_ollama_response, to_ollama_messages, to_ollama_tools
from llm.openai_wire import (
    detect_image_mime_type as openai_mime,
)
from llm.openai_wire import (
    normalize_base_url,
    response_reasoning,
    sniff_image_type,
    to_openai_messages,
    to_openai_tools,
)

_TOOL: ToolSpec = {
    "name": "search",
    "description": "Search the corpus.",
    "parameters": {"type": "object", "properties": {"q": {"type": "string"}}},
}
_CALL = ToolCall(id="call-1", name="search", arguments={"q": "auth"})
_EPHEMERAL = {"type": "ephemeral"}

_PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 8
_JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 8
_GIF87 = b"GIF87a" + b"\x00" * 8
_GIF89 = b"GIF89a" + b"\x00" * 8
_WEBP = b"RIFF\x00\x00\x00\x00WEBPVP8 "
_BMP = b"BM" + b"\x00" * 8


# --- OpenAI -----------------------------------------------------------------


def test_openai_tools_are_function_declarations() -> None:
    assert to_openai_tools([_TOOL]) == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the corpus.",
                "parameters": _TOOL["parameters"],
            },
        }
    ]
    assert to_openai_tools([]) == []


@pytest.mark.parametrize(
    ("given", "expected"),
    [
        ("http://host:1234", "http://host:1234/v1"),
        ("http://host:1234/", "http://host:1234/v1"),
        ("http://host:1234/v1", "http://host:1234/v1"),
        ("http://host:1234/v1/", "http://host:1234/v1"),
        ("https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1"),
        ("http://host/v10", "http://host/v10/v1"),
    ],
)
def test_openai_base_url_always_ends_in_one_v1(given: str, expected: str) -> None:
    assert normalize_base_url(given) == expected


def test_openai_messages_cover_every_role() -> None:
    messages: list[Message] = [
        Message(role="system", content="be brief"),
        Message(role="user", content="hi"),
        Message(role="assistant", content="plain"),
        Message(role="tool", content="result", tool_call_id="call-1"),
        Message(role="tool", content="orphan"),
        Message(role="whatever", content="falls back to user"),
    ]

    assert to_openai_messages(messages) == [
        {"role": "system", "content": "be brief"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "plain"},
        {"role": "tool", "tool_call_id": "call-1", "content": "result"},
        {"role": "tool", "tool_call_id": "", "content": "orphan"},
        {"role": "user", "content": "falls back to user"},
    ]


def test_openai_assistant_tool_calls_carry_json_arguments() -> None:
    converted = to_openai_messages(
        [Message(role="assistant", content="", tool_calls=[_CALL])]
    )

    assert converted == [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q": "auth"}'},
                }
            ],
        }
    ]


def test_openai_assistant_without_calls_has_no_tool_calls_key() -> None:
    [converted] = to_openai_messages(
        [Message(role="assistant", content="x", tool_calls=[])]
    )

    assert "tool_calls" not in converted


def test_openai_reasoning_details_win_over_plain_reasoning() -> None:
    details: list[dict[str, object]] = [{"type": "reasoning.text", "text": "t"}]

    [both] = to_openai_messages(
        [
            Message(
                role="assistant",
                content="a",
                reasoning="plain",
                reasoning_details=details,
            )
        ]
    )
    [plain] = to_openai_messages(
        [Message(role="assistant", content="a", reasoning="plain")]
    )
    [neither] = to_openai_messages(
        [Message(role="assistant", content="a", reasoning="", reasoning_details=[])]
    )

    assert both == {"role": "assistant", "content": "a", "reasoning_details": details}
    assert plain == {"role": "assistant", "content": "a", "reasoning": "plain"}
    assert neither == {"role": "assistant", "content": "a"}


@pytest.mark.parametrize(
    ("data", "kind"),
    [
        (_PNG, "png"),
        (_JPEG, "jpeg"),
        (_GIF87, "gif"),
        (_GIF89, "gif"),
        (_WEBP, "webp"),
        (_BMP, "bmp"),
        (b"RIFF\x00\x00\x00\x00WAVEfmt ", None),
        (b"RIFF", None),
        (b"\x89PNG", None),
        (b"GIF8", None),
        (b"", None),
        (b"plain text", None),
    ],
)
def test_openai_sniffs_image_formats_by_magic_bytes(
    data: bytes, kind: str | None
) -> None:
    assert sniff_image_type(data) == kind


def test_openai_mime_type_is_image_slash_kind_or_unavailable() -> None:
    assert openai_mime(_PNG) == "image/png"
    assert openai_mime(_WEBP) == "image/webp"
    with pytest.raises(LLMUnavailableError, match="Could not detect image MIME type"):
        openai_mime(b"nope")


def _openai_message(**fields: Any) -> ChatCompletionMessage:
    return ChatCompletionMessage.model_validate({"role": "assistant", **fields})


def test_openai_response_reasoning_prefers_reasoning_over_reasoning_content() -> None:
    message = _openai_message(
        content="x", reasoning="primary", reasoning_content="legacy"
    )

    assert response_reasoning(message) == ("primary", [])


@pytest.mark.parametrize("blank", ["", "   ", None])
def test_openai_response_reasoning_falls_back_to_legacy_channel(
    blank: str | None,
) -> None:
    message = _openai_message(content="x", reasoning=blank, reasoning_content="legacy")

    assert response_reasoning(message) == ("legacy", [])


def test_openai_response_reasoning_is_none_when_both_channels_are_blank() -> None:
    message = _openai_message(content="x", reasoning="  ", reasoning_content=" ")

    assert response_reasoning(message) == (None, [])
    assert response_reasoning(_openai_message(content="x")) == (None, [])


def test_openai_response_reasoning_keeps_only_dict_details() -> None:
    message = _openai_message(
        content="x",
        reasoning_details=[{"type": "reasoning.text", "text": "a"}, "junk", 3],
    )

    assert response_reasoning(message) == (
        None,
        [{"type": "reasoning.text", "text": "a"}],
    )
    ignored = _openai_message(content="x", reasoning_details="not a list")
    assert response_reasoning(ignored) == (None, [])


# --- Anthropic --------------------------------------------------------------


def test_anthropic_tools_use_input_schema() -> None:
    assert to_anthropic_tools([_TOOL]) == [
        {
            "name": "search",
            "description": "Search the corpus.",
            "input_schema": _TOOL["parameters"],
        }
    ]


@pytest.mark.parametrize(
    ("data", "mime"),
    [
        (_PNG, "image/png"),
        (_JPEG, "image/jpeg"),
        (_GIF87, "image/gif"),
        (_GIF89, "image/gif"),
        (_WEBP, "image/webp"),
    ],
)
def test_anthropic_mime_type_by_magic_bytes(data: bytes, mime: str) -> None:
    assert anthropic_mime(data) == mime


@pytest.mark.parametrize(
    "data", [_BMP, b"RIFF\x00\x00\x00\x00WAVEfmt ", b"RIFF", b"", b"text"]
)
def test_anthropic_rejects_unsupported_images(data: bytes) -> None:
    with pytest.raises(LLMUnavailableError, match="Could not detect image MIME type"):
        anthropic_mime(data)


def test_anthropic_user_message_is_always_block_shaped() -> None:
    assert user_message("hi") == {
        "role": "user",
        "content": [{"type": "text", "text": "hi"}],
    }


def test_marking_the_last_block_leaves_the_earlier_ones_alone() -> None:
    messages: list[Any] = [
        {"role": "user", "content": [{"type": "text", "text": "a"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "b"},
                {"type": "text", "text": "c"},
            ],
        },
    ]

    mark_last_block(messages)

    assert messages[0]["content"] == [{"type": "text", "text": "a"}]
    assert messages[1]["content"] == [
        {"type": "text", "text": "b"},
        {"type": "text", "text": "c", "cache_control": _EPHEMERAL},
    ]


def test_marking_does_nothing_without_blocks() -> None:
    empty: list[Any] = []
    mark_last_block(empty)
    assert empty == []

    bare: list[Any] = [{"role": "user", "content": "bare"}]
    mark_last_block(bare)
    assert bare == [{"role": "user", "content": "bare"}]

    no_blocks: list[Any] = [{"role": "user", "content": []}]
    mark_last_block(no_blocks)
    assert no_blocks == [{"role": "user", "content": []}]


def test_anthropic_conversation_shape_end_to_end() -> None:
    system, converted = to_anthropic_messages(
        [
            Message(role="system", content="first"),
            Message(role="system", content=""),
            Message(role="system", content="second"),
            Message(role="user", content="question"),
            Message(role="assistant", content="thinking", tool_calls=[_CALL]),
            Message(role="tool", content="found", tool_call_id="call-1"),
            Message(role="tool", content="", tool_call_id="call-2"),
            Message(role="assistant", content="answer"),
        ]
    )

    assert system == [
        {"type": "text", "text": "first\n\nsecond", "cache_control": _EPHEMERAL}
    ]
    assert converted == [
        {"role": "user", "content": [{"type": "text", "text": "question"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "thinking"},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "search",
                    "input": {"q": "auth"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call-1", "content": "found"},
                {
                    "type": "tool_result",
                    "tool_use_id": "call-2",
                    "content": "(no result)",
                },
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "answer", "cache_control": _EPHEMERAL}
            ],
        },
    ]


def test_anthropic_without_a_system_prompt_omits_the_field() -> None:
    system, converted = to_anthropic_messages([Message(role="user", content="hi")])

    assert isinstance(system, Omit)
    assert len(converted) == 1


def test_anthropic_empty_assistant_turns_still_carry_a_block() -> None:
    _, converted = to_anthropic_messages(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content=""),
            Message(role="user", content="again"),
        ]
    )

    assert converted[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "(empty)"}],
    }


def test_anthropic_tool_results_flush_before_the_next_user_turn() -> None:
    _, converted = to_anthropic_messages(
        [
            Message(role="tool", content="r", tool_call_id="c"),
            Message(role="user", content="next"),
        ]
    )

    assert [m["role"] for m in converted] == ["user", "user"]
    assert converted[0]["content"] == [
        {"type": "tool_result", "tool_use_id": "c", "content": "r"}
    ]


def test_anthropic_orphan_tool_result_has_an_empty_id() -> None:
    _, converted = to_anthropic_messages([Message(role="tool", content="r")])

    [block] = cast(Any, converted[0]["content"])
    assert block["tool_use_id"] == ""


# --- Ollama -----------------------------------------------------------------


def test_ollama_messages_translate_tool_calls_and_names() -> None:
    assert to_ollama_messages(
        [
            Message(role="user", content="hi"),
            Message(role="assistant", content="", tool_calls=[_CALL]),
            Message(role="tool", content="found", name="search"),
            Message(role="assistant", content="x", tool_calls=[], name=""),
        ]
    ) == [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "search", "arguments": {"q": "auth"}}}
            ],
        },
        {"role": "tool", "content": "found", "tool_name": "search"},
        {"role": "assistant", "content": "x"},
    ]


def test_ollama_tools_are_function_declarations() -> None:
    assert to_ollama_tools([_TOOL]) == [
        {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the corpus.",
                "parameters": _TOOL["parameters"],
            },
        }
    ]


def _ollama_response(
    content: str | None, calls: list[tuple[str, dict[str, Any]]]
) -> Any:
    return SimpleNamespace(
        message=SimpleNamespace(
            content=content,
            tool_calls=[
                SimpleNamespace(function=SimpleNamespace(name=n, arguments=a))
                for n, a in calls
            ]
            or None,
        )
    )


def test_ollama_structured_calls_become_tool_calls_with_fresh_ids() -> None:
    result = from_ollama_response(
        _ollama_response("text", [("search", {"q": "a"}), ("search", {"q": "b"})])
    )

    assert result.text == "text"
    assert [(c.name, c.arguments) for c in result.tool_calls] == [
        ("search", {"q": "a"}),
        ("search", {"q": "b"}),
    ]
    ids = [c.id for c in result.tool_calls]
    assert all(i.startswith("call_") and len(i) == 5 + 32 for i in ids)
    assert len(set(ids)) == 2


def test_ollama_plain_text_and_missing_content() -> None:
    plain = from_ollama_response(_ollama_response("just text", []))
    empty = from_ollama_response(_ollama_response(None, []))

    assert (plain.text, plain.tool_calls) == ("just text", [])
    assert (empty.text, empty.tool_calls) == ("", [])


def test_ollama_recovers_markup_calls_only_when_none_came_back_structured() -> None:
    markup = (
        '<｜｜DSML｜｜tool_calls> <｜｜DSML｜｜invoke name="search"> '
        '<｜｜DSML｜｜parameter name="q" string="true">x</｜｜DSML｜｜parameter> '
        "</｜｜DSML｜｜invoke> </｜｜DSML｜｜tool_calls>"
    )

    recovered = from_ollama_response(_ollama_response(markup, []))
    kept = from_ollama_response(_ollama_response(markup, [("search", {"q": "real"})]))

    assert [(c.name, c.arguments) for c in recovered.tool_calls] == [
        ("search", {"q": "x"})
    ]
    assert "DSML" not in recovered.text
    assert [c.arguments for c in kept.tool_calls] == [{"q": "real"}]
    assert kept.text == markup
