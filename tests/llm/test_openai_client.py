import json
from collections.abc import Callable

import httpx
import pytest

from llm.base import Message, ReasoningDelta, TextDelta, ToolSpec
from llm.errors import LLMUnavailableError
from llm.openai_client import OpenAIClient

Handler = Callable[[httpx.Request], httpx.Response]


def make_client(handler: Handler) -> OpenAIClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)

    return OpenAIClient(
        base_url="http://openai-compatible.test/v1",
        api_key="test-key",
        chat_model="chat-model",
        embed_model="embed-model",
        vision_model="vision-model",
        http_client=http_client,
    )


def test_generate_uses_chat_completions() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer test-key"

        body = json.loads(request.content)
        assert body["model"] == "chat-model"
        assert body["messages"][0]["content"] == "Hello"

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Hi there",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = make_client(handler)

    assert client.generate([Message(role="user", content="Hello")]) == "Hi there"


def test_stream_yields_tokens() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"

        body = json.loads(request.content)
        assert body["stream"] is True

        first_chunk = {
            "id": "1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "Hel"},
                    "finish_reason": None,
                }
            ],
        }
        second_chunk = {
            "id": "1",
            "object": "chat.completion.chunk",
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": "lo"},
                    "finish_reason": None,
                }
            ],
        }

        stream_body = (
            f"data: {json.dumps(first_chunk)}\n\n"
            f"data: {json.dumps(second_chunk)}\n\n"
            "data: [DONE]\n\n"
        )

        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
        )

    client = make_client(handler)

    assert list(client.stream([Message(role="user", content="Hello")])) == [
        TextDelta("Hel"),
        TextDelta("lo"),
    ]


def test_stream_keeps_reasoning_separate_and_skips_malformed_deltas() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [
            {"reasoning_content": "Checking evidence."},
            {"reasoning_content": ""},
            {"reasoning_content": 42},
            {"content": "Answer."},
        ]
        stream_body = (
            "".join(
                "data: "
                + json.dumps(
                    {
                        "id": "1",
                        "object": "chat.completion.chunk",
                        "choices": [
                            {"index": 0, "delta": delta, "finish_reason": None}
                        ],
                    }
                )
                + "\n\n"
                for delta in chunks
            )
            + "data: [DONE]\n\n"
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body,
        )

    client = make_client(handler)

    assert list(client.stream([Message(role="user", content="Hello")])) == [
        ReasoningDelta("Checking evidence."),
        TextDelta("Answer."),
    ]


def test_embed_uses_embeddings_endpoint() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert request.headers["authorization"] == "Bearer test-key"

        body = json.loads(request.content)
        assert body["model"] == "embed-model"
        assert body["input"] == ["hello world"]

        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {
                        "object": "embedding",
                        "index": 0,
                        "embedding": [0.1, 0.2, 0.3],
                    }
                ],
                "model": "embed-model",
            },
        )

    client = make_client(handler)

    assert client.embed("hello world") == [0.1, 0.2, 0.3]


def test_embed_batch_sends_all_inputs_in_one_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["input"] == ["first", "second"]

        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                ],
                "model": "embed-model",
            },
        )

    client = make_client(handler)

    assert client.embed_batch(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_batch_reorders_by_response_index() -> None:
    """OpenAI-compatible backends may return embeddings out of request order."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.3, 0.4]},
                    {"object": "embedding", "index": 0, "embedding": [0.1, 0.2]},
                ],
                "model": "embed-model",
            },
        )

    client = make_client(handler)

    assert client.embed_batch(["first", "second"]) == [[0.1, 0.2], [0.3, 0.4]]


def test_embed_batch_returns_empty_list_for_no_input() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("should not send a request for an empty batch")

    client = make_client(handler)

    assert client.embed_batch([]) == []


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n"
    b"\x00\x00\x00\rIHDR"
    b"\x00\x00\x00\x01"
    b"\x00\x00\x00\x01"
    b"\x08\x02\x00\x00\x00"
    b"\x90wS\xde"
    b"\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_caption_image_uses_detected_mime_type() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"

        body = json.loads(request.content)
        assert body["model"] == "vision-model"

        content = body["messages"][0]["content"]
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "image_url"
        assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-vision",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Image caption",
                        },
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    client = make_client(handler)

    assert client.caption_image(PNG_BYTES) == "Image caption"


def test_caption_image_without_vision_model_raises() -> None:

    client = OpenAIClient(
        base_url="http://openai-compatible.test/v1",
        api_key="test-key",
        chat_model="chat-model",
        embed_model="embed-model",
        vision_model=None,
    )

    with pytest.raises(LLMUnavailableError):
        client.caption_image(PNG_BYTES)


def test_errors_map_to_llm_unavailable_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={"error": {"message": "backend down"}},
        )

    client = make_client(handler)

    with pytest.raises(LLMUnavailableError):
        client.generate([Message(role="user", content="Hello")])


_TOOL_SPEC: ToolSpec = {
    "name": "retrieve",
    "description": "search",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
}


def test_chat_sends_tools_and_parses_tool_calls() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["tools"][0]["function"]["name"] == "retrieve"

        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "retrieve",
                                        "arguments": '{"query": "x"}',
                                    },
                                }
                            ],
                        },
                    }
                ],
            },
        )

    client = make_client(handler)

    result = client.chat([Message(role="user", content="hi")], tools=[_TOOL_SPEC])

    assert result.text == ""
    assert [(c.name, c.arguments) for c in result.tool_calls] == [
        ("retrieve", {"query": "x"})
    ]


def test_chat_without_tool_calls_returns_text() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "done"},
                    }
                ],
            },
        )

    client = make_client(handler)

    result = client.chat([Message(role="user", content="hi")])

    assert result.text == "done"
    assert result.tool_calls == []
