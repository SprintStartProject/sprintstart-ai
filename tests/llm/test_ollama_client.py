import base64
import os
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import httpx
import ollama
import pytest

from llm.base import Message, ToolSpec
from llm.errors import LLMUnavailableError
from llm.ollama_client import OllamaBackend, OllamaClient
from tests.conftest import vision_required

# Minimal 1×1 red PNG
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
)
_TINY_PNG_BYTES = base64.b64decode(_TINY_PNG_B64)

_TEST_MODEL = os.environ.get("OLLAMA_MODEL", "test-model")
_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "test-embed-model")
_OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")


def _ollama_is_reachable() -> bool:
    try:
        httpx.get(_OLLAMA_BASE, timeout=2)
        return True
    except httpx.HTTPError:
        # Any transport-level failure (refused connection, connect timeout, DNS)
        # means Ollama isn't reachable; skip rather than error out collection.
        return False


ollama_required = pytest.mark.skipif(
    not _ollama_is_reachable(), reason="Ollama not reachable - skipping"
)


class _FakeOllamaClient:
    def __init__(
        self,
        chat_content: str = "",
        stream_tokens: list[str] | None = None,
        embed_vector: list[float] | None = None,
        chat_error: Exception | None = None,
        embed_error: Exception | None = None,
        vision_error: Exception | None = None,
        tool_calls: list[tuple[str, dict[str, object]]] | None = None,
    ) -> None:
        self._chat_content = chat_content
        self._stream_tokens = stream_tokens or [chat_content]
        self._embed_vector = embed_vector or []
        self._chat_error = chat_error
        self._embed_error = embed_error
        self._vision_error = vision_error
        self._tool_calls = tool_calls or []
        self.last_tools: list[Mapping[str, Any]] | None = None
        self.last_chat_model: str | None = None
        self.last_chat_messages: list[Message] | None = None
        self.last_chat_options: Mapping[str, Any] | None = None
        self.last_embed_model: str | None = None
        self.last_embed_input: list[str] | None = None
        self.last_vision_model: str | None = None
        self.last_vision_images: list[bytes] | None = None

    def chat(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ollama.ChatResponse:
        self.last_chat_model = model
        self.last_chat_messages = list(messages) if messages is not None else None
        self.last_chat_options = options
        if self._chat_error is not None:
            raise self._chat_error
        return ollama.ChatResponse(
            message=ollama.Message(role="assistant", content=self._chat_content)
        )

    def chat_tools(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ollama.ChatResponse:
        self.last_chat_model = model
        self.last_tools = list(tools) if tools is not None else None
        if self._chat_error is not None:
            raise self._chat_error
        tool_calls = [
            ollama.Message.ToolCall(
                function=ollama.Message.ToolCall.Function(name=name, arguments=args)
            )
            for name, args in self._tool_calls
        ]
        return ollama.ChatResponse(
            message=ollama.Message(
                role="assistant",
                content=self._chat_content,
                tool_calls=tool_calls or None,
            )
        )

    def chat_stream(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
    ) -> Iterator[ollama.ChatResponse]:
        self.last_chat_model = model
        self.last_chat_messages = list(messages) if messages is not None else None
        if self._chat_error is not None:
            raise self._chat_error
        for token in self._stream_tokens:
            yield ollama.ChatResponse(
                message=ollama.Message(role="assistant", content=token)
            )

    def embed(
        self,
        model: str = "",
        input: Sequence[str] = (),
    ) -> ollama.EmbedResponse:
        self.last_embed_model = model
        self.last_embed_input = list(input)
        if self._embed_error is not None:
            raise self._embed_error
        return ollama.EmbedResponse(
            embeddings=[self._embed_vector for _ in self.last_embed_input]
        )

    def chat_with_images(
        self,
        model: str = "",
        prompt: str = "",
        images: list[bytes] | None = None,
    ) -> ollama.ChatResponse:
        self.last_vision_model = model
        self.last_vision_images = images
        if self._vision_error is not None:
            raise self._vision_error
        return ollama.ChatResponse(
            message=ollama.Message(role="assistant", content=self._chat_content)
        )


_VISION_MODEL = os.environ.get("OLLAMA_VISION_MODEL", "test-vision-model")


def _make_client(
    host: str | None = _OLLAMA_BASE,
    model: str | None = _TEST_MODEL,
    embed_model: str | None = _EMBED_MODEL,
    vision_model: str | None = _VISION_MODEL,
    inner_client: OllamaBackend | None = None,
) -> OllamaClient:
    return OllamaClient(
        host=host,
        model=model,
        embed_model=embed_model,
        vision_model=vision_model,
        client=inner_client,
    )


_TOOL_SPEC: ToolSpec = {
    "name": "retrieve",
    "description": "search",
    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
}


class TestChat:
    def test_returns_text_when_no_tool_calls(self) -> None:
        fake = _FakeOllamaClient(chat_content="hello")
        client = _make_client(inner_client=fake)

        result = client.chat([Message(role="user", content="hi")], tools=[_TOOL_SPEC])

        assert result.text == "hello"
        assert result.tool_calls == []
        assert fake.last_tools is not None  # tool catalogue was forwarded

    def test_parses_tool_calls(self) -> None:
        fake = _FakeOllamaClient(tool_calls=[("retrieve", {"query": "x"})])
        client = _make_client(inner_client=fake)

        result = client.chat([Message(role="user", content="hi")], tools=[_TOOL_SPEC])

        assert [(c.name, c.arguments) for c in result.tool_calls] == [
            ("retrieve", {"query": "x"})
        ]

    def test_synthesises_unique_ids_across_calls(self) -> None:
        fake = _FakeOllamaClient(tool_calls=[("retrieve", {"query": "x"})])
        client = _make_client(inner_client=fake)
        messages = [Message(role="user", content="hi")]

        first = client.chat(messages, tools=[_TOOL_SPEC])
        second = client.chat(messages, tools=[_TOOL_SPEC])

        assert first.tool_calls[0].id != second.tool_calls[0].id

    def test_wraps_backend_errors(self) -> None:
        fake = _FakeOllamaClient(chat_error=ConnectionError("refused"))
        client = _make_client(inner_client=fake)

        with pytest.raises(LLMUnavailableError):
            client.chat([Message(role="user", content="hi")])


class TestGenerateHappyPath:
    def test_returns_assistant_content(self) -> None:
        fake = _FakeOllamaClient(chat_content="Hello there!")
        client = _make_client(inner_client=fake)
        messages = [Message(role="user", content="Say hello")]

        result = client.generate(messages)

        assert result == "Hello there!"
        assert fake.last_chat_model == _TEST_MODEL
        assert fake.last_chat_messages == [{"role": "user", "content": "Say hello"}]


@ollama_required
class TestGenerateIntegration:
    def test_returns_a_string(self) -> None:
        client = _make_client()
        result = client.generate(
            [Message(role="user", content="Reply with one word: hello")]
        )
        assert isinstance(result, str)
        assert len(result) > 0


class TestStreamHappyPath:
    def test_yields_tokens(self) -> None:
        fake = _FakeOllamaClient(stream_tokens=["Hello", " there", "!"])
        client = _make_client(inner_client=fake)
        messages = [Message(role="user", content="Say hello")]

        result = list(client.stream(messages))

        assert result == ["Hello", " there", "!"]
        assert fake.last_chat_model == _TEST_MODEL

    def test_yields_single_token(self) -> None:
        fake = _FakeOllamaClient(chat_content="Hi!")
        client = _make_client(inner_client=fake)

        result = list(client.stream([Message(role="user", content="Hi")]))

        assert result == ["Hi!"]


@ollama_required
class TestStreamIntegration:
    def test_yields_strings(self) -> None:
        client = _make_client()
        tokens = list(
            client.stream([Message(role="user", content="Reply with one word: hello")])
        )
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)


class TestEmbedHappyPath:
    def test_returns_vector(self) -> None:
        fake_vector = [0.1, 0.2, 0.3]
        fake = _FakeOllamaClient(embed_vector=fake_vector)
        client = _make_client(inner_client=fake)

        result = client.embed("Say hello")

        assert result == fake_vector
        assert fake.last_embed_model == _EMBED_MODEL
        assert fake.last_embed_input == ["Say hello"]


class TestEmbedBatchHappyPath:
    def test_sends_all_texts_in_one_call(self) -> None:
        fake_vector = [0.1, 0.2, 0.3]
        fake = _FakeOllamaClient(embed_vector=fake_vector)
        client = _make_client(inner_client=fake)

        result = client.embed_batch(["hello", "world"])

        assert result == [fake_vector, fake_vector]
        assert fake.last_embed_model == _EMBED_MODEL
        assert fake.last_embed_input == ["hello", "world"]

    def test_returns_empty_list_for_no_input(self) -> None:
        fake = _FakeOllamaClient()
        client = _make_client(inner_client=fake)

        assert client.embed_batch([]) == []
        assert fake.last_embed_input is None

    def test_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(embed_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)

        with pytest.raises(LLMUnavailableError):
            client.embed_batch(["hello"])


@ollama_required
class TestEmbedIntegration:
    def test_returns_a_float_vector(self) -> None:
        client = _make_client()
        result = client.embed("Hello world")
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(v, float) for v in result)


class TestLLMUnavailableError:
    def test_generate_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(chat_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.generate([Message(role="user", content="hello")])

    def test_stream_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(chat_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            list(client.stream([Message(role="user", content="hello")]))

    def test_embed_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(embed_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.embed("hello")

    def test_generate_raises_on_ollama_response_error(self) -> None:
        fake = _FakeOllamaClient(chat_error=ollama.ResponseError("model not found"))
        client = _make_client(inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.generate([Message(role="user", content="hello")])

    def test_embed_raises_on_ollama_response_error(self) -> None:
        fake = _FakeOllamaClient(embed_error=ollama.ResponseError("model not found"))
        client = _make_client(inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.embed("hello")

    def test_caption_image_raises_when_vision_model_not_configured(self) -> None:
        fake = _FakeOllamaClient()
        client = _make_client(vision_model=None, inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.caption_image(b"\x89PNG")

    def test_caption_image_raises_on_connection_error(self) -> None:
        fake = _FakeOllamaClient(vision_error=ConnectionError("refused"))
        client = _make_client(host="http://localhost:1", inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.caption_image(b"\x89PNG")

    def test_caption_image_raises_on_ollama_response_error(self) -> None:
        fake = _FakeOllamaClient(vision_error=ollama.ResponseError("model not found"))
        client = _make_client(inner_client=fake)
        with pytest.raises(LLMUnavailableError):
            client.caption_image(b"\x89PNG")


class TestCaptionImageHappyPath:
    def test_returns_caption_string(self) -> None:
        fake = _FakeOllamaClient(chat_content="A red circle on a white background.")
        client = _make_client(inner_client=fake)
        image_bytes = b"\x89PNG\r\n\x1a\n"

        result = client.caption_image(image_bytes)

        assert result == "A red circle on a white background."
        assert fake.last_vision_model == _VISION_MODEL
        assert fake.last_vision_images == [image_bytes]

    def test_passes_correct_model(self) -> None:
        fake = _FakeOllamaClient(chat_content="desc")
        client = _make_client(vision_model="llava:latest", inner_client=fake)

        client.caption_image(b"img")

        assert fake.last_vision_model == "llava:latest"


@vision_required
class TestCaptionImageIntegration:
    def test_returns_a_non_empty_string(self) -> None:
        client = _make_client()
        result = client.caption_image(_TINY_PNG_BYTES)
        assert isinstance(result, str)
        assert len(result) > 0
