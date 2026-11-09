import logging
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any

import pytest

from llm import ollama_client
from llm.ollama_client import _OllamaAdapter


class _RecordingClient:
    """Stands in for ``ollama.Client``: records calls and returns canned responses."""

    def __init__(self, host: str | None = None, timeout: float | None = None) -> None:
        self.host = host
        self.timeout = timeout
        self.chat_calls: list[dict[str, Any]] = []
        self.embed_calls: list[dict[str, Any]] = []
        self.response = SimpleNamespace(prompt_eval_count=None)
        self.stream_chunks: list[SimpleNamespace] = []

    def chat(self, **kwargs: Any) -> Any:
        self.chat_calls.append(kwargs)
        if kwargs.get("stream"):
            return iter(self.stream_chunks)
        return self.response

    def embed(self, **kwargs: Any) -> Any:
        self.embed_calls.append(kwargs)
        return SimpleNamespace(embeddings=[[0.1, 0.2]])


@pytest.fixture
def clients(monkeypatch: pytest.MonkeyPatch) -> list[_RecordingClient]:
    created: list[_RecordingClient] = []

    def make(host: str | None = None, timeout: float | None = None) -> _RecordingClient:
        client = _RecordingClient(host=host, timeout=timeout)
        created.append(client)
        return client

    monkeypatch.setattr(ollama_client.ollama, "Client", make)
    return created


def _adapter(
    clients: list[_RecordingClient], *, num_ctx: int | None = None
) -> tuple[_OllamaAdapter, _RecordingClient]:
    adapter = _OllamaAdapter(
        host="http://ollama:11434", temperature=0.3, num_ctx=num_ctx, timeout=42.0
    )
    return adapter, clients[-1]


class TestConstruction:
    def test_passes_host_and_timeout_to_the_sdk(
        self, clients: list[_RecordingClient]
    ) -> None:
        _, client = _adapter(clients)

        assert (client.host, client.timeout) == ("http://ollama:11434", 42.0)

    def test_sends_only_the_temperature_without_a_context_size(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)

        adapter.chat(model="m", messages=[])

        assert client.chat_calls[0]["options"] == {"temperature": 0.3}

    def test_sends_the_context_size_when_one_is_configured(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients, num_ctx=8192)

        adapter.chat(model="m", messages=[])

        assert client.chat_calls[0]["options"] == {"temperature": 0.3, "num_ctx": 8192}


class TestChat:
    def test_forwards_model_and_messages_and_returns_the_response(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)
        messages = ({"role": "user", "content": "hi"},)

        response = adapter.chat(model="llama", messages=messages)

        assert response is client.response
        assert client.chat_calls == [
            {
                "model": "llama",
                "messages": [{"role": "user", "content": "hi"}],
                "options": {"temperature": 0.3},
            }
        ]

    def test_sends_an_empty_list_when_no_messages_are_given(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)

        adapter.chat(model="m")

        assert client.chat_calls[0]["messages"] == []

    def test_call_options_override_the_defaults_without_changing_them(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients, num_ctx=4096)

        adapter.chat(model="m", messages=[], options={"temperature": 0.9, "seed": 7})
        adapter.chat(model="m", messages=[])

        assert client.chat_calls[0]["options"] == {
            "temperature": 0.9,
            "num_ctx": 4096,
            "seed": 7,
        }
        assert client.chat_calls[1]["options"] == {"temperature": 0.3, "num_ctx": 4096}


class TestChatTools:
    def test_forwards_the_tools_as_a_list(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)
        tool = {"type": "function", "function": {"name": "grep"}}

        response = adapter.chat_tools(
            model="m", messages=[{"role": "user", "content": "x"}], tools=(tool,)
        )

        assert response is client.response
        assert client.chat_calls == [
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": [tool],
                "options": {"temperature": 0.3},
            }
        ]

    def test_sends_no_tools_rather_than_an_empty_list(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)

        adapter.chat_tools(model="m", tools=[])

        assert client.chat_calls[0]["tools"] is None
        assert client.chat_calls[0]["messages"] == []


class TestChatStream:
    def test_streams_every_chunk_with_stream_enabled(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)
        client.stream_chunks = [
            SimpleNamespace(prompt_eval_count=None),
            SimpleNamespace(prompt_eval_count=None),
        ]

        chunks = list(
            adapter.chat_stream(model="m", messages=[{"role": "user", "content": "x"}])
        )

        assert chunks == client.stream_chunks
        assert client.chat_calls == [
            {
                "model": "m",
                "messages": [{"role": "user", "content": "x"}],
                "tools": None,
                "stream": True,
                "options": {"temperature": 0.3},
            }
        ]

    def test_forwards_tools_and_defaults_messages(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)
        tool = {"type": "function", "function": {"name": "grep"}}

        stream: Iterator[Any] = adapter.chat_stream(model="m", tools=[tool])
        list(stream)

        assert client.chat_calls[0]["tools"] == [tool]
        assert client.chat_calls[0]["messages"] == []

    def test_warns_when_the_final_chunk_shows_a_full_window(
        self, clients: list[_RecordingClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter, client = _adapter(clients, num_ctx=100)
        client.stream_chunks = [
            SimpleNamespace(prompt_eval_count=None),
            SimpleNamespace(prompt_eval_count=100),
        ]

        with caplog.at_level(logging.WARNING, logger="llm.ollama_client"):
            list(adapter.chat_stream(model="m"))

        assert len(caplog.records) == 1


class TestEmbed:
    def test_forwards_model_and_input_as_a_list(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)

        response = adapter.embed(model="nomic", input=("a", "b"))

        assert response.embeddings == [[0.1, 0.2]]
        assert client.embed_calls == [{"model": "nomic", "input": ["a", "b"]}]

    def test_sends_an_empty_input_by_default(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)

        adapter.embed(model="nomic")

        assert client.embed_calls == [{"model": "nomic", "input": []}]


class TestChatWithImages:
    def test_sends_one_user_message_carrying_the_prompt_and_images(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)

        response = adapter.chat_with_images(
            model="llava", prompt="what is this?", images=[b"png"]
        )

        assert response is client.response
        assert client.chat_calls == [
            {
                "model": "llava",
                "messages": [
                    {"role": "user", "content": "what is this?", "images": [b"png"]}
                ],
                "options": {"temperature": 0.3},
            }
        ]

    def test_sends_an_empty_image_list_when_none_are_given(
        self, clients: list[_RecordingClient]
    ) -> None:
        adapter, client = _adapter(clients)

        adapter.chat_with_images(model="llava", prompt="p")

        assert client.chat_calls[0]["messages"][0]["images"] == []


class TestTruncationWarning:
    @pytest.mark.parametrize("used", [100, 150])
    def test_warns_when_the_prompt_fills_the_window(
        self,
        used: int,
        clients: list[_RecordingClient],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        adapter, client = _adapter(clients, num_ctx=100)
        client.response = SimpleNamespace(prompt_eval_count=used)

        with caplog.at_level(logging.WARNING, logger="llm.ollama_client"):
            adapter.chat(model="m", messages=[])

        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert record.levelno == logging.WARNING
        assert record.args == (used, 100)
        assert "truncated" in record.getMessage()

    def test_stays_quiet_below_the_window(
        self, clients: list[_RecordingClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter, client = _adapter(clients, num_ctx=100)
        client.response = SimpleNamespace(prompt_eval_count=99)

        with caplog.at_level(logging.WARNING, logger="llm.ollama_client"):
            adapter.chat(model="m", messages=[])

        assert caplog.records == []

    def test_stays_quiet_without_a_count(
        self, clients: list[_RecordingClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter, _ = _adapter(clients, num_ctx=100)

        with caplog.at_level(logging.WARNING, logger="llm.ollama_client"):
            adapter.chat_tools(model="m")

        assert caplog.records == []

    def test_stays_quiet_without_a_configured_window(
        self, clients: list[_RecordingClient], caplog: pytest.LogCaptureFixture
    ) -> None:
        adapter, client = _adapter(clients)
        client.response = SimpleNamespace(prompt_eval_count=10_000)

        with caplog.at_level(logging.WARNING, logger="llm.ollama_client"):
            adapter.chat_with_images(model="m", prompt="p")

        assert caplog.records == []
