import logging
from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Protocol
from uuid import uuid4

import ollama

from llm.base import ChatResult, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from llm.tool_call_recovery import guard_stream, recover_tool_calls

logger = logging.getLogger(__name__)


def _to_ollama_messages(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {
            "role": message["role"],
            "content": message["content"],
        }
        tool_calls = message.get("tool_calls")
        if tool_calls:
            item["tool_calls"] = [
                {"function": {"name": call.name, "arguments": call.arguments}}
                for call in tool_calls
            ]
        name = message.get("name")
        if name:
            item["tool_name"] = name
        out.append(item)
    return out


def _to_ollama_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
            },
        }
        for tool in tools
    ]


def _from_ollama_response(response: ollama.ChatResponse) -> ChatResult:
    message = response.message
    calls = [
        ToolCall(
            id=f"call_{uuid4().hex}",
            name=call.function.name,
            arguments=dict(call.function.arguments),
        )
        for call in message.tool_calls or []
    ]
    text = message.content or ""
    # A local model may also emit tool calls as markup in the content instead of
    # returning them structurally; recover them when none came through the API so
    # the agent runs the tool rather than showing the hire the raw markup.
    if not calls:
        recovered, cleaned = recover_tool_calls(text)
        if recovered:
            return ChatResult(text=cleaned, tool_calls=recovered)
    return ChatResult(text=text, tool_calls=calls)


class OllamaBackend(Protocol):
    def chat(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ollama.ChatResponse: ...

    def chat_tools(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ollama.ChatResponse: ...

    def chat_stream(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
    ) -> Iterator[ollama.ChatResponse]: ...

    def embed(
        self,
        model: str = "",
        input: Sequence[str] = (),
    ) -> ollama.EmbedResponse: ...

    def chat_with_images(
        self,
        model: str = "",
        prompt: str = "",
        images: list[bytes] | None = None,
    ) -> ollama.ChatResponse: ...


class _OllamaAdapter:
    def __init__(
        self, host: str | None, temperature: float, num_ctx: int | None = None
    ) -> None:
        self._client = ollama.Client(host=host)
        self._options: dict[str, Any] = {"temperature": temperature}
        # num_ctx must be set explicitly: Ollama otherwise defaults to a small
        # context (typically 4096) and silently truncates oversized prompts,
        # which breaks the onboarding synthesis step (large grounded prompt ->
        # truncated -> non-JSON output -> SynthesisError fallback).
        if num_ctx is not None:
            self._options["num_ctx"] = num_ctx

    def _warn_if_truncated(self, response: ollama.ChatResponse) -> None:
        """Warn when the prompt filled (and was likely clipped to) the window.

        Ollama silently truncates prompts larger than ``num_ctx`` to the last
        ``num_ctx`` tokens, so ``prompt_eval_count >= num_ctx`` is a strong
        signal the prompt was cut. Only checked when ``num_ctx`` is set; a warm
        prompt cache can make the count smaller than the true prompt, so this
        may under-report (never false-positive).
        """
        num_ctx = self._options.get("num_ctx")
        if not num_ctx:
            return
        used = getattr(response, "prompt_eval_count", None)
        if used is not None and used >= num_ctx:
            logger.warning(
                "Ollama prompt filled the context window "
                "(prompt_eval_count=%s >= num_ctx=%s); the prompt was likely "
                "truncated. Increase OLLAMA_NUM_CTX or reduce the prompt size.",
                used,
                num_ctx,
            )

    def chat(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
        *,
        options: Mapping[str, Any] | None = None,
    ) -> ollama.ChatResponse:
        merged = {**self._options, **options} if options else self._options
        response = self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=model, messages=list(messages or []), options=merged
        )
        self._warn_if_truncated(response)
        return response

    def chat_tools(
        self,
        model: str = "",
        messages: Sequence[Mapping[str, Any]] | None = None,
        tools: Sequence[Mapping[str, Any]] | None = None,
    ) -> ollama.ChatResponse:
        response = self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            messages=list(messages or []),
            tools=list(tools) if tools else None,
            options=self._options,
        )
        self._warn_if_truncated(response)
        return response

    def chat_stream(
        self,
        model: str = "",
        messages: Sequence[Message] | None = None,
    ) -> Iterator[ollama.ChatResponse]:
        stream: Iterator[ollama.ChatResponse] = self._client.chat(  # type: ignore[assignment]
            model=model,
            messages=list(messages or []),
            stream=True,
            options=self._options,
        )
        for chunk in stream:
            # Truncation metadata rides on the final chunk; checking every chunk
            # is harmless (it's None until then).
            self._warn_if_truncated(chunk)
            yield chunk

    def embed(
        self,
        model: str = "",
        input: Sequence[str] = (),
    ) -> ollama.EmbedResponse:
        return self._client.embed(model=model, input=list(input))

    def chat_with_images(
        self,
        model: str = "",
        prompt: str = "",
        images: list[bytes] | None = None,
    ) -> ollama.ChatResponse:
        response = self._client.chat(  # pyright: ignore[reportUnknownMemberType]
            model=model,
            messages=[{"role": "user", "content": prompt, "images": images or []}],
            options=self._options,
        )
        self._warn_if_truncated(response)
        return response


_DEFAULT_TEMPERATURE = 0.1


class OllamaClient:
    def __init__(
        self,
        host: str | None = None,
        model: str | None = None,
        embed_model: str | None = None,
        vision_model: str | None = None,
        client: OllamaBackend | None = None,
        temperature: float = _DEFAULT_TEMPERATURE,
        num_ctx: int | None = None,
    ) -> None:
        self._host = host
        self._model = model
        self._embed_model = embed_model
        self._vision_model = vision_model
        self._client: OllamaBackend = (
            client
            if client is not None
            else _OllamaAdapter(host=host, temperature=temperature, num_ctx=num_ctx)
        )

    @property
    def model_name(self) -> str | None:
        return self._model

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        if self._model is None:
            raise ValueError("No model specified")
        try:
            response = self._client.chat_tools(
                model=self._model,
                messages=_to_ollama_messages(messages),
                tools=_to_ollama_tools(tools) if tools else None,
            )
            return _from_ollama_response(response)
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        if self._model is None:
            raise ValueError("No model specified")
        try:
            options = {"temperature": temperature} if temperature is not None else None
            response = self._client.chat(
                model=self._model, messages=messages, options=options
            )
            return response.message.content or ""
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

    def stream(self, messages: list[Message]) -> Iterator[str]:
        # See OpenAiCompatibleClient.stream: the same models reach this client, and the
        # streamed answer has no tool loop to hand a recovered call to.
        if self._model is None:
            raise ValueError("No model specified")
        # The model is passed in rather than re-read off self: this method raises
        # eagerly, the generator body does not, so the check here is only load-bearing
        # if the value it checked is the one that reaches the call.
        return guard_stream(self._stream_raw(self._model, messages))

    def _stream_raw(self, model: str, messages: list[Message]) -> Iterator[str]:
        try:
            for chunk in self._client.chat_stream(model=model, messages=messages):
                yield chunk.message.content or ""
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self._embed_model is None:
            raise ValueError("No embed model specified")
        if not texts:
            return []
        try:
            response = self._client.embed(model=self._embed_model, input=texts)
            return [list(embedding) for embedding in response.embeddings]
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc

    def caption_image(self, image_bytes: bytes) -> str:
        if self._vision_model is None:
            raise LLMUnavailableError(self._host)
        try:
            response = self._client.chat_with_images(
                model=self._vision_model,
                prompt="Describe this image concisely.",
                images=[image_bytes],
            )
            return response.message.content or ""
        except (ollama.ResponseError, ConnectionError, OSError) as exc:
            raise LLMUnavailableError(self._host, cause=exc) from exc
