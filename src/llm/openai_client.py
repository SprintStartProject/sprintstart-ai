import base64
import json
from collections.abc import Iterator
from typing import Any, cast

from openai import OpenAI, OpenAIError, omit
from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionChunk,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)
from openai.types.chat.chat_completion_chunk import ChoiceDelta

from llm.base import ChatResult, LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from llm.tool_call_recovery import guard_stream, recover_tool_calls


def _to_openai_tools(tools: list[ToolSpec]) -> list[ChatCompletionToolParam]:
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


def _loads_arguments(raw: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}


def _normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")

    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"

    return normalized


def _to_openai_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
    openai_messages: list[ChatCompletionMessageParam] = []

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            system_message: ChatCompletionSystemMessageParam = {
                "role": "system",
                "content": content,
            }
            openai_messages.append(system_message)
        elif role == "tool":
            tool_message: ChatCompletionToolMessageParam = {
                "role": "tool",
                "tool_call_id": message.get("tool_call_id", ""),
                "content": content,
            }
            openai_messages.append(tool_message)
        elif role == "assistant":
            assistant_message: ChatCompletionAssistantMessageParam = {
                "role": "assistant",
                "content": content,
            }
            tool_calls = message.get("tool_calls")
            if tool_calls:
                assistant_message["tool_calls"] = [
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": json.dumps(call.arguments),
                        },
                    }
                    for call in tool_calls
                ]
            openai_messages.append(assistant_message)
        else:
            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": content,
            }
            openai_messages.append(user_message)

    return openai_messages


def _detect_image_mime_type(image_bytes: bytes) -> str:
    image_type = _sniff_image_type(image_bytes)

    if image_type is None:
        raise LLMUnavailableError("Could not detect image MIME type")

    return f"image/{image_type}"


def _sniff_image_type(image_bytes: bytes) -> str | None:
    """Detect an image format from its magic bytes.

    Replaces the deprecated stdlib ``imghdr`` (removed in Python 3.13). Covers
    the formats accepted by the ingest API (png, jpeg, gif, webp, bmp).
    """
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "webp"
    if image_bytes.startswith(b"BM"):
        return "bmp"
    return None


class OpenAIClient(LLMClient):
    def __init__(
        self,
        base_url: str,
        api_key: str,
        chat_model: str,
        embed_model: str | None,
        vision_model: str | None = None,
        http_client: Any | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.chat_model = chat_model
        self.embed_model = embed_model
        self.vision_model = vision_model

        self.client = OpenAI(
            base_url=self.base_url,
            api_key=api_key,
            http_client=http_client,
        )

    @property
    def model_name(self) -> str | None:
        return self.chat_model

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=_to_openai_messages(messages),
                tools=_to_openai_tools(tools) if tools else omit,
            )
        except OpenAIError as exc:
            raise LLMUnavailableError(
                "OpenAI-compatible backend unavailable during chat "
                f"using model {self.chat_model!r} at {self.base_url}: {exc}"
            ) from exc

        message = response.choices[0].message
        calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            if call.type != "function":
                continue
            calls.append(
                ToolCall(
                    id=call.id,
                    name=call.function.name,
                    arguments=_loads_arguments(call.function.arguments),
                )
            )
        text = message.content or ""
        # Some models (e.g. DeepSeek via OpenRouter) write tool calls as markup in
        # the content instead of returning them structurally; the endpoint doesn't
        # lift them out, so they leak into the answer. Recover them when the API
        # gave us no structured calls, so the agent runs the tool instead of
        # showing a hire the raw markup.
        if not calls:
            recovered, cleaned = recover_tool_calls(text)
            if recovered:
                return ChatResult(text=cleaned, tool_calls=recovered)
        return ChatResult(text=text, tool_calls=calls)

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=_to_openai_messages(messages),
                temperature=temperature if temperature is not None else omit,
            )

            content = response.choices[0].message.content
            return content or ""

        except OpenAIError as exc:
            raise LLMUnavailableError(
                "OpenAI-compatible backend unavailable during chat "
                f"using model {self.chat_model!r} at {self.base_url}: {exc}"
            ) from exc

    def stream(self, messages: list[Message]) -> Iterator[str]:
        # Guarded for the same reason chat_with_tools recovers: this backend serves
        # models that write tool calls as markup in the content. There the markup can
        # be parsed back into a call and run; here the answer phase has no tool loop,
        # so the only thing to do with it is not show it to the hire.
        return guard_stream(self._stream_raw(messages))

    def _stream_raw(self, messages: list[Message]) -> Iterator[str]:
        try:
            stream: Iterator[ChatCompletionChunk] = self.client.chat.completions.create(
                model=self.chat_model,
                messages=_to_openai_messages(messages),
                stream=True,
            )

            for event in stream:
                if not event.choices:
                    continue

                delta: ChoiceDelta = event.choices[0].delta
                content = delta.content

                if content:
                    yield content

        except OpenAIError as exc:
            raise LLMUnavailableError(
                "OpenAI-compatible backend unavailable during streaming "
                f"using model {self.chat_model!r} at {self.base_url}: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if self.embed_model is None:
            raise ValueError("No embed model specified")
        if not texts:
            return []
        try:
            response = self.client.embeddings.create(
                model=self.embed_model,
                input=texts,
            )

            by_index = sorted(response.data, key=lambda item: item.index)
            return [list(item.embedding) for item in by_index]

        except OpenAIError as exc:
            raise LLMUnavailableError(
                "OpenAI-compatible backend unavailable during embedding "
                f"using model {self.embed_model!r} at {self.base_url}: {exc}"
            ) from exc

    def caption_image(self, image_bytes: bytes) -> str:
        if self.vision_model is None:
            raise LLMUnavailableError(
                "OpenAI-compatible vision model is not configured"
            )

        try:
            encoded = base64.b64encode(image_bytes).decode("ascii")
            mime_type = _detect_image_mime_type(image_bytes)

            messages: list[ChatCompletionMessageParam] = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this image.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{encoded}",
                            },
                        },
                    ],
                }
            ]

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=messages,
            )

            content = response.choices[0].message.content
            return content or ""

        except OpenAIError as exc:
            raise LLMUnavailableError(
                "OpenAI-compatible backend unavailable during vision "
                f"using model {self.vision_model!r} at {self.base_url}: {exc}"
            ) from exc
