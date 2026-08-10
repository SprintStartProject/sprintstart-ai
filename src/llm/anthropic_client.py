import base64
from collections.abc import Iterator
from typing import Literal, cast

from anthropic import Anthropic, APIError, Omit, omit
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.tool_param import InputSchema

from llm.base import ChatResult, LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError

_DEFAULT_MAX_TOKENS = 4096

ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

_MIME_MAGIC: list[tuple[bytes, ImageMediaType]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def _detect_image_mime_type(image_bytes: bytes) -> ImageMediaType:
    for magic, mime in _MIME_MAGIC:
        if image_bytes.startswith(magic):
            return mime
    # WebP: "RIFF....WEBP"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise LLMUnavailableError("Could not detect image MIME type")


def _to_anthropic_tools(tools: list[ToolSpec]) -> list[ToolParam]:
    return [
        ToolParam(
            name=tool["name"],
            description=tool["description"],
            input_schema=cast("InputSchema", tool["parameters"]),
        )
        for tool in tools
    ]


def _to_anthropic_messages(
    messages: list[Message],
) -> tuple[str | Omit, list[MessageParam]]:
    system_parts: list[str] = []
    out: list[MessageParam] = []
    pending_results: list[ToolResultBlockParam] = []

    def flush_results() -> None:
        if pending_results:
            out.append({"role": "user", "content": list(pending_results)})
            pending_results.clear()

    for message in messages:
        role = message["role"]
        content = message["content"]

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        if role == "tool":
            pending_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": message.get("tool_call_id", ""),
                    "content": content or "(no result)",
                }
            )
            continue

        flush_results()

        if role == "assistant":
            blocks: list[TextBlockParam | ToolUseBlockParam] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id,
                        "name": call.name,
                        "input": call.arguments,
                    }
                )
            if not blocks:
                blocks.append({"type": "text", "text": content or "(empty)"})
            out.append({"role": "assistant", "content": blocks})
        else:
            out.append({"role": "user", "content": content})

    flush_results()

    system: str | Omit = "\n\n".join(system_parts) if system_parts else omit
    return system, out


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        chat_model: str,
        vision_model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
    ) -> None:
        self.chat_model = chat_model
        self.vision_model = vision_model or chat_model
        self.max_tokens = max_tokens
        self.client = Anthropic(api_key=api_key, base_url=base_url)

    @property
    def model_name(self) -> str | None:
        return self.chat_model

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        system, converted = _to_anthropic_messages(messages)
        try:
            response = self.client.messages.create(
                model=self.chat_model,
                max_tokens=self.max_tokens,
                system=system,
                messages=converted,
                tools=_to_anthropic_tools(tools) if tools else omit,
            )
        except APIError as exc:
            raise LLMUnavailableError(
                "Anthropic backend unavailable during chat "
                f"using model {self.chat_model!r}: {exc}"
            ) from exc

        text_parts: list[str] = []
        calls: list[ToolCall] = []
        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append(
                    ToolCall(
                        id=block.id,
                        name=block.name,
                        arguments=block.input,
                    )
                )
        return ChatResult(text="".join(text_parts), tool_calls=calls)

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        system, converted = _to_anthropic_messages(messages)
        try:
            response = self.client.messages.create(
                model=self.chat_model,
                max_tokens=self.max_tokens,
                system=system,
                messages=converted,
                temperature=temperature if temperature is not None else omit,
            )
        except APIError as exc:
            raise LLMUnavailableError(
                "Anthropic backend unavailable during chat "
                f"using model {self.chat_model!r}: {exc}"
            ) from exc

        return "".join(block.text for block in response.content if block.type == "text")

    def stream(self, messages: list[Message]) -> Iterator[str]:
        system, converted = _to_anthropic_messages(messages)
        try:
            with self.client.messages.stream(
                model=self.chat_model,
                max_tokens=self.max_tokens,
                system=system,
                messages=converted,
            ) as stream:
                yield from stream.text_stream
        except APIError as exc:
            raise LLMUnavailableError(
                "Anthropic backend unavailable during streaming "
                f"using model {self.chat_model!r}: {exc}"
            ) from exc

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        raise LLMUnavailableError(
            "Anthropic does not provide an embeddings API. Configure EMBED_BACKEND "
            "to a provider that does (e.g. the OpenAI-compatible LiteLLM endpoint)."
        )

    def caption_image(self, image_bytes: bytes) -> str:
        mime_type = _detect_image_mime_type(image_bytes)
        encoded = base64.b64encode(image_bytes).decode("ascii")
        image_block: ImageBlockParam = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": encoded,
            },
        }
        messages: list[MessageParam] = [
            {
                "role": "user",
                "content": [
                    image_block,
                    {"type": "text", "text": "Describe this image."},
                ],
            }
        ]
        try:
            response = self.client.messages.create(
                model=self.vision_model,
                max_tokens=self.max_tokens,
                messages=messages,
            )
        except APIError as exc:
            raise LLMUnavailableError(
                "Anthropic backend unavailable during vision "
                f"using model {self.vision_model!r}: {exc}"
            ) from exc

        return "".join(block.text for block in response.content if block.type == "text")
