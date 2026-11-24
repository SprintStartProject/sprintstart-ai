"""The OpenAI chat wire format: our messages and tools in, reasoning out of replies."""

import json
from typing import Any, cast

from openai.types.chat import (
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessage,
    ChatCompletionMessageParam,
    ChatCompletionSystemMessageParam,
    ChatCompletionToolMessageParam,
    ChatCompletionToolParam,
    ChatCompletionUserMessageParam,
)

from llm.base import (
    Message,
    ToolSpec,
)
from llm.errors import LLMUnavailableError


def to_openai_tools(tools: list[ToolSpec]) -> list[ChatCompletionToolParam]:
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


def normalize_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")

    if not normalized.endswith("/v1"):
        normalized = f"{normalized}/v1"

    return normalized


def to_openai_messages(messages: list[Message]) -> list[ChatCompletionMessageParam]:
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
            # OpenRouter documents reasoning and reasoning_details as
            # alternative preservation formats and requires the structured
            # sequence for signed/encrypted blocks. Prefer it verbatim and fall
            # back to plaintext only when no structured details are present.
            reasoning_details = message.get("reasoning_details")
            if reasoning_details:
                cast("Any", assistant_message)["reasoning_details"] = reasoning_details
            else:
                reasoning = message.get("reasoning")
                if reasoning:
                    cast("Any", assistant_message)["reasoning"] = reasoning
            openai_messages.append(assistant_message)
        else:
            user_message: ChatCompletionUserMessageParam = {
                "role": "user",
                "content": content,
            }
            openai_messages.append(user_message)

    return openai_messages


def detect_image_mime_type(image_bytes: bytes) -> str:
    image_type = sniff_image_type(image_bytes)

    if image_type is None:
        raise LLMUnavailableError("Could not detect image MIME type")

    return f"image/{image_type}"


def sniff_image_type(image_bytes: bytes) -> str | None:
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


def response_reasoning(
    message: ChatCompletionMessage,
) -> tuple[str | None, list[dict[str, object]]]:
    """Extract the exact reasoning context OpenRouter requires after tool use."""
    message_fields = cast("dict[str, object]", message.model_dump())
    raw_reasoning = message_fields.get("reasoning")
    if not isinstance(raw_reasoning, str) or not raw_reasoning.strip():
        # Legacy OpenAI-compatible backends (DeepSeek-R1, Qwen) expose the
        # channel as reasoning_content, mirroring the streaming normalization.
        raw_reasoning = message_fields.get("reasoning_content")
    reasoning = (
        raw_reasoning
        if isinstance(raw_reasoning, str) and raw_reasoning.strip()
        else None
    )

    raw_details = message_fields.get("reasoning_details")
    if not isinstance(raw_details, list):
        return reasoning, []

    details: list[dict[str, object]] = []
    for detail in cast("list[object]", raw_details):
        if isinstance(detail, dict):
            details.append(cast("dict[str, object]", detail))
    return reasoning, details
