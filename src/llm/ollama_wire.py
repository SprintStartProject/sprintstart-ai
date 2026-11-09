"""The Ollama chat wire format: our messages and tools in, a ChatResult out."""

import logging
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

import ollama

from llm.base import (
    ChatResult,
    Message,
    ToolCall,
    ToolSpec,
)
from llm.tool_call_recovery import (
    recover_tool_calls,
)

logger = logging.getLogger(__name__)


def to_ollama_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Internal messages in Ollama's wire shape.

    Every path that sends messages to the daemon goes through this. Our
    ``Message`` carries ``ToolCall`` objects and a ``name``/``tool_call_id``
    pair; Ollama expects ``{"function": {...}}`` and ``tool_name``. Handing it
    the internal shape fails validation or silently loses the tool
    association, so no caller may skip the conversion.
    """
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


def to_ollama_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
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


def from_ollama_response(response: ollama.ChatResponse) -> ChatResult:
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
