"""The Anthropic messages wire format, with the prompt-cache breakpoints."""

from typing import Literal, cast

from anthropic import Omit, omit
from anthropic.types import (
    CacheControlEphemeralParam,
    MessageParam,
    TextBlockParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.tool_param import InputSchema

from llm.base import (
    Message,
    ToolSpec,
)
from llm.errors import LLMUnavailableError

CACHE_CONTROL: CacheControlEphemeralParam = {"type": "ephemeral"}

ImageMediaType = Literal["image/png", "image/jpeg", "image/gif", "image/webp"]

MIME_MAGIC: list[tuple[bytes, ImageMediaType]] = [
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
]


def detect_image_mime_type(image_bytes: bytes) -> ImageMediaType:
    for magic, mime in MIME_MAGIC:
        if image_bytes.startswith(magic):
            return mime
    # WebP: "RIFF....WEBP"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    raise LLMUnavailableError("Could not detect image MIME type")


def to_anthropic_tools(tools: list[ToolSpec]) -> list[ToolParam]:
    return [
        ToolParam(
            name=tool["name"],
            description=tool["description"],
            input_schema=cast("InputSchema", tool["parameters"]),
        )
        for tool in tools
    ]


def user_message(content: str) -> MessageParam:
    """A user turn, always in block form.

    The bare-string shorthand would be equivalent to the API, but only until
    this turn stops being the last one: ``mark_last_block`` has to attach the
    breakpoint to a *block*, so a string turn would be reshaped into a list the
    moment it was marked and reshaped back on the next request. Caching matches
    on exact bytes, so a prefix that changes representation by position never
    hits. Emitting one shape always is what makes the breakpoint pay off.
    """
    return {"role": "user", "content": [TextBlockParam(type="text", text=content)]}


def mark_last_block(messages: list[MessageParam]) -> None:
    """Put a cache breakpoint on the final content block, in place.

    Anthropic renders a request as tools → system → messages and caches by
    exact prefix match, so a breakpoint here covers everything before it. What
    reads it is the *next* deciding call: successive turns of a conversation
    share the whole prefix up to the previous turn's last block, so each turn
    pays a 0.1x read where it would otherwise pay 1x.

    The streaming call that answers a turn cannot read what that turn's
    deciding call wrote: `chat` sends tool definitions and `stream` does not,
    so the two prefixes differ at the first rendered byte. Passing the same
    tools to `stream` would fix the prefix but needs a policy for tool_use
    blocks arriving mid-answer — silently dropping them would lose a tool call
    the model asked for. Don't add `tools` there without deciding that.
    """
    if not messages:
        return
    content = messages[-1]["content"]
    # Never a bare string: user text is normalised to a block list on the way
    # in, precisely so that marking it does not also reshape it. See
    # ``user_message``.
    blocks = list(content) if not isinstance(content, str) else []
    if not blocks:
        return
    # Every block type reaching here (text, tool_use, tool_result) carries
    # cache_control; the cast keeps that fact from the type checker's blind
    # spot over the heterogeneous content union.
    cast("dict[str, object]", blocks[-1])["cache_control"] = CACHE_CONTROL
    messages[-1]["content"] = blocks


def to_anthropic_messages(
    messages: list[Message],
) -> tuple[list[TextBlockParam] | Omit, list[MessageParam]]:
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
            out.append(user_message(content))

    flush_results()
    mark_last_block(out)

    if not system_parts:
        return omit, out

    # One block, breakpoint on it: the tools render before the system prompt,
    # so this single entry covers the whole fixed preamble every chat request
    # shares. Both are byte-identical per deployment — keep them that way
    # (no timestamps, no per-request ids) or nothing here caches.
    system: list[TextBlockParam] = [
        TextBlockParam(
            type="text",
            text="\n\n".join(system_parts),
            cache_control=CACHE_CONTROL,
        )
    ]
    return system, out
