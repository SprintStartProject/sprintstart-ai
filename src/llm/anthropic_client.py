import base64
import json
from collections.abc import Iterator
from typing import Any, Literal, cast

from anthropic import NOT_GIVEN, Anthropic, APIError, Omit, omit
from anthropic.types import (
    CacheControlEphemeralParam,
    ImageBlockParam,
    MessageParam,
    TextBlockParam,
    ThinkingConfigEnabledParam,
    ToolParam,
    ToolResultBlockParam,
    ToolUseBlockParam,
)
from anthropic.types.tool_param import InputSchema

from llm.base import (
    ChatResult,
    LLMChatStreamEvent,
    LLMClient,
    LLMStreamEvent,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from llm.errors import LLMUnavailableError

_DEFAULT_MAX_TOKENS = 4096
_MIN_THINKING_BUDGET_TOKENS = 1024

_CACHE_CONTROL: CacheControlEphemeralParam = {"type": "ephemeral"}

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


def _user_message(content: str) -> MessageParam:
    """A user turn, always in block form.

    The bare-string shorthand would be equivalent to the API, but only until
    this turn stops being the last one: ``_mark_last_block`` has to attach the
    breakpoint to a *block*, so a string turn would be reshaped into a list the
    moment it was marked and reshaped back on the next request. Caching matches
    on exact bytes, so a prefix that changes representation by position never
    hits. Emitting one shape always is what makes the breakpoint pay off.
    """
    return {"role": "user", "content": [TextBlockParam(type="text", text=content)]}


def _mark_last_block(messages: list[MessageParam]) -> None:
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
    # ``_user_message``.
    blocks = list(content) if not isinstance(content, str) else []
    if not blocks:
        return
    # Every block type reaching here (text, tool_use, tool_result) carries
    # cache_control; the cast keeps that fact from the type checker's blind
    # spot over the heterogeneous content union.
    cast("dict[str, object]", blocks[-1])["cache_control"] = _CACHE_CONTROL
    messages[-1]["content"] = blocks


def _to_anthropic_messages(
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
            out.append(_user_message(content))

    flush_results()
    _mark_last_block(out)

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
            cache_control=_CACHE_CONTROL,
        )
    ]
    return system, out


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        chat_model: str,
        vision_model: str | None = None,
        base_url: str | None = None,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        thinking_budget_tokens: int | None = None,
        timeout: float | None = None,
    ) -> None:
        if thinking_budget_tokens is not None:
            if thinking_budget_tokens < _MIN_THINKING_BUDGET_TOKENS:
                raise ValueError(
                    "Anthropic thinking budget must be at least "
                    f"{_MIN_THINKING_BUDGET_TOKENS} tokens"
                )
            if thinking_budget_tokens >= max_tokens:
                raise ValueError(
                    "Anthropic thinking budget must be lower than max_tokens"
                )
        self.chat_model = chat_model
        self.vision_model = vision_model or chat_model
        self.max_tokens = max_tokens
        self.thinking_budget_tokens = thinking_budget_tokens
        self.client = Anthropic(
            api_key=api_key,
            base_url=base_url,
            # NOT_GIVEN, not None: to these SDKs ``timeout=None`` means "wait
            # forever", so an unconfigured timeout has to leave the parameter
            # unset for the SDK's own default to apply.
            timeout=NOT_GIVEN if timeout is None else timeout,
        )

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

    def chat_stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Iterator[LLMChatStreamEvent]:
        """Stream one tool-decision turn with live reasoning and answer deltas.

        Mirrors the buffered tool-mode ``chat`` parameter shape: tools are sent,
        thinking is not. Extended thinking plus tools would require signed
        thinking blocks to round-trip through the conversation history — the
        plain-answer ``stream`` path has no tool loop, so its history shape
        cannot carry them. Yields thinking deltas when a thinking budget is
        configured, and tool calls accumulate from ``input_json_delta``.
        """
        system, converted = _to_anthropic_messages(messages)
        # Same guard as `stream`: budget None must leave the parameter unset
        # (None disables the timeout/feature rather than meaning "default").
        # Tools additionally force thinking off: the assistant turn would come
        # back with thinking blocks that the buffered `chat` history shape
        # cannot carry signed into the follow-up request, which the API rejects.
        thinking: ThinkingConfigEnabledParam | Omit = (
            {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            if self.thinking_budget_tokens is not None and not tools
            else omit
        )
        try:
            with self.client.messages.stream(
                model=self.chat_model,
                max_tokens=self.max_tokens,
                system=system,
                messages=converted,
                tools=_to_anthropic_tools(tools) if tools else omit,
                thinking=thinking,
            ) as stream:
                text_parts: list[str] = []
                calls: list[ToolCall] = []
                current_tool: tuple[str, str] | None = None
                tool_input: list[str] = []
                for event in stream:
                    if event.type == "content_block_start":
                        block = cast("Any", event).content_block
                        if block.type == "tool_use":
                            current_tool = (block.id, block.name)
                            tool_input = []
                    elif event.type == "content_block_delta":
                        delta = cast("Any", event).delta
                        if delta.type == "thinking_delta":
                            if delta.thinking.strip():
                                yield ReasoningDelta(delta.thinking)
                        elif delta.type == "text_delta" and delta.text:
                            text_parts.append(delta.text)
                            yield TextDelta(delta.text)
                        elif delta.type == "input_json_delta" and current_tool:
                            tool_input.append(delta.partial_json)
                    elif (
                        event.type == "content_block_stop" and current_tool is not None
                    ):
                        raw = "".join(tool_input) or "{}"
                        try:
                            parsed = json.loads(raw)
                        except json.JSONDecodeError:
                            parsed = {}
                        calls.append(
                            ToolCall(
                                id=current_tool[0],
                                name=current_tool[1],
                                arguments=cast("dict[str, object]", parsed),
                            )
                        )
                        current_tool = None
                        tool_input = []
        except APIError as exc:
            raise LLMUnavailableError(
                "Anthropic backend unavailable during streaming chat "
                f"using model {self.chat_model!r}: {exc}"
            ) from exc

        yield ChatResult(
            text="".join(text_parts),
            tool_calls=calls,
        )

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

    def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
        system, converted = _to_anthropic_messages(messages)
        thinking: ThinkingConfigEnabledParam | Omit = (
            {
                "type": "enabled",
                "budget_tokens": self.thinking_budget_tokens,
            }
            if self.thinking_budget_tokens is not None
            else omit
        )
        try:
            with self.client.messages.stream(
                model=self.chat_model,
                max_tokens=self.max_tokens,
                system=system,
                messages=converted,
                thinking=thinking,
            ) as stream:
                for event in stream:
                    if event.type != "content_block_delta":
                        continue
                    delta = event.delta
                    if delta.type == "thinking_delta":
                        if delta.thinking.strip():
                            yield ReasoningDelta(delta.thinking)
                    elif delta.type == "text_delta" and delta.text:
                        yield TextDelta(delta.text)
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
