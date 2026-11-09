import base64
import json
from collections.abc import Iterator
from typing import Any, cast

from anthropic import NOT_GIVEN, Anthropic, APIError, Omit, omit
from anthropic.types import (
    ImageBlockParam,
    MessageParam,
    ThinkingConfigEnabledParam,
)

from llm.anthropic_wire import (
    detect_image_mime_type,
    to_anthropic_messages,
    to_anthropic_tools,
)
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
        system, converted = to_anthropic_messages(messages)
        try:
            response = self.client.messages.create(
                model=self.chat_model,
                max_tokens=self.max_tokens,
                system=system,
                messages=converted,
                tools=to_anthropic_tools(tools) if tools else omit,
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
        system, converted = to_anthropic_messages(messages)
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
                tools=to_anthropic_tools(tools) if tools else omit,
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
        system, converted = to_anthropic_messages(messages)
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
        system, converted = to_anthropic_messages(messages)
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
        mime_type = detect_image_mime_type(image_bytes)
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
