from collections.abc import Iterator

from llm.base import (
    ChatResult,
    LLMChatStreamEvent,
    LLMClient,
    LLMStreamEvent,
    Message,
    ToolSpec,
)


class SplitLLMClient(LLMClient):
    def __init__(self, chat: LLMClient, embed: LLMClient) -> None:
        self._chat = chat
        self._embed = embed

    @property
    def model_name(self) -> str | None:
        return self._chat.model_name

    def chat(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> ChatResult:
        return self._chat.chat(messages, tools)

    def chat_stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Iterator[LLMChatStreamEvent]:
        return self._chat.chat_stream(messages, tools)

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        return self._chat.generate(messages, temperature=temperature)

    def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
        return self._chat.stream(messages)

    def caption_image(self, image_bytes: bytes) -> str:
        return self._chat.caption_image(image_bytes)

    def embed(self, text: str) -> list[float]:
        return self._embed.embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return self._embed.embed_batch(texts)
