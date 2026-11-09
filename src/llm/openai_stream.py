"""Reads one streamed OpenAI-compatible tool-decision turn.

Providers cut a reply up in their own ways: reasoning arrives on one of three
channels, structured reasoning blocks arrive in pieces, tool calls arrive as
fragments, and some models write their calls into the answer text as markup. This
module folds all of that into the events and the terminal result the Buddy reads.
"""

import json
from collections.abc import Iterator
from typing import cast

from openai.types.chat import ChatCompletionChunk
from openai.types.chat.chat_completion_chunk import ChoiceDelta, ChoiceDeltaToolCall

from llm.base import (
    ChatResult,
    LLMChatStreamEvent,
    ReasoningDelta,
    TextDelta,
    ToolCall,
)
from llm.tool_call_recovery import GuardedTextCollector


def loads_arguments(raw: str) -> dict[str, object]:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else {}


def reasoning_text(delta: ChoiceDelta) -> list[str]:
    """Return provider-normalized reasoning fragments from one stream delta.

    OpenRouter currently exposes the plain-text channel as reasoning and
    structured blocks as reasoning_details. Older OpenAI-compatible backends
    use reasoning_content. Prefer the plain channel when several
    representations are present so the same fragment is not emitted twice.
    """
    for field in ("reasoning", "reasoning_content"):
        value = getattr(delta, field, None)
        if isinstance(value, str) and value.strip():
            return [value]

    details = getattr(delta, "reasoning_details", None)
    if not isinstance(details, list):
        return []

    fragments: list[str] = []
    for detail in cast("list[object]", details):
        if isinstance(detail, dict):
            detail_fields = cast("dict[str, object]", detail)
            text = detail_fields.get("text") or detail_fields.get("summary")
        else:
            text = getattr(detail, "text", None) or getattr(detail, "summary", None)
        if isinstance(text, str) and text.strip():
            fragments.append(text)
    return fragments


# Fields whose payload a provider streams in pieces; every other field is a
# per-block constant that the last delta carrying it wins.
_DETAIL_PAYLOAD_FIELDS = ("text", "summary", "data")


def reasoning_detail_fragments(delta: ChoiceDelta) -> list[dict[str, object]]:
    """Return the structured reasoning blocks carried by one stream delta.

    The plain-text channel ``reasoning_text`` normalizes is what gets *shown*;
    it is not what a provider accepts back. OpenRouter requires the structured
    sequence verbatim on the turn after a tool call (the hotfix #170 bug class),
    so the streaming path has to capture these as well — for signed or encrypted
    blocks they are the only preservable form.
    """
    details = getattr(delta, "reasoning_details", None)
    if not isinstance(details, list):
        return []

    fragments: list[dict[str, object]] = []
    for detail in cast("list[object]", details):
        if isinstance(detail, dict):
            fragments.append(dict(cast("dict[str, object]", detail)))
            continue
        dump = getattr(detail, "model_dump", None)
        if callable(dump):
            fragments.append(cast("dict[str, object]", dump()))
    return fragments


def _continues_block(block: dict[str, object], fragment: dict[str, object]) -> bool:
    """Whether ``fragment`` carries more of the block still open.

    ``index`` on its own is not a block identity: a provider may reuse it
    across channels (a ``reasoning.summary`` followed by a
    ``reasoning.encrypted`` payload), and merging on the index alone would
    overwrite ``type`` and produce one hybrid block that matches neither.
    A fragment continues the open block only while nothing contradicts it —
    a differing ``type``, or an ``id``/``index`` the two disagree on. Fields
    the fragment omits contradict nothing: providers that stream the payload
    without repeating the block header still belong to the block above.
    """
    for field in ("type", "id", "index"):
        block_value = block.get(field)
        fragment_value = fragment.get(field)
        if (
            block_value is not None
            and fragment_value is not None
            and block_value != fragment_value
        ):
            return False
    return True


def merge_reasoning_details(
    fragments: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Reassemble streamed reasoning fragments into whole blocks.

    A block's payload arrives across several deltas; the buffered response
    contains only the assembled block, and that is what the provider validates
    on the follow-up turn. Echoing the raw fragments back would replay one
    signed block as a dozen partial ones. Fragments are merged in arrival
    order into the block still open, which keeps two blocks separate whenever
    their identity differs — including when they share an ``index``.
    """
    blocks: list[dict[str, object]] = []

    for fragment in fragments:
        block = blocks[-1] if blocks else None
        if block is None or not _continues_block(block, fragment):
            blocks.append(dict(fragment))
            continue

        for field, value in fragment.items():
            if field in _DETAIL_PAYLOAD_FIELDS and isinstance(value, str):
                previous = block.get(field)
                block[field] = (previous if isinstance(previous, str) else "") + value
            elif value is not None:
                block[field] = value

    return blocks


class ChatStreamAccumulator:
    """Folds the chunks of one streamed tool-decision turn into events.

    :meth:`feed` turns each chunk into the live reasoning and answer deltas it makes
    safe to show; :meth:`finish` releases whatever the leak guard still held back and
    ends with the terminal :class:`ChatResult`, built the way the buffered ``chat``
    builds it: structured calls win, markup the model leaked into the content is
    recovered only when there are none.
    """

    def __init__(self) -> None:
        self._collector = GuardedTextCollector()
        self._detail_fragments: list[dict[str, object]] = []
        self._calls: dict[int, dict[str, object]] = {}

    def feed(self, chunk: ChatCompletionChunk) -> Iterator[LLMChatStreamEvent]:
        choice = chunk.choices[0] if chunk.choices else None
        delta = choice.delta if choice else None
        if delta is None:
            return

        for fragment in reasoning_text(delta):
            self._collector.add_reasoning(fragment)
            yield ReasoningDelta(fragment)

        self._detail_fragments.extend(reasoning_detail_fragments(delta))

        content = delta.content
        if content:
            visible = self._collector.add_text(content)
            if visible:
                yield TextDelta(visible)

        for fragment_call in delta.tool_calls or []:
            self._add_tool_call_fragment(fragment_call)

    def finish(self) -> Iterator[LLMChatStreamEvent]:
        """The held-back tail of the answer, then the terminal result."""
        structured = self._structured_calls()
        # Release whatever the leak guard was still holding back (partial-marker
        # tail of a plain answer). Never re-emit full_content here: when the
        # guard cut markup and recovery found nothing, that text must stay cut.
        tail = self._collector.flush_visible()
        if tail:
            yield TextDelta(tail)
        reasoning = self._collector.finish()[2]
        details = merge_reasoning_details(self._detail_fragments)

        if structured:
            # Structured calls won: buffered parity — content returned raw
            # (any markup leak stays in the result text for the model context).
            yield ChatResult(
                text=self._collector.full_content,
                tool_calls=structured,
                reasoning=reasoning or None,
                reasoning_details=details,
            )
            return

        recovered, cleaned, _ = self._collector.finish()
        if recovered:
            yield ChatResult(
                text=cleaned,
                tool_calls=recovered,
                reasoning=reasoning or None,
                reasoning_details=details,
            )
            return

        yield ChatResult(
            text=self._collector.full_content,
            tool_calls=[],
            reasoning=reasoning or None,
            reasoning_details=details,
        )

    def _add_tool_call_fragment(self, fragment: ChoiceDeltaToolCall) -> None:
        slot = self._calls.setdefault(
            fragment.index,
            {"id": "", "name": "", "arguments": ""},
        )
        if fragment.id:
            slot["id"] = str(fragment.id)
        if fragment.function and fragment.function.name:
            slot["name"] = str(fragment.function.name)
        if fragment.function and fragment.function.arguments:
            slot["arguments"] = str(slot["arguments"]) + str(
                fragment.function.arguments
            )

    def _structured_calls(self) -> list[ToolCall]:
        return [
            ToolCall(
                id=str(slot["id"]) or f"call_{index}",
                name=str(slot["name"]),
                arguments=loads_arguments(str(slot["arguments"])),
            )
            for index, slot in sorted(self._calls.items())
        ]
