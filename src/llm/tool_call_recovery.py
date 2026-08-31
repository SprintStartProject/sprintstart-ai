"""Recover tool calls a model emitted as text instead of as structured calls.

Some models served through OpenAI-/Ollama-compatible endpoints (notably DeepSeek
via OpenRouter) emit tool calls in their own XML-ish markup *inside the message
content* rather than in the API's structured ``tool_calls`` field. The endpoint
then fails to lift them out, so the raw markup leaks into the assistant's visible
answer, e.g.::

    <｜｜DSML｜｜tool_calls><｜｜DSML｜｜invoke name="search_docs">
    <｜｜DSML｜｜parameter name="query" string="true">...</｜｜DSML｜｜parameter>
    </｜｜DSML｜｜invoke></｜｜DSML｜｜tool_calls>

The buddy agent drives itself off structured ``tool_calls``; when they arrive as
text like this the tool is never run and the markup is shown to the hire. This
module detects that markup, parses it back into :class:`ToolCall` objects, and
strips it from the text, so the agent loop can run the tool as if it had been
returned structurally.

The parser keys on the structural tokens (``invoke name=``, ``parameter name=``),
never on the surrounding delimiters, so it tolerates the various pipe/prefix
wrappers different models dress them in.
"""

import json
import re
from collections.abc import Iterable, Iterator
from uuid import uuid4

from llm.base import LLMStreamEvent, ReasoningDelta, TextDelta, ToolCall

# One ``invoke name="…"`` block and everything up to its matching close (or the
# end of the string, if the model never closed it).
_INVOKE_RE = re.compile(
    r'invoke\s+name="([^"]+)"(.*?)(?:</[^>]*\binvoke\b[^>]*>|$)',
    re.IGNORECASE | re.DOTALL,
)
# ``parameter name="…"…>value</…parameter>`` inside one invoke block.
_PARAM_RE = re.compile(
    r'parameter\s+name="([^"]+)"[^>]*>(.*?)</[^>]*\bparameter\b[^>]*>',
    re.IGNORECASE | re.DOTALL,
)
# The visible start of the leaked block, so the prose before it (if any) is kept.
_BLOCK_START_RE = re.compile(
    r'<[^>]*\b(?:tool_calls|invoke)\b|invoke\s+name="',
    re.IGNORECASE,
)
# Corroborating markup that tells a genuine leaked call apart from prose that
# merely happens to contain the substring ``invoke name="…"``.
_MARKUP_RE = re.compile(
    r"tool_calls|DSML|</[^>]*\binvoke\b|parameter\s+name=",
    re.IGNORECASE,
)


def _coerce(value: str) -> object:
    """Parse a parameter value as JSON when it looks like one, else keep the text."""
    stripped = value.strip()
    try:
        return json.loads(stripped)
    except (json.JSONDecodeError, ValueError):
        return stripped


def recover_tool_calls(content: str) -> tuple[list[ToolCall], str]:
    """Extract tool calls a model wrote into ``content`` as markup.

    Returns the recovered calls and the text with the markup removed. When no
    leaked call is present, returns ``([], content)`` unchanged — callers should
    only fall back to this when the structured ``tool_calls`` field was empty.
    """
    if not content or not _MARKUP_RE.search(content):
        return [], content

    invokes = list(_INVOKE_RE.finditer(content))
    if not invokes:
        return [], content

    calls = [
        ToolCall(
            id=f"call_{uuid4().hex}",
            name=match.group(1).strip(),
            arguments={
                name: _coerce(value)
                for name, value in _PARAM_RE.findall(match.group(2))
            },
        )
        for match in invokes
    ]

    cut = invokes[0].start()
    start = _BLOCK_START_RE.search(content)
    if start is not None and start.start() < cut:
        cut = start.start()
    return calls, content[:cut].rstrip()


# Substrings that identify leaked markup as it arrives. Kept lower-case and matched
# against a lower-cased buffer, since models vary the casing.
_STREAM_MARKERS = ("tool_calls", "dsml", "invoke name=", "parameter name=")

_LONGEST_MARKER = max(len(marker) for marker in _STREAM_MARKERS)

# How far past an unresolved `<` to keep waiting for a marker.
#
# The delimiters between a tag's `<` and its marker are arbitrary (`<｜｜DSML｜｜…`),
# so the `<` cannot be released until either a marker completes or enough ordinary
# text has followed to prove it was prose. An opening tag is short; well beyond this
# and a `<` is a comparison or a generic type.
_TAG_LOOKAHEAD = 32


def guard_stream(chunks: Iterable[str]) -> Iterator[str]:
    """Emit a model's streamed answer, cutting it off if leaked tool markup appears.

    ### Why the streaming path needs its own guard

    :func:`recover_tool_calls` protects the *tool* phase, where a whole message is in
    hand and a leaked call can be parsed back out and run. The visible answer is
    streamed instead (``AgentLoop.answer_stream``), and that phase has no tool loop at
    all -- so a model that decides mid-answer to search the docs cannot be given what
    it asked for, and its markup goes straight to the hire. That is exactly what a
    tutor saw: a friendly greeting followed by raw ``<｜｜DSML｜｜tool_calls>``.

    ### Cutting, not repairing

    Everything from the start of the markup is dropped and the stream ends there,
    matching what :func:`recover_tool_calls` does to the same markup in a whole
    message. **The model's intent is lost** -- the answer simply stops early. That is
    a worse answer than one where the tool ran, and a much better one than an answer
    with machine markup in the middle of it, which is the only alternative available
    without giving the answer phase its own tool loop.

    @param chunks: The raw stream from the model.
    @return: The same text, minus any leaked markup and anything after it.
    """
    pending = ""

    for chunk in chunks:
        if not chunk:
            continue
        pending += chunk

        cut = _markup_start(pending)
        if cut is not None:
            head = pending[:cut].rstrip()
            if head:
                yield head
            return

        keep = _unsafe_tail_len(pending)
        if keep < len(pending):
            emit, pending = (
                pending[: len(pending) - keep],
                pending[len(pending) - keep :],
            )
            if emit:
                yield emit

    if pending:
        yield pending


def guard_event_stream(events: Iterable[LLMStreamEvent]) -> Iterator[LLMStreamEvent]:
    """Apply :func:`guard_stream` semantics to answer deltas, preserving reasoning.

    A reasoning delta is a boundary between runs of visible text. Flush any pending
    visible text before forwarding it so event order is unchanged; leaked tool markup
    is only meaningful inside the visible answer stream.
    """
    pending = ""

    for event in events:
        if isinstance(event, ReasoningDelta):
            if pending:
                yield TextDelta(pending)
                pending = ""
            yield event
            continue

        if not event.text:
            continue
        pending += event.text

        cut = _markup_start(pending)
        if cut is not None:
            head = pending[:cut].rstrip()
            if head:
                yield TextDelta(head)
            return

        keep = _unsafe_tail_len(pending)
        if keep < len(pending):
            emit, pending = (
                pending[: len(pending) - keep],
                pending[len(pending) - keep :],
            )
            if emit:
                yield TextDelta(emit)

    if pending:
        yield TextDelta(pending)


def _unsafe_tail_len(pending: str) -> int:
    """How much of the tail must be withheld because it could still become markup.

    Withholding a fixed-size tail instead would stop short answers streaming at
    all — anything shorter than the window arrives in one lump at the end, which the
    client tests caught. So this withholds only what is genuinely ambiguous, and
    ordinary prose flows at full speed.

    Three things are ambiguous:

    - a partial marker at the very end (``…parameter nam``);
    - an unresolved ``<``, since what sits between it and a marker is arbitrary — but
      only for [_TAG_LOOKAHEAD] characters, after which it was prose;
    - trailing whitespace, which is held so that the space before a leaked block is
      never emitted. Once text is streamed it cannot be taken back, and the cut
      itself can only trim what is still in hand.
    """
    lowered = pending.lower()
    candidates = [0]

    # A candidate, never an early return. `<｜｜D` ends in a one-character prefix of
    # `dsml`, and returning that 1 would release the `<｜｜` in front of it — which is
    # precisely the stray opening this function exists to keep back. The longest hold
    # wins, not the first one found.
    for length in range(min(len(pending), _LONGEST_MARKER), 0, -1):
        if any(marker.startswith(lowered[-length:]) for marker in _STREAM_MARKERS):
            candidates.append(length)
            break

    opening = pending.rfind("<")
    if opening != -1 and len(pending) - opening <= _TAG_LOOKAHEAD:
        # Extended back over the whitespace in front of the tag. The cut trims what is
        # still in hand, so a space released before the `<` arrived can never be taken
        # back -- and "Here you go. " with a dangling space is the visible remains of a
        # block that was supposed to vanish completely.
        while opening > 0 and pending[opening - 1].isspace():
            opening -= 1
        candidates.append(len(pending) - opening)

    trailing_space = len(pending) - len(pending.rstrip())
    if trailing_space <= _TAG_LOOKAHEAD:
        candidates.append(trailing_space)

    return max(candidates)


def _markup_start(buffer: str) -> int | None:
    """Where leaked markup begins in ``buffer``, or None when it holds no marker.

    The cut is taken at the ``<`` that opens the tag rather than at the marker itself,
    so the delimiters a model wraps its markup in go too instead of being left behind
    as a stray ``<｜｜``.
    """
    lowered = buffer.lower()
    hits = [
        index for index in (lowered.find(m) for m in _STREAM_MARKERS) if index != -1
    ]
    if not hits:
        return None

    marker = min(hits)
    opening = buffer.rfind("<", 0, marker)
    return opening if opening != -1 else marker


class GuardedTextCollector:
    """Accumulate a streamed tool-decision turn behind the leak guard.

    ### Why the decision turn needs both guarding and recovery

    A decision turn (``chat_stream``) can end two ways for a model that leaks
    its tool calls as markup: the calls may arrive structurally in
    ``tool_calls`` while the markup still sits in the content, or the calls may
    exist *only* as markup. The visible answer must never show the markup
    either way — that is what the :func:`guard_stream` machinery does — but
    unlike the plain answer stream, a decision turn can still *act* on the
    markup: everything seen is accumulated unguarded in full, so
    :func:`recover_tool_calls` can lift the calls out of the complete block at
    stream end, exactly as the buffered ``chat`` path does.

    ### Emission contract

    ``add_text`` returns the text that may be shown now (possibly empty — the
    tail is withheld while it could still turn out to be markup, per
    :func:`_unsafe_tail_len`). Reasoning is a boundary between visible runs, so
    the caller flushes held-back text before yielding a reasoning delta via
    :meth:`flush_visible`; ``finish`` releases whatever remains.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._full = ""
        self._reasoning: list[str] = []
        self._cut = False

    def add_reasoning(self, text: str) -> None:
        """Record a plain reasoning fragment for the terminal result."""
        if text.strip():
            self._reasoning.append(text)

    def flush_visible(self) -> str:
        """Release held-back visible text; called at reasoning boundaries."""
        flushed, self._pending = self._pending, ""
        return flushed

    def add_text(self, text: str) -> str:
        """Feed one content chunk; returns the text that may be emitted now.

        Full content keeps accumulating even after the guard has cut the
        visible stream — the remainder of a leaked block is exactly what
        :func:`recover_tool_calls` needs to parse the call out of.
        """
        if not text:
            return ""
        self._full += text
        if self._cut:
            return ""

        self._pending += text
        cut = _markup_start(self._pending)
        if cut is not None:
            self._cut = True
            head = self._pending[:cut].rstrip()
            self._pending = ""
            return head

        keep = _unsafe_tail_len(self._pending)
        if keep < len(self._pending):
            emit = self._pending[: len(self._pending) - keep]
            self._pending = self._pending[len(self._pending) - keep :]
            return emit
        return ""

    @property
    def full_content(self) -> str:
        """Everything the model streamed as content, guard cuts notwithstanding.

        The buffered path returns the message content unchanged when structured
        tool calls exist (only the recovered fallback cleans the text), so the
        terminal result needs the raw full text for that case.
        """
        return self._full

    def finish(self) -> tuple[list[ToolCall], str, str]:
        """End the stream; returns ``(recovered_calls, cleaned_text, reasoning)``.

        ``cleaned_text`` is the full content minus any recovered markup — the
        same string the buffered path would have returned, and (when markup was
        cut) identical to what was actually shown. ``recovered_calls`` is empty
        unless the model wrote its calls as markup instead of structured ones.
        """
        self.flush_visible()
        calls, cleaned = recover_tool_calls(self._full)
        return calls, cleaned, "".join(self._reasoning)


__all__ = [
    "GuardedTextCollector",
    "guard_event_stream",
    "guard_stream",
    "recover_tool_calls",
]
