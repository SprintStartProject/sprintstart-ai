"""The opening stream is pinned event for event, across every way a reply can be cut up.

The marker machinery decides what a hire watches arrive, and its edge cases are chunk
boundaries: a marker split anywhere, a prefix of it that is really prose, empty and
non-text chunks, a model that fails halfway. Each reply is replayed in every chunk
size, so a change to the machinery shows up as a changed event. After a deliberate
change rerun with ``UPDATE_PROMPT_GOLDENS=1`` and review the diff.
"""

import json
import os
from collections.abc import Iterator
from pathlib import Path

from buddy.opening import stream_session
from llm.base import LLMClient, LLMStreamEvent, Message, ReasoningDelta, TextDelta
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient
from tests.support.golden import assert_matches_golden

_EVENTS = Path(__file__).parent / "opening_events.txt"
_ACTION = '{"label": "What next?", "question": "What should I work on?"}'
_REPLIES = {
    "plain": "Hello there, welcome back.",
    "action": f"Hi Sam.\n<<<ACTION>>>\n{_ACTION}",
    "no-action": "Hi Sam.\n<<<ACTION>>>\nnone",
    "marker-first": f"<<<ACTION>>>{_ACTION}",
    "trailing-space": f"Hi Sam.   \n\n<<<ACTION>>>{_ACTION}",
    "prefix-prose": "Use <<<ACT as a label, then <<<ACTIONS carry on.",
    "ends-on-prefix": "Fine <<<ACTI",
    "fenced-action": f"Hi.\n<<<ACTION>>>\n```json\n{_ACTION}\n```",
    "bad-json": "Hi.\n<<<ACTION>>>\n{not json",
    "half-action": 'Hi.\n<<<ACTION>>>\n{"label": "only a label"}',
    "array-action": "Hi.\n<<<ACTION>>>\n[1, 2]",
    "blank": "   ",
    "empty": "",
    "two-markers": f"A<<<ACTION>>>{_ACTION}<<<ACTION>>>B",
}


class _ChunkedLLM(StubLLMClient):
    """Replays scripted events, then optionally fails; remembers the prompt it got."""

    def __init__(
        self, items: list[LLMStreamEvent], then: Exception | None = None
    ) -> None:
        super().__init__()
        self._items = items
        self._then = then
        self.prompt: list[Message] = []

    def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
        self.prompt = list(messages)
        yield from self._items
        if self._then is not None:
            raise self._then


def _cut(text: str, size: int) -> list[LLMStreamEvent]:
    return [TextDelta(text[i : i + size]) for i in range(0, len(text), size)]


def _run(llm: LLMClient, *, team_mode: bool = False) -> str:
    events = list(stream_session(None, [], "", llm, team_mode=team_mode))
    return json.dumps(events, sort_keys=True)


def _lines() -> str:
    lines = []
    for name, reply in _REPLIES.items():
        for size in (*range(1, 14), len(reply) or 1):
            for team in (False, True):
                llm = _ChunkedLLM(_cut(reply, size))
                lines.append(
                    f"{name} size={size} team={team} {_run(llm, team_mode=team)}\n"
                )
    noise: list[LLMStreamEvent] = [
        TextDelta(""),
        ReasoningDelta("thinking"),
        TextDelta("Hi"),
        ReasoningDelta("thinking"),
        TextDelta("!"),
    ]
    lines.append(f"noise {_run(_ChunkedLLM(noise))}\n")
    failing = _ChunkedLLM(_cut("Hello <<<AC", 3), then=LLMUnavailableError("down"))
    lines.append(f"failing-midway {_run(failing)}\n")
    lines.append(
        f"failing-first {_run(_ChunkedLLM([], then=LLMUnavailableError('down')))}\n"
    )
    return "".join(lines)


def test_every_chunking_of_every_reply_matches_the_golden() -> None:
    actual = _lines()
    if os.environ.get("UPDATE_PROMPT_GOLDENS") == "1":
        _EVENTS.write_text(actual, encoding="utf-8", newline="\n")
    assert actual == _EVENTS.read_text(encoding="utf-8")


def _prompt_of(
    name: str,
    memory: str | None,
    recent: list[Message],
    state: str,
    *,
    team_mode: bool = False,
) -> list[Message]:
    llm = _ChunkedLLM([])
    list(stream_session(memory, recent, state, llm, team_mode=team_mode))
    return [
        {"role": f"{name}:{m['role']}", "content": m.get("content") or ""}
        for m in llm.prompt
    ]


def test_the_prompts_match_the_golden() -> None:
    recent: list[Message] = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": ""},
        {"role": "assistant", "content": "hello"},
    ]
    prompts = [
        *_prompt_of("hire", "the note", recent, "state"),
        *_prompt_of("team", "the note", recent, "state", team_mode=True),
        *_prompt_of("hire-empty", None, [], ""),
    ]
    assert_matches_golden("buddy_opening", prompts)
