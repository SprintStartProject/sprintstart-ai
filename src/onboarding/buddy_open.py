"""Open a buddy visit: greet the hire.

A visit opens like walking up to a mentor: the buddy recalls what it knows about
the hire (its durable memory), notes their current state, and greets them with the
one thing worth saying.

Stateless like every onboarding endpoint: the backend supplies the memory, the
recent (not-yet-remembered) messages and a snapshot of the hire's state, and
persists the greeting this returns.

⚠️ **This call must not write the durable memory note.** Composing it here puts it
in the same model call as the greeting, so a hire's memory is written while the
model is busy greeting them. Folding is ``onboarding.buddy_compact``, on its own
endpoint, run by the backend when nobody is waiting.

⚠️ **The greeting comes first and is streamed.** Nothing the hire never sees may be
generated ahead of it — strict JSON cannot be streamed as prose, which is why this
uses markers. ⚠️ A marker arrives one chunk at a time, so a partial one must be
held back as a candidate, never matched early or emitted.
"""

import json
from collections.abc import Iterator
from typing import cast

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError

_FALLBACK_GREETING = "Welcome back! How can I help with your onboarding today?"


def _format_recent(recent: list[Message]) -> str:
    lines = [
        f"{m['role']}: {m.get('content') or ''}" for m in recent if m.get("content")
    ]
    return "\n".join(lines) if lines else "(nothing since the last memory update)"


_ACTION_MARKER = "<<<ACTION>>>"

_STREAM_SYSTEM = (
    "You are a warm, perceptive onboarding mentor greeting a new hire as they open "
    "the chat. You keep a private, durable memory note about this hire, and you "
    "speak to them directly.\n"
    "You are given: your MEMORY of the hire (may be empty on the first visit), the "
    "RECENT conversation since you last updated that memory (may be empty), and the "
    "hire's current STATE (their work in flight, tasks, competencies). What the "
    "state contains depends on the hire's role -- describe only what is actually "
    "there, and never assume they write code.\n"
    "Write your reply in exactly two parts, in this order, with nothing before the "
    "first part:\n"
    "PART 1 -- the greeting, as plain prose with no label and no quotes: a short, "
    "warm, first-person opener (2-4 sentences) that greets the hire and proactively "
    "says the one thing most worth saying right now, grounded in the memory and the "
    "current state (work waiting on somebody else, something of theirs that landed "
    "and is worth celebrating, a stall, an open thread from last time). Be specific, "
    "not generic. Never invent facts that are not in the memory or the state. The "
    "hire reads this as you type it, so it must come first.\n"
    f"PART 2 -- the line {_ACTION_MARKER} on its own, then ONE suggested next step "
    'as JSON {"label": short button text, "question": the message to send when the '
    "hire clicks it}, or the word none when nothing fits.\n"
    # The note is rewritten elsewhere, by a call nobody is waiting on. Saying so stops
    # a model that has seen the memory from helpfully offering an updated one, which
    # would land in the greeting the hire reads.
    "Do not rewrite or restate your memory note. It is maintained separately; here "
    "you only read from it."
)


def stream_session(
    memory: str | None,
    recent: list[Message],
    state: str,
    llm: LLMClient,
) -> Iterator[dict[str, object]]:
    """Stream the greeting as the model writes it.

    ### Why the greeting comes first

    Opening the buddy took about thirty seconds, and the reason was ordering rather
    than model speed: the version this replaced asked for strict JSON whose **first**
    field was a memory note of up to 200 words that **the hire never sees**, with the
    2-4 sentence greeting after it.

    ⚠️ **Strict JSON cannot be streamed as prose** -- the first tokens are
    ``{"memory": "`` -- so this call uses a marker instead and puts the visible part
    first.

    ### Degrading

    A model that ignores the format and just writes prose yields all of it as the
    greeting, which is harmless: there is nothing behind the marker but a suggestion.
    An unavailable model yields the plain welcome -- opening a visit must never fail
    the page.

    @param memory: The mentor's durable note, or None on a first visit. Read from,
        never rewritten here -- see the module docstring.
    @param recent: The window since the memory was last updated.
    @param state: A snapshot of the hire's current state.
    @return: ``token`` events carrying the greeting as it arrives, then one terminal
        ``done`` carrying the whole greeting and any action.
    """
    prompt = [
        Message(role="system", content=_STREAM_SYSTEM),
        Message(
            role="user",
            content=(
                f"MEMORY:\n{memory or '(no memory yet -- first visit)'}\n\n"
                "RECENT conversation since the last memory update:\n"
                f"{_format_recent(recent)}\n\n"
                f"STATE (current):\n{state or '(no state available)'}\n\n"
                "Write the two parts."
            ),
        ),
    ]

    greeting = ""
    tail = ""
    pending = ""
    in_greeting = True

    try:
        for chunk in llm.stream(prompt):
            if not chunk:
                continue
            if not in_greeting:
                tail += chunk
                continue
            pending += chunk
            cut = pending.find(_ACTION_MARKER)
            if cut != -1:
                # Trailing whitespace before the marker is formatting, not greeting.
                head = pending[:cut].rstrip()
                if head:
                    greeting += head
                    yield {"type": "token", "content": head}
                tail = pending[cut + len(_ACTION_MARKER) :]
                pending = ""
                in_greeting = False
                continue
            # ⚠️ A suffix that could still grow into the marker is held back, never
            # emitted and never treated as the end of the greeting: "<<<ACT" is both a
            # partial marker and a legitimate six characters of prose, and only the
            # next chunk decides which. Holding back only that suffix is what keeps a
            # short greeting streaming instead of waiting for a fixed-size buffer.
            keep = _held_back_len(pending)
            if keep < len(pending):
                emit, pending = (
                    pending[: len(pending) - keep],
                    pending[len(pending) - keep :],
                )
                greeting += emit
                yield {"type": "token", "content": emit}
    except LLMUnavailableError:
        yield {"type": "done", "greeting": _FALLBACK_GREETING, "action": None}
        return

    # Whatever is still held back was never a marker after all, so it is prose.
    if in_greeting and pending:
        greeting += pending
        yield {"type": "token", "content": pending}

    label, question = _read_action_tail(tail)
    yield {
        "type": "done",
        # ⚠️ Byte-identical to the concatenated tokens, deliberately. The client renders
        # the tokens and the caller persists this; if they differed, the message a hire
        # watched arrive would not be the one they see after a reload.
        "greeting": greeting if greeting.strip() else _FALLBACK_GREETING,
        "action": (
            {"label": label, "question": question} if label and question else None
        ),
    }


def _held_back_len(text: str) -> int:
    """How many trailing characters could still turn out to be the action marker."""
    for size in range(min(len(_ACTION_MARKER) - 1, len(text)), 0, -1):
        if _ACTION_MARKER.startswith(text[-size:]):
            return size
    return 0


def _read_action_tail(tail: str) -> tuple[str | None, str | None]:
    """Read the suggested next step from everything after the action marker."""
    action = _loads_object(tail)
    if action is None:
        return None, None
    return _read_action(action)


def _read_action(action: object) -> tuple[str | None, str | None]:
    if not isinstance(action, dict):
        return None, None
    action_dict = cast("dict[str, object]", action)
    label = action_dict.get("label")
    question = action_dict.get("question")
    has_label = isinstance(label, str) and label.strip()
    has_question = isinstance(question, str) and question.strip()
    if has_label and has_question:
        return cast("str", label), cast("str", question)
    return None, None


def _loads_object(raw: str) -> dict[str, object] | None:
    # Models sometimes wrap JSON in prose or code fences; take the outermost object.
    text = raw.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed: object = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return cast("dict[str, object]", parsed) if isinstance(parsed, dict) else None


__all__ = ["stream_session"]
