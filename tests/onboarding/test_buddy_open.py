"""Tests for opening a buddy visit: the streamed greeting, and nothing else.

⚠️ The memory fold used to happen here too, from the same model call. These tests
kept every rule the marker machinery needs — it now guards ``<<<ACTION>>>`` rather
than ``<<<MEMORY>>>``, because the problem it solves is unchanged: a marker arrives
one chunk at a time, and a partial one is indistinguishable from prose until the
next chunk lands.
"""

from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from onboarding.buddy_open import stream_session

_ACTION = '{"label": "What next?", "question": "What should I work on?"}'


class _StubLLM(LLMClient):
    def __init__(self, reply: str | Exception) -> None:
        self._reply = reply
        self.last_prompt: list[Message] | None = None

    def generate(self, messages, *, temperature=None):  # type: ignore[override]
        self.last_prompt = messages
        if isinstance(self._reply, Exception):
            raise self._reply
        return self._reply

    def chat(self, messages, tools=None):  # pragma: no cover - unused
        raise NotImplementedError

    def stream(self, messages):  # pragma: no cover - unused
        raise NotImplementedError

    def embed(self, text):  # pragma: no cover - unused
        raise NotImplementedError

    def embed_batch(self, texts):  # pragma: no cover - unused
        raise NotImplementedError

    def caption_image(self, image_bytes):  # pragma: no cover - unused
        raise NotImplementedError

    @property
    def model_name(self):  # pragma: no cover - unused
        return "stub"


class _StreamingStubLLM(_StubLLM):
    """Replays a scripted list of chunks, so chunk boundaries can be chosen per test."""

    def __init__(self, chunks: list[str] | Exception) -> None:
        super().__init__("")
        self._chunks = chunks

    def stream(self, messages):  # type: ignore[override]
        self.last_prompt = messages
        if isinstance(self._chunks, Exception):
            raise self._chunks
        yield from self._chunks


def _tokens(events: list[dict[str, object]]) -> str:
    return "".join(str(e["content"]) for e in events if e["type"] == "token")


def _done(events: list[dict[str, object]]) -> dict[str, object]:
    assert events[-1]["type"] == "done"
    return events[-1]


def test_streams_the_greeting_and_carries_the_suggested_step() -> None:
    llm = _StreamingStubLLM(
        [
            "Welcome back, Sam! ",
            "Your PR landed.",
            "\n<<<ACTION>>>\n",
            _ACTION,
        ]
    )

    events = list(stream_session(memory="old", recent=[], state="", llm=llm))

    assert _tokens(events) == "Welcome back, Sam! Your PR landed."
    done = _done(events)
    assert done["greeting"] == "Welcome back, Sam! Your PR landed."
    assert done["action"] == {
        "label": "What next?",
        "question": "What should I work on?",
    }


def test_the_open_does_not_rewrite_the_memory_note() -> None:
    """⚠️ The defect this closed: the note used to be written by *this* call.

    So a hire's durable memory was composed while the model was busy greeting them.
    Folding is `/onboarding/buddy/compact`, and the terminal event carries no note at
    all — a caller cannot persist what it is never handed.
    """
    llm = _StreamingStubLLM(["Hi!\n<<<ACTION>>>\n" + _ACTION])

    done = _done(list(stream_session(memory="the note", recent=[], state="", llm=llm)))

    assert "memory" not in done


def test_the_model_is_told_not_to_restate_the_note() -> None:
    """A model shown the memory will otherwise helpfully offer an updated one.

    Which would land in the greeting the hire reads.
    """
    llm = _StreamingStubLLM(["Hi!"])

    list(stream_session(memory="the note", recent=[], state="", llm=llm))

    assert llm.last_prompt is not None
    assert "Do not rewrite or restate your memory note" in llm.last_prompt[0]["content"]


def test_the_memory_still_reaches_the_model_as_context() -> None:
    """Read from, even though never written: a greeting is specific or it is generic."""
    llm = _StreamingStubLLM(["Hi!"])

    list(stream_session(memory="Sam is learning Kotlin.", recent=[], state="", llm=llm))

    assert llm.last_prompt is not None
    assert "Sam is learning Kotlin." in llm.last_prompt[-1]["content"]


def test_the_greeting_starts_arriving_before_the_marker_is_reached() -> None:
    """The reason the whole change exists: first token out before generation ends."""
    llm = _StreamingStubLLM(["Hi Sam!", " Nice to see you.", "\n<<<ACTION>>>\nnone"])

    events = list(stream_session(memory=None, recent=[], state="", llm=llm))

    assert events[0] == {"type": "token", "content": "Hi Sam!"}


def test_holds_back_only_a_possible_marker_prefix() -> None:
    """⚠️ "<<<ACT" is both a partial marker and legitimate prose.

    Only the next chunk decides which, so it must be a candidate, never an early return.
    """
    llm = _StreamingStubLLM(["Hello there<<<ACT", "ION>>>\n" + _ACTION])

    events = list(stream_session(memory=None, recent=[], state="", llm=llm))

    assert _tokens(events) == "Hello there"
    assert _done(events)["action"] is not None


def test_a_prefix_that_turns_out_to_be_prose_is_emitted_after_all() -> None:
    llm = _StreamingStubLLM(["Careful with <<<angle", " brackets>>> in code."])

    events = list(stream_session(memory="keep me", recent=[], state="", llm=llm))

    assert _tokens(events) == "Careful with <<<angle brackets>>> in code."
    assert _done(events)["action"] is None


def test_a_marker_split_across_chunks_is_still_found() -> None:
    llm = _StreamingStubLLM(["Hi!\n<<<", "ACT", "ION", ">>>\n" + _ACTION])

    events = list(stream_session(memory=None, recent=[], state="", llm=llm))

    assert _tokens(events).strip() == "Hi!"
    assert _done(events)["action"] is not None


def test_a_model_that_ignores_the_format_yields_all_of_it_as_the_greeting() -> None:
    """Harmless now: there is nothing behind the marker but a suggestion."""
    llm = _StreamingStubLLM(["Just some prose with no markers at all."])

    events = list(stream_session(memory="keep me", recent=[], state="", llm=llm))

    assert _tokens(events) == "Just some prose with no markers at all."
    assert _done(events)["action"] is None


def test_missing_action_section_is_no_action_rather_than_a_failure() -> None:
    llm = _StreamingStubLLM(["Hi!"])

    assert _done(list(stream_session(None, [], "", llm)))["action"] is None


def test_unparseable_action_json_is_dropped_without_losing_the_greeting() -> None:
    llm = _StreamingStubLLM(["Hi there!\n<<<ACTION>>>\nnone"])

    done = _done(list(stream_session(None, [], "", llm)))

    assert done["greeting"] == "Hi there!"
    assert done["action"] is None


def test_an_unavailable_model_still_yields_a_usable_opening() -> None:
    llm = _StreamingStubLLM(LLMUnavailableError("down"))

    done = _done(list(stream_session(memory="keep me", recent=[], state="", llm=llm)))

    assert done["greeting"]
    assert done["action"] is None


def test_a_blank_greeting_falls_back_rather_than_showing_nothing() -> None:
    llm = _StreamingStubLLM(["\n<<<ACTION>>>\n" + _ACTION])

    assert _done(list(stream_session(None, [], "", llm)))["greeting"]


def test_the_streamed_prompt_carries_the_recent_window_too() -> None:
    llm = _StreamingStubLLM(["hi"])
    recent = [Message(role="user", content="how do I run the tests?")]

    list(stream_session(memory=None, recent=recent, state="", llm=llm))

    assert llm.last_prompt is not None
    assert "how do I run the tests?" in llm.last_prompt[-1]["content"]


def test_the_done_greeting_is_exactly_what_was_streamed() -> None:
    """⚠️ The client renders the tokens; the caller persists ``done``.

    They must not differ.
    """
    llm = _StreamingStubLLM(["Hi Sam!", " Nice work.", "\n\n<<<ACTION>>>\n" + _ACTION])

    events = list(stream_session(None, [], "", llm))

    assert _done(events)["greeting"] == _tokens(events)
