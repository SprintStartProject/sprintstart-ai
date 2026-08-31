from llm.base import Message
from llm.errors import LLMUnavailableError
from onboarding.buddy_compact import compact_memory
from tests.stubs.llm import ScriptedLLMClient


class _RecordingLLM(ScriptedLLMClient):
    """Records what `generate` was asked, so the fold prompt can be asserted."""

    def __init__(self, answer: str = "The hire is set up and awaiting review.") -> None:
        super().__init__(turns=[], answer=answer)
        self.generate_calls: list[list[Message]] = []
        self.temperatures: list[float | None] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.generate_calls.append(messages)
        self.temperatures.append(temperature)
        return self.answer


class _UnavailableLLM(ScriptedLLMClient):
    def __init__(self) -> None:
        super().__init__(turns=[])

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        raise LLMUnavailableError("model down")


def _msg(role: str, text: str) -> Message:
    return Message(role=role, content=text)


def test_folds_the_transcript_into_a_rewritten_note() -> None:
    llm = _RecordingLLM()

    memory = compact_memory(
        "The hire joined on Monday.",
        [_msg("user", "how do I run the tests?"), _msg("assistant", "uv run pytest")],
        llm,
    )

    assert memory == "The hire is set up and awaiting review."
    prompt = llm.generate_calls[0][1].get("content") or ""
    # Both halves reach the model: what it already remembers, and what is sliding out.
    assert "The hire joined on Monday." in prompt
    assert "how do I run the tests?" in prompt
    assert "uv run pytest" in prompt


def test_compacts_deterministically() -> None:
    # The same conversation must fold the same way; a note that drifts between runs
    # makes a hire's durable memory depend on nothing they did.
    llm = _RecordingLLM()

    compact_memory(None, [_msg("user", "hello")], llm)

    assert llm.temperatures == [0]


def test_first_fold_says_there_is_nothing_yet_rather_than_sending_none() -> None:
    llm = _RecordingLLM()

    compact_memory(None, [_msg("user", "hello")], llm)

    assert "(nothing yet)" in (llm.generate_calls[0][1].get("content") or "")


def test_an_unavailable_model_folds_nothing_rather_than_blanking_the_note() -> None:
    # None is the caller's signal to leave its cursor alone and try again later. The
    # note is a prompt-shaping device, never the record -- the transcript it
    # compresses stays durable on the backend, so a fold that never happens costs a
    # longer prompt and nothing else.
    llm = _UnavailableLLM()

    folded = compact_memory("Earlier notes.", [_msg("user", "hello")], llm)

    assert folded is None


def test_a_slice_with_no_words_in_it_keeps_the_note_and_calls_no_model() -> None:
    # Not a failure: an empty slice folds successfully to the note as it stands, so
    # the caller may still advance its cursor past messages that carried no content.
    llm = _RecordingLLM()

    memory = compact_memory("Earlier notes.", [_msg("assistant", "   ")], llm)

    assert memory == "Earlier notes."
    assert llm.generate_calls == []


def test_an_empty_slice_before_the_first_fold_is_an_empty_note_never_none() -> None:
    # None means "could not fold" to the caller, so the no-op path must not return it.
    assert compact_memory(None, [], _RecordingLLM()) == ""
