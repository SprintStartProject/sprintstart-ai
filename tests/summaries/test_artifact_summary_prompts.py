import pytest

from corpus.types import Chunk
from llm.base import Message
from summaries.artifact_summary import (
    _MAX_BATCH_CHARS,
    _MAX_SUMMARY_BATCHES,
    ArtifactSummaryAgent,
    SummaryCitation,
    _chunk_batches,
    _citations_for_chunks,
    _format_chunk,
)
from tests.stubs.llm import StubLLMClient
from tests.support.golden import assert_matches_golden


class _RecordingLLM(StubLLMClient):
    def __init__(self, reply: str = "note") -> None:
        super().__init__(generate_response=reply)
        self.calls: list[list[Message]] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.calls.append(messages)
        return super().generate(messages, temperature=temperature)


def _chunk(
    filename: str = "a.md",
    text: str = "body",
    position: int | None = None,
    source_url: str | None = None,
) -> Chunk:
    return Chunk(
        id=f"{filename}-{position}",
        artifact_id="art",
        filename=filename,
        text=text,
        embedding=[1.0, 0.0],
        position=position,
        source_url=source_url,
    )


def test_final_prompt_without_a_previous_version_matches_golden() -> None:
    agent = ArtifactSummaryAgent(StubLLMClient())

    messages = agent._final_summary_messages("art-1", "current notes", None)

    assert_matches_golden("summary_final_current_only", messages)


def test_final_prompt_with_a_previous_version_matches_golden() -> None:
    agent = ArtifactSummaryAgent(StubLLMClient())

    messages = agent._final_summary_messages("art-1", "current notes", "old notes")

    assert_matches_golden("summary_final_with_previous", messages)


def test_a_single_batch_is_used_as_the_notes_without_asking_the_model() -> None:
    llm = _RecordingLLM()

    notes = ArtifactSummaryAgent(llm)._notes_for_chunks(
        "current artifact", [_chunk(text="short")]
    )

    assert notes == "\n--- Source: a.md | chunk 0 ---\nshort\n"
    assert llm.calls == []


def test_each_batch_of_a_long_artifact_is_summarised_with_its_own_prompt() -> None:
    llm = _RecordingLLM("  a partial  ")
    chunks = [_chunk(f"f{i}.md", "x" * 6000, position=i) for i in range(3)]

    notes = ArtifactSummaryAgent(llm)._notes_for_chunks("current artifact", chunks)

    assert notes == "a partial\n\na partial\n\na partial"
    assert len(llm.calls) == 3
    assert "Summarize batch 2 of the current artifact." in llm.calls[1][1]["content"]
    assert_matches_golden("summary_batch_notes", [llm.calls[0][0], llm.calls[0][1]])


def test_batches_beyond_the_limit_are_dropped_with_a_notice() -> None:
    llm = _RecordingLLM("n")
    chunks = [
        _chunk(f"f{i}.md", "x" * 6000, position=i)
        for i in range(_MAX_SUMMARY_BATCHES + 2)
    ]

    notes = ArtifactSummaryAgent(llm)._notes_for_chunks("current artifact", chunks)

    assert len(llm.calls) == _MAX_SUMMARY_BATCHES
    assert notes.endswith(
        "\n\n[Note: Additional source batches were omitted because the "
        "artifact exceeds the summary batch limit.]"
    )


def test_summarising_nothing_is_an_error() -> None:
    with pytest.raises(ValueError, match="without chunks"):
        list(ArtifactSummaryAgent(StubLLMClient()).summarize_stream("a", []))


def test_the_summary_is_the_streamed_text_joined_and_trimmed() -> None:
    stream = ArtifactSummaryAgent(StubLLMClient("  the summary \n")).summarize_stream(
        "a", [_chunk()]
    )
    try:
        while True:
            next(stream)
    except StopIteration as stop:
        result = stop.value

    assert result.summary == "the summary"
    assert result.artifact_id == "a"


def test_chunks_are_batched_in_position_order() -> None:
    chunks = [_chunk("b.md", "second", position=2), _chunk("a.md", "first", position=1)]

    (batch,) = _chunk_batches(chunks, _MAX_BATCH_CHARS)

    assert batch.index("first") < batch.index("second")


def test_a_chunk_without_a_position_counts_as_position_zero() -> None:
    chunks = [_chunk("b.md", "later", position=1), _chunk("a.md", "unplaced")]

    (batch,) = _chunk_batches(chunks, _MAX_BATCH_CHARS)

    assert batch.index("unplaced") < batch.index("later")
    assert "| chunk 0 ---" in batch


def test_a_new_batch_starts_only_when_the_next_block_would_overflow() -> None:
    block = len(_format_chunk(_chunk(text="x" * 10)))

    fits = _chunk_batches([_chunk(text="x" * 10), _chunk(text="x" * 10)], block * 2)
    overflows = _chunk_batches(
        [_chunk(text="x" * 10), _chunk(text="x" * 10)], block * 2 - 1
    )

    assert len(fits) == 1
    assert len(overflows) == 2


def test_a_block_longer_than_a_batch_is_cut_and_marked() -> None:
    (batch,) = _chunk_batches([_chunk(text="x" * 50)], 20)

    assert batch.endswith("\n[truncated]\n")
    assert len(batch) == 20 + len("\n[truncated]\n")


def test_a_chunk_is_formatted_with_its_file_and_position() -> None:
    assert _format_chunk(_chunk("a.md", "  text  ", position=3)) == (
        "\n--- Source: a.md | chunk 3 ---\ntext\n"
    )


def test_a_file_cited_twice_keeps_its_real_url_over_the_fallback() -> None:
    chunks = [_chunk("a.md"), _chunk("a.md", source_url="https://x/a.md")]

    assert _citations_for_chunks("art", chunks) == [
        SummaryCitation("art", "a.md", "https://x/a.md")
    ]


def test_a_real_url_is_not_replaced_by_a_later_one() -> None:
    chunks = [
        _chunk("a.md", source_url="https://x/first"),
        _chunk("a.md", source_url="https://x/second"),
    ]

    assert _citations_for_chunks("art", chunks)[0].source_url == "https://x/first"
