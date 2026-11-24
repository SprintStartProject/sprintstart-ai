import json

import pytest
from pytest import MonkeyPatch

from ingestion.chunking.chunker import (
    chunk_text,
    group_paragraphs_into_chunks,
    split_into_paragraphs,
)
from ingestion.chunking.context_aware_chunker import (
    chunk_text_context_aware,
    exceeds_llm_limit,
)
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient


class _RaisingLLMClient(StubLLMClient):
    """Raises LLMUnavailableError instead of returning a response."""

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        raise LLMUnavailableError(host="http://stub")


def test_both_flags_disabled_skips_llm_and_matches_chunk_text():
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"
    llm = StubLLMClient(generate_response="should never be used")

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=False,
        contextualize=False,
        chunk_size=25,
    )

    assert len(chunks) == 2
    assert all(chunk.metadata["has_context_block"] == "false" for chunk in chunks)
    assert all(chunk.metadata["has_overlap"] == "false" for chunk in chunks)


def test_empty_text_falls_back_to_chunk_text():
    llm = StubLLMClient(generate_response="should never be used")

    chunks = chunk_text_context_aware("file.txt", "", llm)

    assert chunks == []


def test_text_exceeding_llm_limit_falls_back(monkeypatch: MonkeyPatch):
    monkeypatch.setattr(
        "ingestion.chunking.context_aware_chunker.max_chars", 10, raising=False
    )
    llm = StubLLMClient(generate_response="should never be used")
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"

    chunks = chunk_text_context_aware("file.txt", text, llm, chunk_size=25)

    # Falls back to plain chunk_text grouping (paragraph accumulation).
    assert len(chunks) == 2
    assert "Paragraph A" in chunks[0].content


def test_exceeds_llm_limit_helper():
    assert exceeds_llm_limit("a" * 100, max_chars=50) is True
    assert exceeds_llm_limit("a" * 10, max_chars=50) is False


def test_semantic_boundaries_group_paragraphs_per_llm_plan():
    text = "Intro paragraph.\n\nDetail paragraph.\n\nUnrelated paragraph."
    # boundaries=[0, 2] -> chunk 0 = paragraphs[0:2], chunk 1 = paragraphs[2:3]
    plan = json.dumps({"boundaries": [0, 2], "context_blocks": {}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=True,
        contextualize=False,
        chunk_size=1000,
    )

    assert len(chunks) == 2
    assert "Intro paragraph." in chunks[0].content
    assert "Detail paragraph." in chunks[0].content
    assert "Unrelated paragraph." in chunks[1].content
    assert chunks[0].metadata["has_context_block"] == "false"


def test_contextualize_only_prepends_context_block_to_fixed_chunks():
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"
    plan = json.dumps(
        {"boundaries": [], "context_blocks": {"0": "This chunk is about A and B."}}
    )
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=False,
        contextualize=True,
        chunk_size=1000,
    )

    assert len(chunks) == 1
    assert chunks[0].content.startswith("This chunk is about A and B.")
    assert chunks[0].metadata["has_context_block"] == "true"
    assert chunks[0].metadata["context_block_range"] != ""


def test_invalid_llm_output_falls_back_to_chunk_text():
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"
    llm = StubLLMClient(generate_response="not valid json at all")

    chunks = chunk_text_context_aware(
        "file.txt", text, llm, semantic_boundaries=True, chunk_size=25
    )

    assert len(chunks) == 2
    assert all(chunk.metadata["has_context_block"] == "false" for chunk in chunks)


def test_out_of_range_boundaries_falls_back_to_chunk_text():
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"
    plan = json.dumps({"boundaries": [0, 99], "context_blocks": {}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt", text, llm, semantic_boundaries=True, chunk_size=25
    )

    assert len(chunks) == 2  # chunk_text fallback result


def test_llm_unavailable_falls_back_to_chunk_text():
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"
    llm = _RaisingLLMClient()

    chunks = chunk_text_context_aware(
        "file.txt", text, llm, semantic_boundaries=True, chunk_size=25
    )

    assert len(chunks) == 2


def test_context_block_causing_overflow_is_hard_split_and_duplicated():
    # A single paragraph forming one chunk; the context block pushes the
    # combined chunk over chunk_size, triggering the hard-split path.
    text = "B" * 90
    context = "Context sentence."
    plan = json.dumps({"boundaries": [0], "context_blocks": {"0": context}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=True,
        contextualize=True,
        chunk_size=100,
        chunk_overlap=10,
    )

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 100
        assert chunk.content.startswith(context)
        assert chunk.metadata["has_context_block"] == "true"
    # sub-chunks after the first carry overlap of the body text
    assert chunks[1].metadata["has_overlap"] == "true"


def test_context_block_shorter_than_min_chars_is_dropped():
    text = "Paragraph A\n\nParagraph B"
    # "Intro" is only 5 chars < _CONTEXT_MIN_CHARS (15)
    plan = json.dumps({"boundaries": [0, 1], "context_blocks": {"0": "Intro"}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=True,
        contextualize=True,
        chunk_size=512,
    )

    assert chunks[0].metadata["has_context_block"] == "false"
    assert "Intro\n\n" not in chunks[0].content


def test_context_block_exceeding_max_fraction_is_trimmed():
    text = "Short paragraph."
    # 200 chars context block, chunk_size=200 -> max_len = 50 (25% of 200)
    long_context = (
        "This is a very long situating context sentence that keeps going and "
        "going and exceeds the limit. " * 3
    )
    plan = json.dumps({"boundaries": [0], "context_blocks": {"0": long_context}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=True,
        contextualize=True,
        chunk_size=200,
    )

    assert chunks[0].metadata["has_context_block"] == "true"
    # The context block was trimmed to 25% of chunk_size (50 chars)
    expected_trimmed = long_context[:50]
    assert chunks[0].content.startswith(expected_trimmed)


def test_hard_split_overlap_range_does_not_exceed_content_length():
    # Long body with context that causes multiple sub-chunks, testing tail
    # overlap range clamping
    text = "C" * 150
    context = "Situating context block."
    plan = json.dumps({"boundaries": [0], "context_blocks": {"0": context}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=True,
        contextualize=True,
        chunk_size=100,
        chunk_overlap=20,
    )

    for chunk in chunks:
        if chunk.metadata["has_overlap"] == "true":
            start_str, end_str = chunk.metadata["overlap_range"].split(":")
            start, end = int(start_str), int(end_str)
            assert end <= len(chunk.content)
            assert start < end


@pytest.mark.parametrize("semantic_boundaries", [True, False])
def test_chunks_without_needs_context_are_left_untouched(
    semantic_boundaries: bool,
):
    text = "Paragraph A\n\nParagraph B"
    plan = json.dumps({"boundaries": [0, 1], "context_blocks": {}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        text,
        llm,
        semantic_boundaries=semantic_boundaries,
        contextualize=True,
        chunk_size=1000,
    )

    assert all(chunk.metadata["has_context_block"] == "false" for chunk in chunks)


def test_build_prompt_includes_segment_count_and_range():
    from ingestion.chunking.context_aware_chunker import (
        _build_prompt,  # pyright: ignore[reportPrivateUsage]
    )

    segments = ["Para 0", "Para 1", "Para 2"]
    messages = _build_prompt(segments, semantic_boundaries=True, contextualize=True)

    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert "There are exactly 3 numbered paragraphs" in system_msg
    assert "between 0 and 2 (inclusive)" in system_msg
    assert "Do not invent indices above this range." in system_msg


class _RecordingLLMClient(StubLLMClient):
    def __init__(self, generate_response: str) -> None:
        super().__init__(generate_response=generate_response)
        self.prompts: list[str] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.prompts.append("\n".join(message["content"] for message in messages))
        return self.generate_response


_LONG_TEXT = "\n\n".join(f"Paragraph number {n} with some words." for n in range(8))


def test_fallback_forwards_chunk_size_and_overlap_to_chunk_text():
    expected = chunk_text("file.txt", _LONG_TEXT, 90, 25)
    llm = StubLLMClient(generate_response="not valid json at all")

    chunks = chunk_text_context_aware(
        "file.txt", _LONG_TEXT, llm, chunk_size=90, chunk_overlap=25
    )

    assert [c.content for c in chunks] == [c.content for c in expected]
    assert [c.metadata for c in chunks] == [c.metadata for c in expected]


def test_oversized_text_fallback_forwards_chunk_size_and_overlap(
    monkeypatch: MonkeyPatch,
):
    monkeypatch.setattr(
        "ingestion.chunking.context_aware_chunker.max_chars", 10, raising=False
    )
    expected = chunk_text("file.txt", _LONG_TEXT, 90, 25)

    chunks = chunk_text_context_aware(
        "file.txt", _LONG_TEXT, StubLLMClient(), chunk_size=90, chunk_overlap=25
    )

    assert [c.content for c in chunks] == [c.content for c in expected]


def test_disabled_flags_forward_chunk_size_and_overlap_to_chunk_text():
    expected = chunk_text("file.txt", _LONG_TEXT, 90, 25)

    chunks = chunk_text_context_aware(
        "file.txt",
        _LONG_TEXT,
        StubLLMClient(),
        semantic_boundaries=False,
        contextualize=False,
        chunk_size=90,
        chunk_overlap=25,
    )

    assert [c.content for c in chunks] == [c.content for c in expected]


def test_whitespace_only_text_falls_back_without_calling_the_llm():
    llm = _RecordingLLMClient("{}")

    blank = " \n\n \n\n "

    chunks = chunk_text_context_aware(
        "file.txt", blank, llm, chunk_size=90, chunk_overlap=10
    )

    assert chunks == chunk_text("file.txt", blank, 90, 10)
    assert llm.prompts == []


def test_fixed_segments_are_the_pre_grouped_chunks_for_the_given_size_and_overlap():
    expected = group_paragraphs_into_chunks(split_into_paragraphs(_LONG_TEXT), 90, 25)
    llm = _RecordingLLMClient(json.dumps({"boundaries": [], "context_blocks": {}}))

    chunks = chunk_text_context_aware(
        "file.txt",
        _LONG_TEXT,
        llm,
        semantic_boundaries=False,
        contextualize=True,
        chunk_size=90,
        chunk_overlap=25,
    )

    assert [c.content for c in chunks] == expected
    assert f"[{len(expected) - 1}]" in llm.prompts[0]


def test_prompt_depends_on_the_semantic_boundaries_and_contextualize_flags():
    plan = json.dumps({"boundaries": [0], "context_blocks": {}})
    semantic = _RecordingLLMClient(plan)
    plain = _RecordingLLMClient(plan)

    chunk_text_context_aware(
        "f.txt", _LONG_TEXT, semantic, semantic_boundaries=True, contextualize=False
    )
    chunk_text_context_aware(
        "f.txt", _LONG_TEXT, plain, semantic_boundaries=False, contextualize=True
    )

    assert semantic.prompts != plain.prompts
    assert "paragraph" in semantic.prompts[0]


def test_context_blocks_are_ignored_when_contextualize_is_off():
    plan = json.dumps({"boundaries": [0], "context_blocks": {"0": "SHOULD NOT APPEAR"}})
    llm = StubLLMClient(generate_response=plan)

    chunks = chunk_text_context_aware(
        "file.txt",
        "Paragraph A\n\nParagraph B",
        llm,
        semantic_boundaries=True,
        contextualize=False,
        chunk_size=1000,
    )

    assert [c.content for c in chunks] == ["Paragraph A\n\nParagraph B"]
    assert chunks[0].metadata["has_context_block"] == "false"


def test_a_failed_plan_is_logged_with_the_file_and_the_reason(
    caplog: pytest.LogCaptureFixture,
):
    with caplog.at_level("WARNING", logger="ingestion.chunking.context_aware_chunker"):
        chunk_text_context_aware(
            "report.txt", _LONG_TEXT, StubLLMClient(generate_response="nope")
        )

    message = caplog.records[-1].getMessage()
    assert message.startswith("Context-aware chunking failed for report.txt (")
    assert message.endswith("— falling back to chunk_text")
