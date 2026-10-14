import json

import pytest
from pytest import MonkeyPatch

from ingestion.context_aware_chunker import (
    chunk_text_context_aware,
    exceeds_llm_limit,
)
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient


class _RaisingLLMClient(StubLLMClient):
    """Raises LLMUnavailableError instead of returning a response."""

    def generate(self, messages: list[Message]) -> str:
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
    monkeypatch.setattr("ingestion.context_aware_chunker.max_chars", 10, raising=False)
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
