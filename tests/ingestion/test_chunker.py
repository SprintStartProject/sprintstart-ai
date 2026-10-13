import importlib
import sys

import pytest
from pytest import MonkeyPatch

from ingestion.chunker import _paragraphs_with_start_lines, chunk_text  # type: ignore
from ingestion.code_parser import chunk_code
from ingestion.parser import parse


def test_chunk_overlap_must_be_smaller_than_chunk_size(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")

    sys.modules.pop("ingestion.chunker", None)

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        importlib.import_module("ingestion.chunker")


def test_chunk_overlap_larger_than_chunk_size_raises(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "200")

    sys.modules.pop("ingestion.chunker", None)

    with pytest.raises(
        ValueError,
        match="chunk_overlap must be smaller than chunk_size",
    ):
        importlib.import_module("ingestion.chunker")


def test_valid_chunk_configuration(monkeypatch: MonkeyPatch):
    monkeypatch.setenv("CHUNK_SIZE", "512")
    monkeypatch.setenv("CHUNK_OVERLAP", "64")

    sys.modules.pop("ingestion.chunker", None)

    module = importlib.import_module("ingestion.chunker")

    assert module.chunk_size == 512
    assert module.chunk_overlap == 64


def test_chunks_have_correct_order():
    filename = "big.txt"
    content = b"A" * 1500

    result = parse(filename, content)

    for i, chunk in enumerate(result):
        assert chunk.metadata["chunk_index"] == str(i)


def test_large_paragraph_is_hard_split():
    text = "A" * 1200

    chunks = chunk_text(
        "file.txt",
        text,
        chunk_size=512,
        chunk_overlap=64,
    )

    assert len(chunks) >= 3
    for chunk in chunks:
        assert len(chunk.content) <= 512


def test_chunk_code_respects_size_directly():
    code = "\n".join(
        [
            "def foo():",
            "    pass",
        ]
        * 200
    )

    chunks = chunk_code("test.py", code, chunk_size=50)

    for chunk in chunks[:-1]:
        assert len(chunk.content) <= 50


def test_text_chunks_split_on_paragraph_boundaries():
    text = "Paragraph A\n\nParagraph B\n\nParagraph C"

    chunks = chunk_text("file.txt", text, chunk_size=25)

    assert len(chunks) == 2

    assert "Paragraph A" in chunks[0].content
    assert "Paragraph B" in chunks[0].content

    assert "Paragraph C" in chunks[1].content
    assert "Paragraph B" not in chunks[1].content


def test_chunking_does_not_cut_typical_prose_mid_sentence():
    text = (
        "This is the first paragraph. "
        "It contains several complete sentences.\n\n"
        "This is the second paragraph. "
        "It also contains complete sentences.\n\n"
        "This is the third paragraph."
    )

    chunks = chunk_text(
        "file.txt",
        text,
        chunk_size=120,
    )

    for chunk in chunks:
        assert not chunk.content.startswith("t contains")
        assert not chunk.content.startswith("plete sentences")


def test_paragraphs_not_duplicated_across_chunks():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_text(
        "file.txt",
        text,
        chunk_size=35,
    )

    assert len(chunks) == 2

    assert "First paragraph." in chunks[0].content
    assert "Second paragraph." in chunks[0].content
    assert "Third paragraph." in chunks[1].content
    assert "Second paragraph." not in chunks[1].content


def test_no_chunk_exceeds_chunk_size_with_mixed_paragraphs():
    text = (
        "Short intro.\n\n"
        "A somewhat longer paragraph that holds substantial text.\n\n"
        "Another paragraph that is medium length.\n\n"
        "Final paragraph to wrap up."
    )

    chunks = chunk_text("file.txt", text, chunk_size=60)

    assert len(chunks) >= 2
    for chunk in chunks:
        assert len(chunk.content) <= 60


def test_chunk_text_tracks_start_line_per_paragraph():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_text("file.txt", text, chunk_size=1000)

    assert len(chunks) == 1
    assert chunks[0].metadata["start_line"] == "1"


def test_chunk_text_start_line_accounts_for_blank_lines_between_chunks():
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."

    chunks = chunk_text("file.txt", text, chunk_size=20)

    assert len(chunks) >= 2
    # "First paragraph." is on line 1.
    assert chunks[0].metadata["start_line"] == "1"
    # "Second paragraph." starts after one blank line, i.e. on line 3.
    assert any(c.metadata["start_line"] == "3" for c in chunks)


def test_chunk_text_start_line_skips_extra_blank_lines():
    # Three blank lines (not just one) between paragraphs.
    text = "First paragraph.\n\n\n\nSecond paragraph."

    paragraphs = _paragraphs_with_start_lines(text)

    assert paragraphs == [
        ("First paragraph.", 1),
        ("Second paragraph.", 5),
    ]


def test_chunk_code_tags_every_chunk_with_given_start_line():
    code = "\n".join(
        [
            "def foo():",
            "    pass",
        ]
        * 200
    )

    chunks = chunk_code("test.py", code, chunk_size=50, start_line=7)

    assert len(chunks) > 1
    assert all(chunk.metadata["start_line"] == "7" for chunk in chunks)


def test_chunk_code_without_start_line_omits_metadata_key():
    chunks = chunk_code("test.py", "def foo():\n    pass\n")

    assert "start_line" not in chunks[0].metadata
