from pathlib import Path

import pytest

from ingestion.models import ParsedChunk
from ingestion.parser import parse

FIXTURES_DIR = Path(__file__).parent / "fixtures"

CHUNK_SIZE: int = 512


@pytest.fixture
def markdown_small_file_content() -> bytes:
    return (FIXTURES_DIR / "markdown_small_sample.md").read_bytes()


@pytest.fixture
def text_small_file_content() -> bytes:
    return (FIXTURES_DIR / "text_small_sample.txt").read_bytes()


@pytest.fixture
def json_small_file_content() -> bytes:
    return (FIXTURES_DIR / "json_small_sample.json").read_bytes()


@pytest.fixture
def markdown_large_file_content() -> bytes:
    return (FIXTURES_DIR / "markdown_large_sample.md").read_bytes()


@pytest.fixture
def text_large_file_content() -> bytes:
    return (FIXTURES_DIR / "text_large_sample.txt").read_bytes()


@pytest.fixture
def json_large_file_content() -> bytes:
    return (FIXTURES_DIR / "json_large_sample.json").read_bytes()


def test_reject_incompatible_file_type():
    filename = "c++.cpp"
    result: list[ParsedChunk] = parse(filename, b"")
    assert len(result) == 0


def test_parse_small_markdown_as_single_chunk(markdown_small_file_content: bytes):
    filename = "markdown_small_sample.md"
    result: list[ParsedChunk] = parse(filename, markdown_small_file_content)

    assert len(result) == 1

    chunk = result[0]

    assert chunk.kind == "text"

    metadata = chunk.metadata
    assert metadata["filename"] == filename
    assert metadata["type"] == ".md"


def test_parse_small_txt_file_as_single_chunk(text_small_file_content: bytes):
    filename = "text_small_sample.txt"
    result: list[ParsedChunk] = parse(filename, text_small_file_content)

    assert len(result) == 1

    chunk = result[0]

    assert chunk.kind == "text"

    metadata = chunk.metadata
    assert metadata["filename"] == filename
    assert metadata["type"] == ".txt"


def test_parse_small_json_as_single_chunk(json_small_file_content: bytes):
    filename = "json_small_sample.json"
    result: list[ParsedChunk] = parse(filename, json_small_file_content)

    assert len(result) == 1

    chunk = result[0]

    assert chunk.kind == "text"

    metadata = chunk.metadata
    assert metadata["filename"] == filename
    assert metadata["type"] == ".json"


def test_parse_large_markdown_as_multiple_chunks(markdown_large_file_content: bytes):
    filename = "markdown_large_sample.md"
    result = parse(filename, markdown_large_file_content)

    assert len(result) > 1

    assert [chunk.metadata["chunk_index"] for chunk in result] == [
        str(i) for i in range(len(result))
    ]

    for chunk in result[:-1]:
        assert len(chunk.content) <= CHUNK_SIZE

    for chunk in result:
        assert chunk.kind == "text"
        assert chunk.metadata["filename"] == filename
        assert chunk.metadata["type"] == ".md"


def test_parse_large_json_as_multiple_chunks(json_large_file_content: bytes):
    filename = "json_large_sample.json"
    result = parse(filename, json_large_file_content)

    assert len(result) > 1

    assert [chunk.metadata["chunk_index"] for chunk in result] == [
        str(i) for i in range(len(result))
    ]

    for chunk in result[:-1]:
        assert len(chunk.content) <= CHUNK_SIZE

    for chunk in result:
        assert chunk.kind == "text"
        assert chunk.metadata["filename"] == filename
        assert chunk.metadata["type"] == ".json"


def test_parse_large_txt_as_multiple_chunks(text_large_file_content: bytes):
    filename = "text_large_sample.txt"
    result = parse(filename, text_large_file_content)

    assert len(result) > 1

    assert [chunk.metadata["chunk_index"] for chunk in result] == [
        str(i) for i in range(len(result))
    ]

    for chunk in result[:-1]:
        assert len(chunk.content) <= CHUNK_SIZE

    for chunk in result:
        assert chunk.kind == "text"
        assert chunk.metadata["filename"] == filename
        assert chunk.metadata["type"] == ".txt"


def test_parse_python_single_function():
    code = b"""
import os

def foo():
    return 1
"""

    result = parse("test.py", code)

    assert len(result) >= 1
    assert all(chunk.kind == "code" for chunk in result)


def test_parse_markdown_with_crlf_line_endings():
    content = b"# Header\r\n\r\nFirst paragraph.\r\n\r\nSecond paragraph."
    result = parse("test.md", content)

    assert len(result) >= 1
    assert result[0].kind == "text"
    assert "Header" in result[0].content
