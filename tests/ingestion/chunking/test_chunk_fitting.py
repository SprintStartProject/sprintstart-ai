from ingestion.chunking.chunk_fitting import (
    fit_chunks,
    hard_split_with_context,
    sanitize_context_block,
    to_parsed_chunks_from_pieces,
)


def test_hard_split_repeats_the_context_block_and_overlaps_bodies() -> None:
    pieces = hard_split_with_context("CTX\n\n", "abcdefghij", 9, 2)

    assert pieces == [
        ("CTX\n\nabcd", (0, 5), False, None),
        ("CTX\n\ncdef", (0, 5), True, (5, 7)),
        ("CTX\n\nefgh", (0, 5), True, (5, 7)),
        ("CTX\n\nghij", (0, 5), True, (5, 7)),
    ]


def test_hard_split_without_context_has_no_context_range() -> None:
    pieces = hard_split_with_context("", "abcdefg", 4, 1)

    assert pieces == [
        ("abcd", None, False, None),
        ("defg", None, True, (0, 1)),
    ]


def test_hard_split_of_an_empty_body_keeps_the_context_block() -> None:
    assert hard_split_with_context("C\n\n", "", 10, 2) == [
        ("C\n\n", (0, 3), False, None)
    ]


def test_hard_split_keeps_one_body_character_when_the_context_fills_the_budget() -> (
    None
):
    pieces = hard_split_with_context("CTX\n\n", "ab", 3, 2)

    assert pieces == [
        ("CTX\n\na", (0, 5), False, None),
        ("CTX\n\nb", (0, 5), False, None),
    ]


def test_hard_split_caps_the_overlap_below_the_body_budget() -> None:
    pieces = hard_split_with_context("", "abc", 2, 5)

    assert pieces == [("ab", None, False, None), ("bc", None, True, (0, 1))]


def test_fit_chunks_prepends_context_and_splits_only_oversized_chunks() -> None:
    pieces = fit_chunks(["aa", "bbbbbbbb"], {"0": "X"}, 6, 1)

    assert pieces == [
        ("X\n\naa", (0, 3), False, None),
        ("bbbbbb", None, False, None),
        ("bbb", None, True, (0, 1)),
    ]


def test_fit_chunks_keeps_a_chunk_that_exactly_fills_the_budget() -> None:
    assert fit_chunks(["abcdef"], {}, 6, 1) == [("abcdef", None, False, None)]


def test_fit_chunks_looks_up_the_context_block_by_chunk_order() -> None:
    pieces = fit_chunks(["aa", "bb"], {"1": "Intro to the topic"}, 100, 1)

    assert pieces == [
        ("aa", None, False, None),
        ("Intro to the topic\n\nbb", (0, 20), False, None),
    ]


def test_sanitize_context_block_drops_a_missing_blank_or_too_short_block() -> None:
    assert sanitize_context_block(None, 100) is None
    assert sanitize_context_block("   ", 100) is None
    assert sanitize_context_block("too short", 100) is None


def test_sanitize_context_block_trims_and_keeps_a_block_that_fits() -> None:
    assert sanitize_context_block("  Intro to the topic  ", 100) == "Intro to the topic"


def test_sanitize_context_block_truncates_to_a_quarter_of_the_chunk_size() -> None:
    assert sanitize_context_block("x" * 80, 100) == "x" * 25


def test_sanitize_context_block_scales_its_minimum_down_for_tiny_chunks() -> None:
    assert sanitize_context_block("abcde", 20) == "abcde"
    assert sanitize_context_block("abcd", 20) is None


def test_parsed_chunks_carry_position_and_range_metadata() -> None:
    chunks = to_parsed_chunks_from_pieces(
        "notes.md",
        [("X\n\naa", (0, 3), False, None), ("aa", None, True, (0, 1))],
    )

    assert [c.content for c in chunks] == ["X\n\naa", "aa"]
    assert {c.kind for c in chunks} == {"text"}
    first, second = (c.metadata for c in chunks)
    assert (first["chunk_index"], first["total_chunks"]) == ("0", "2")
    assert (second["chunk_index"], second["total_chunks"]) == ("1", "2")
    assert (first["has_context_block"], first["context_block_range"]) == ("true", "0:3")
    assert (second["has_context_block"], second["context_block_range"]) == ("false", "")
    assert (first["has_overlap"], first["overlap_range"]) == ("false", "")
    assert (second["has_overlap"], second["overlap_range"]) == ("true", "0:1")
    assert first["filename"] == "notes.md"
