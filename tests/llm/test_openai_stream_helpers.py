"""The pure helpers behind the streamed tool turn, one rule per test."""

from types import SimpleNamespace
from typing import cast

from openai.types.chat.chat_completion_chunk import ChoiceDelta

from llm.openai_stream import (
    loads_arguments,
    merge_reasoning_details,
    reasoning_detail_fragments,
    reasoning_text,
)


def _delta(**fields: object) -> ChoiceDelta:
    return cast("ChoiceDelta", SimpleNamespace(**fields))


class TestLoadsArguments:
    def test_reads_a_json_object(self) -> None:
        assert loads_arguments('{"query": "x", "n": 2}') == {"query": "x", "n": 2}

    def test_invalid_json_is_no_arguments(self) -> None:
        assert loads_arguments("{oops") == {}

    def test_json_that_is_not_an_object_is_no_arguments(self) -> None:
        assert loads_arguments("[1, 2]") == {}
        assert loads_arguments('"text"') == {}
        assert loads_arguments("3") == {}


class TestReasoningText:
    def test_the_plain_channel_is_returned_whole(self) -> None:
        assert reasoning_text(_delta(reasoning="  think ")) == ["  think "]

    def test_a_blank_plain_channel_falls_through_to_reasoning_content(self) -> None:
        delta = _delta(reasoning="  ", reasoning_content="legacy")
        assert reasoning_text(delta) == ["legacy"]

    def test_the_plain_channel_wins_over_structured_details(self) -> None:
        delta = _delta(reasoning="plain", reasoning_details=[{"text": "block"}])
        assert reasoning_text(delta) == ["plain"]

    def test_dict_details_give_text_or_summary(self) -> None:
        delta = _delta(
            reasoning_details=[
                {"text": "a"},
                {"summary": "b"},
                {"text": "c", "summary": "d"},
            ]
        )
        assert reasoning_text(delta) == ["a", "b", "c"]

    def test_object_details_give_text_or_summary(self) -> None:
        delta = _delta(
            reasoning_details=[
                SimpleNamespace(text="a"),
                SimpleNamespace(summary="b"),
                SimpleNamespace(text="c", summary="d"),
            ]
        )
        assert reasoning_text(delta) == ["a", "b", "c"]

    def test_blank_and_non_text_details_are_dropped(self) -> None:
        delta = _delta(
            reasoning_details=[
                {"text": "  "},
                {"text": 3},
                {"data": "opaque"},
                SimpleNamespace(text=None, summary=None),
                SimpleNamespace(),
                {"text": "kept"},
            ]
        )
        assert reasoning_text(delta) == ["kept"]

    def test_details_that_are_not_a_list_are_ignored(self) -> None:
        assert reasoning_text(_delta(reasoning_details={"text": "a"})) == []
        assert reasoning_text(_delta()) == []


class _Block:
    def __init__(self, **fields: object) -> None:
        self._fields = fields

    def model_dump(self) -> dict[str, object]:
        return dict(self._fields)


class TestReasoningDetailFragments:
    def test_dict_blocks_are_copied(self) -> None:
        block: dict[str, object] = {"type": "reasoning.text", "text": "a"}
        fragments = reasoning_detail_fragments(_delta(reasoning_details=[block]))

        assert fragments == [block]
        assert fragments[0] is not block

    def test_model_objects_are_dumped(self) -> None:
        delta = _delta(reasoning_details=[_Block(type="reasoning.text", text="a")])
        assert reasoning_detail_fragments(delta) == [
            {"type": "reasoning.text", "text": "a"}
        ]

    def test_an_object_that_cannot_dump_is_skipped_without_ending_the_scan(
        self,
    ) -> None:
        delta = _delta(reasoning_details=[object(), {"text": "kept"}, _Block(text="b")])
        assert reasoning_detail_fragments(delta) == [{"text": "kept"}, {"text": "b"}]

    def test_details_that_are_not_a_list_give_nothing(self) -> None:
        assert reasoning_detail_fragments(_delta(reasoning_details="x")) == []
        assert reasoning_detail_fragments(_delta()) == []


class TestMergeReasoningDetails:
    def test_payload_fields_are_concatenated(self) -> None:
        merged = merge_reasoning_details(
            [
                {"type": "reasoning.text", "text": "a", "index": 0},
                {"text": "b"},
                {"type": "reasoning.text", "text": "c", "index": 0},
            ]
        )
        assert merged == [{"type": "reasoning.text", "text": "abc", "index": 0}]

    def test_summary_and_data_are_payload_too(self) -> None:
        merged = merge_reasoning_details(
            [{"summary": "s1", "data": "d1"}, {"summary": "s2", "data": "d2"}]
        )
        assert merged == [{"summary": "s1s2", "data": "d1d2"}]

    def test_a_later_constant_overwrites_and_none_does_not(self) -> None:
        merged = merge_reasoning_details(
            [{"format": "v1", "signature": "x"}, {"format": "v2", "signature": None}]
        )
        assert merged == [{"format": "v2", "signature": "x"}]

    def test_a_payload_that_started_non_text_is_replaced_not_joined(self) -> None:
        merged = merge_reasoning_details([{"text": None}, {"text": "a"}])
        assert merged == [{"text": "a"}]

    def test_a_differing_type_opens_a_new_block(self) -> None:
        merged = merge_reasoning_details(
            [
                {"type": "reasoning.summary", "index": 0, "summary": "s"},
                {"type": "reasoning.encrypted", "index": 0, "data": "d"},
            ]
        )
        assert [b["type"] for b in merged] == [
            "reasoning.summary",
            "reasoning.encrypted",
        ]

    def test_a_differing_id_or_index_opens_a_new_block(self) -> None:
        assert len(merge_reasoning_details([{"id": "a"}, {"id": "b"}])) == 2
        assert len(merge_reasoning_details([{"index": 0}, {"index": 1}])) == 2

    def test_a_fragment_that_omits_identity_continues_the_open_block(self) -> None:
        merged = merge_reasoning_details(
            [{"type": "t", "id": "a", "index": 0, "text": "x"}, {"text": "y"}]
        )
        assert merged == [{"type": "t", "id": "a", "index": 0, "text": "xy"}]

    def test_the_block_that_continues_is_the_last_one_only(self) -> None:
        merged = merge_reasoning_details(
            [
                {"id": "a", "text": "1"},
                {"id": "b", "text": "2"},
                {"id": "a", "text": "3"},
            ]
        )
        assert [b["text"] for b in merged] == ["1", "2", "3"]

    def test_nothing_in_nothing_out(self) -> None:
        assert merge_reasoning_details([]) == []
