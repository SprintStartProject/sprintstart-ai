import logging

import pytest

from assembly.grading import (
    AnswerGrade,
    AnswerToGrade,
    _build_prompt,
    _correction,
    grade_answers,
)
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient
from tests.support.golden import assert_matches_golden


class _ScriptedLLM(StubLLMClient):
    def __init__(self, *replies: str) -> None:
        super().__init__()
        self._replies = list(replies)
        self.calls: list[list[Message]] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.calls.append(messages)
        return self._replies.pop(0)


def _answer(answer_id: str, user_answer: str = "an answer") -> AnswerToGrade:
    return AnswerToGrade(
        id=answer_id,
        question=f"Question {answer_id}?",
        reference_answer=f"Reference {answer_id}",
        user_answer=user_answer,
    )


def test_grading_prompt_matches_golden() -> None:
    assert_matches_golden(
        "grading_two_answers", _build_prompt([_answer("q1"), _answer("q2", "other")])
    )


def test_correction_round_matches_golden() -> None:
    assert_matches_golden(
        "grading_correction", _correction("{broken", ValueError("boom"))
    )


def test_every_answer_is_graded_in_one_call_and_in_request_order() -> None:
    llm = _ScriptedLLM(
        '{"results": [{"id": "q2", "correct": false, "confidence": 0.25,'
        ' "feedback": "No."}, {"id": "q1", "correct": true, "confidence": 0.5,'
        ' "feedback": "Yes."}]}'
    )

    grades = grade_answers(llm, [_answer("q1"), _answer("q2")])

    assert grades == [
        AnswerGrade(id="q1", correct=True, confidence=0.5, feedback="Yes."),
        AnswerGrade(id="q2", correct=False, confidence=0.25, feedback="No."),
    ]
    assert len(llm.calls) == 1


def test_a_blank_answer_is_incorrect_without_asking_the_model() -> None:
    llm = _ScriptedLLM()

    grades = grade_answers(llm, [_answer("q1", "   ")])

    assert grades == [
        AnswerGrade(id="q1", correct=False, feedback="No answer submitted.")
    ]
    assert llm.calls == []


def test_only_the_non_blank_answers_are_sent_to_the_model() -> None:
    llm = _ScriptedLLM('{"results": [{"id": "q2", "correct": true}]}')

    grades = grade_answers(llm, [_answer("q1", ""), _answer("q2")])

    assert [g.id for g in grades] == ["q1", "q2"]
    assert grades[1].correct is True
    assert "id: q1" not in llm.calls[0][1]["content"]
    assert "id: q2" in llm.calls[0][1]["content"]


def test_an_unusable_reply_gets_one_correction_round_with_the_reply_and_the_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _ScriptedLLM("{}", '{"results": [{"id": "q1", "correct": true}]}')

    with caplog.at_level(logging.WARNING):
        grades = grade_answers(llm, [_answer("q1")])

    assert grades[0].correct is True
    first, second = llm.calls
    assert second[: len(first)] == first
    assert second[len(first)] == {"role": "assistant", "content": "{}"}
    assert second[-1]["role"] == "user"
    assert second[-1]["content"].startswith("That response could not be validated: ")
    assert (
        caplog.records[0]
        .getMessage()
        .startswith("Could not parse grade-answers output (attempt 1): ")
    )


def test_after_two_unusable_replies_the_answers_count_as_not_gradable(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _ScriptedLLM("nope", "still nope")

    with caplog.at_level(logging.WARNING):
        grades = grade_answers(llm, [_answer("q1")])

    assert grades == [
        AnswerGrade(
            id="q1", correct=False, feedback="Could not be graded automatically."
        )
    ]
    assert len(llm.calls) == 2
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 2
    assert messages[0].startswith("Could not parse grade-answers output (attempt 1): ")
    assert messages[1].startswith("Could not parse grade-answers output (attempt 2): ")


def test_an_unavailable_model_propagates() -> None:
    class Down(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            raise LLMUnavailableError("offline")

    with pytest.raises(LLMUnavailableError):
        grade_answers(Down(), [_answer("q1")])
