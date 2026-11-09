"""Semantic grading of short-text answers to onboarding knowledge-check questions."""

import json
import logging
from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from llm.base import LLMClient, Message
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

_MAX_GRADING_ATTEMPTS = 2

SYSTEM_PROMPT = """
You grade short-text answers to onboarding knowledge-check questions.

For each answer, decide whether it is semantically correct against the
reference answer:

- Judge the core meaning, not exact wording. Paraphrases, different casing,
  and typos are fine as long as the key fact/command is right.
- No extra facts beyond the reference answer are required.
- If the core claim is wrong, missing, or contradicts the reference, mark it
  incorrect.
- 'feedback' is one short sentence shown to the user explaining the verdict.

Return STRICT JSON only (no prose, no markdown fences), correlating results
by 'id', in the same order as the input:
{"results": [{"id": str, "correct": bool, "confidence": number 0..1, "feedback": str}]}
"""


@dataclass(frozen=True)
class AnswerToGrade:
    id: str
    question: str
    reference_answer: str
    user_answer: str


@dataclass(frozen=True)
class AnswerGrade:
    id: str
    correct: bool
    confidence: float | None = None
    feedback: str = ""


class _GradedItem(BaseModel):
    id: str
    correct: bool
    confidence: float | None = None
    feedback: str = ""


class _Payload(BaseModel):
    results: list[_GradedItem]


def grade_answers(llm: LLMClient, answers: list[AnswerToGrade]) -> list[AnswerGrade]:
    """Grade the answers, returning one grade per answer in the order given.

    A blank answer is incorrect without asking the model; the others are graded in
    one call. An answer the model did not grade is incorrect with generic feedback
    rather than failing the request. ``LLMUnavailableError`` propagates.
    """
    to_grade = [answer for answer in answers if answer.user_answer.strip()]
    graded_by_id = _grade_batch(llm, to_grade) if to_grade else {}
    return [_grade_of(answer, graded_by_id) for answer in answers]


def _grade_of(
    answer: AnswerToGrade, graded_by_id: dict[str, _GradedItem]
) -> AnswerGrade:
    if not answer.user_answer.strip():
        return AnswerGrade(id=answer.id, correct=False, feedback="No answer submitted.")
    graded = graded_by_id.get(answer.id)
    if graded is None:
        return AnswerGrade(
            id=answer.id,
            correct=False,
            feedback="Could not be graded automatically.",
        )
    return AnswerGrade(
        id=answer.id,
        correct=graded.correct,
        confidence=graded.confidence,
        feedback=graded.feedback,
    )


def _build_prompt(answers: list[AnswerToGrade]) -> list[Message]:
    blocks = [
        f"id: {item.id}\n"
        f"question: {item.question}\n"
        f"reference_answer: {item.reference_answer}\n"
        f"user_answer: {item.user_answer}"
        for item in answers
    ]
    return [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content="\n\n".join(blocks)),
    ]


def _parse_grades(raw: str, to_grade: list[AnswerToGrade]) -> dict[str, _GradedItem]:
    """The grades in ``raw`` by id.

    Raises ``ValueError`` unless there is exactly one grade per answer.
    """
    payload = _Payload.model_validate_json(extract_json_object(raw))
    graded_by_id = {item.id: item for item in payload.results}
    expected_ids = {item.id for item in to_grade}
    if len(payload.results) != len(to_grade) or set(graded_by_id) != expected_ids:
        raise ValueError(
            "grading response does not match the requested answer ids: "
            f"expected {sorted(expected_ids)}, got "
            f"{[item.id for item in payload.results]}"
        )
    return graded_by_id


def _correction(raw: str, error: Exception) -> list[Message]:
    return [
        Message(role="assistant", content=raw),
        Message(
            role="user",
            content=(
                f"That response could not be validated: {error}. Return the same "
                "grades again as one valid JSON object matching the schema "
                "exactly. Return JSON only."
            ),
        ),
    ]


def _grade_batch(
    llm: LLMClient, to_grade: list[AnswerToGrade]
) -> dict[str, _GradedItem]:
    """Grade in one call, asking once more when the reply is unusable.

    An unreadable reply marks every answer in it "could not be graded" -- which the
    caller records as wrong. A small local model breaks its JSON now and then, and
    a hire whose right answer came back wrong for that reason has been told
    something false about their own knowledge. One correction round is cheap
    against that.

    Valid JSON is not enough: a reply that leaves an answer out (``{}``, or one id
    missing) would record that answer as wrong just the same, so it gets the same
    correction round. It must hold exactly one grade per requested id.
    """
    messages = _build_prompt(to_grade)
    for attempt in range(_MAX_GRADING_ATTEMPTS):
        raw = llm.generate(messages)
        try:
            return _parse_grades(raw, to_grade)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Could not parse grade-answers output (attempt %d): %s",
                attempt + 1,
                exc,
            )
            if attempt + 1 == _MAX_GRADING_ATTEMPTS:
                break
            messages = [*messages, *_correction(raw, exc)]
    return {}
