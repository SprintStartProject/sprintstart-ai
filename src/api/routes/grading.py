import json
import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ValidationError

from api.dependencies import get_llm
from api.schemas import (
    GradeAnswerItem,
    GradeAnswerResult,
    GradeAnswersRequest,
    GradeAnswersResponse,
    ValidationErrorResponse,
)
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

router = APIRouter()

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


class _GradedItem(BaseModel):
    id: str
    correct: bool
    confidence: float | None = None
    feedback: str = ""


class _Payload(BaseModel):
    results: list[_GradedItem] = []


def _build_prompt(answers: list[GradeAnswerItem]) -> list[Message]:
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


def _grade(llm: LLMClient, to_grade: list[GradeAnswerItem]) -> dict[str, _GradedItem]:
    """Grade in one call, asking once more when the reply is not valid JSON.

    An unreadable reply marks every answer in it "could not be graded" -- which the
    caller records as wrong. A small local model breaks its JSON now and then, and
    a hire whose right answer came back wrong for that reason has been told
    something false about their own knowledge. One correction round is cheap
    against that.
    """
    messages = _build_prompt(to_grade)
    for attempt in range(_MAX_GRADING_ATTEMPTS):
        try:
            raw = llm.generate(messages)
        except LLMUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        try:
            payload = _Payload.model_validate_json(extract_json_object(raw))
            return {item.id: item for item in payload.results}
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "Could not parse grade-answers output (attempt %d): %s",
                attempt + 1,
                exc,
            )
            messages = [
                *messages,
                Message(role="assistant", content=raw),
                Message(
                    role="user",
                    content=(
                        f"That response could not be validated: {exc}. Return the same "
                        "grades again as one valid JSON object matching the schema "
                        "exactly. Return JSON only."
                    ),
                ),
            ]
    return {}


@router.post(
    "/grade-answers",
    summary="Semantically grade short-text knowledge-check answers",
    response_model=GradeAnswersResponse,
    description=(
        "Grades a batch of short-text answers against their reference answers "
        "using the LLM, since exact-match comparison rejects valid paraphrases.\n\n"
        "Behavior:\n"
        "- Answers are graded semantically: wording, casing, and typos don't "
        "matter as long as the core fact is right.\n"
        "- Blank/whitespace-only answers are marked incorrect without an LLM call.\n"
        "- All non-blank answers are graded in a single LLM call for latency.\n"
        "- Results are returned in the same order as the request, correlated by "
        "'id'.\n"
        "- If the LLM's output can't be parsed, ungraded answers are marked "
        "incorrect with generic feedback rather than failing the request.\n\n"
        "This endpoint is synchronous and returns a single JSON response."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "content": {
                "application/json": {
                    "example": {
                        "detail": "LLM backend unreachable at 'http://localhost:11434'"
                    }
                }
            },
        },
    },
)
def grade_answers(
    body: GradeAnswersRequest, llm: LLMClient = Depends(get_llm)
) -> GradeAnswersResponse:
    """
    Semantically grade a batch of short-text knowledge-check answers.

    Args:
        body: The answers to grade, each with its question and reference answer.
        llm: Injected LLM client used to judge each answer.

    Returns:
        GradeAnswersResponse: Per-answer grading results, in request order.

    Raises:
        HTTPException:
            - 503: If the LLM backend is unavailable or unreachable.
    """
    results: dict[str, GradeAnswerResult] = {}
    to_grade = [item for item in body.answers if item.user_answer.strip()]
    for item in body.answers:
        if not item.user_answer.strip():
            results[item.id] = GradeAnswerResult(
                id=item.id, correct=False, feedback="No answer submitted."
            )

    if to_grade:
        graded_by_id = _grade(llm, to_grade)

        for item in to_grade:
            graded = graded_by_id.get(item.id)
            if graded is None:
                results[item.id] = GradeAnswerResult(
                    id=item.id,
                    correct=False,
                    feedback="Could not be graded automatically.",
                )
            else:
                results[item.id] = GradeAnswerResult(
                    id=item.id,
                    correct=graded.correct,
                    confidence=graded.confidence,
                    feedback=graded.feedback,
                )

    return GradeAnswersResponse(results=[results[item.id] for item in body.answers])
