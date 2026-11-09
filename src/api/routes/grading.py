from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import get_llm
from api.schemas import (
    GradeAnswerResult,
    GradeAnswersRequest,
    GradeAnswersResponse,
    ValidationErrorResponse,
)
from assembly import grading
from llm.base import LLMClient
from llm.errors import LLMUnavailableError

router = APIRouter()


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
    answers = [
        grading.AnswerToGrade(
            id=item.id,
            question=item.question,
            reference_answer=item.reference_answer,
            user_answer=item.user_answer,
        )
        for item in body.answers
    ]
    try:
        grades = grading.grade_answers(llm, answers)
    except LLMUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return GradeAnswersResponse(
        results=[
            GradeAnswerResult(
                id=grade.id,
                correct=grade.correct,
                confidence=grade.confidence,
                feedback=grade.feedback,
            )
            for grade in grades
        ]
    )
