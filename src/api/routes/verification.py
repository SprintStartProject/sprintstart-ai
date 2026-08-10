"""Grading API for a graph node's "Verify" zone.

On the hire's request path -- the backend calls this synchronously per
verification attempt (Seam 1). ``knowledge``/``artifact`` grading each make
one LLM call; ``exact``/``attest`` make none.
"""

from typing import Annotated, get_args

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm
from api.schemas import ValidationErrorResponse, VerifyRequest
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.verification import (
    ArtifactEvidence,
    GradeResult,
    GradingType,
    grade_artifact,
    grade_attest,
    grade_exact,
    grade_knowledge,
)

router = APIRouter(prefix="/onboarding/verify", tags=["onboarding-verification"])

_VALID_TYPES = set(get_args(GradingType))


@router.post(
    "",
    response_model=GradeResult,
    summary="Grade one verification attempt",
    description=(
        "Grades a single verification attempt by `type`:\n\n"
        "- `knowledge`: LLM-judge against `rubric` + `evidence`, with an escalating "
        "`hint` on fail keyed off `attempt_no`.\n"
        "- `exact`: normalized string match against `canonical_answer`, no LLM call.\n"
        "- `attest`: self-confirmation -- logs a non-blank `answer` as passed, no "
        "LLM call, no judgement.\n"
        "- `artifact`: LLM-judge whether `artifact_evidence` (PR/repo state the "
        "backend already gathered) satisfies `rubric`. No evidence or explicit "
        "failing CI short-circuits to a fail with no LLM call.\n\n"
        "This is synchronous and on the hire's request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable (knowledge/artifact grading only).",
        }
    },
)
def verify(
    request: VerifyRequest, llm: Annotated[LLMClient, Depends(get_llm)]
) -> GradeResult:
    if request.type not in _VALID_TYPES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Unknown type {request.type!r}; expected one of {sorted(_VALID_TYPES)}"
            ),
        )

    if request.type == "attest":
        return grade_attest(answer=request.answer)

    if request.type == "exact":
        if request.canonical_answer is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="'canonical_answer' is required for 'exact' grading.",
            )
        return grade_exact(
            canonical_answer=request.canonical_answer, answer=request.answer
        )

    if request.type == "artifact":
        if request.rubric is None:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="'rubric' is required for 'artifact' grading.",
            )
        evidence = (
            request.artifact_evidence.to_model()
            if request.artifact_evidence is not None
            else ArtifactEvidence()
        )
        try:
            return grade_artifact(
                llm,
                task_description=request.question,
                rubric=request.rubric,
                evidence=evidence,
            )
        except LLMUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
            ) from exc

    if request.rubric is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="'rubric' is required for 'knowledge' grading.",
        )
    try:
        return grade_knowledge(
            llm,
            question=request.question,
            rubric=request.rubric,
            evidence=request.evidence,
            answer=request.answer,
            attempt_no=request.attempt_no,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
