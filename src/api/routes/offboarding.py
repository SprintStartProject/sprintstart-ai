from dataclasses import asdict
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm, get_store
from api.interview_schemas import (
    InterviewQuestionSchema,
    InterviewQuestionsRequest,
    InterviewQuestionsResponse,
)
from corpus.ids import ProjectKey
from insights.interview import (
    InterviewLimits,
    InterviewQuestion,
    InterviewRequest,
    plan_interview,
)
from insights.interview_components import RiskFile
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter(prefix="/offboarding", tags=["offboarding"])


def _to_domain(body: InterviewQuestionsRequest) -> InterviewRequest:
    files = [
        RiskFile(
            repository=f.repository,
            path=f.path,
            level=f.level,
            score=f.score,
            kind=f.kind,
            hotspot=f.hotspot,
            abandoned_share=f.abandoned_share,
            coupled_with=tuple(f.coupled_with),
            reasons=tuple(f.reasons),
        )
        for f in body.files
    ]
    limits = InterviewLimits(
        max_components=body.max_components, per_component=body.per_component
    )
    return InterviewRequest(
        project=ProjectKey(body.project_id), files=files, limits=limits
    )


def _to_schema(question: InterviewQuestion) -> InterviewQuestionSchema:
    return InterviewQuestionSchema.model_validate(asdict(question))


@router.post(
    "/interview-questions",
    response_model=InterviewQuestionsResponse,
    response_model_by_alias=True,
    summary="Plan an exit interview from an offboarding risk map",
    description=(
        "PM-only, called by the backend's offboarding module. Groups the files at "
        "risk into components, writes questions that answer what the risk map "
        "found, and checks each against the project's indexed records: questions "
        "the records answer are returned apart, with the answer and its sources. "
        "Stateless; the request names no person, only files and their findings."
    ),
)
def interview_questions(
    body: InterviewQuestionsRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    store: Annotated[VectorStore, Depends(get_store)],
) -> InterviewQuestionsResponse:
    try:
        plan = plan_interview(llm, store, _to_domain(body))
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return InterviewQuestionsResponse(
        questions=[_to_schema(q) for q in plan.questions],
        answered_by_records=[_to_schema(q) for q in plan.answered_by_records],
    )
