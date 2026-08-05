from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    FaqDocumentSchema,
    FaqGroupRequest,
    FaqGroupResponse,
    FaqGroupSchema,
)
from ingestion.metadata_store import IngestionMetadataStore
from insights.faq import FaqQuestionInput, group_faqs
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter(prefix="/insights/faq", tags=["insights"])


@router.post(
    "/group",
    response_model=FaqGroupResponse,
    summary="Group recurring questions into FAQ clusters",
    description=(
        "PM-only. Semantically groups the given questions, redacts PII from "
        "the sample questions returned per group, and attaches the documents "
        "that answered each group. Called by the backend's insights refresh "
        "(pull-based); the AI service is stateless, so the backend supplies "
        "the full set of questions to group on every request."
    ),
)
def group_faq_questions(
    body: FaqGroupRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    metadata_store: IngestionMetadataStore = Depends(get_ingestion_metadata_store),
) -> FaqGroupResponse:
    questions = [FaqQuestionInput(id=q.id, text=q.text) for q in body.questions]

    try:
        groups = group_faqs(
            questions, llm, store, metadata_store, project_id=body.project_id
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return FaqGroupResponse(
        groups=[
            FaqGroupSchema(
                question=g.question,
                count=g.count,
                questions=g.questions,
                documents=[
                    FaqDocumentSchema(id=d.id, title=d.title, source=d.source)
                    for d in g.documents
                ],
            )
            for g in groups
        ]
    )
