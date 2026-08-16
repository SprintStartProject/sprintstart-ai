from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    FaqClassifyRequest,
    FaqClassifyResponse,
    FaqDocumentSchema,
    FaqGroupRefSchema,
    FaqGroupRequest,
    FaqGroupResponse,
    FaqGroupSchema,
    FaqMergeGroupsRequest,
    FaqMergeResponse,
    FaqMergeSchema,
    FaqSampleQuestionSchema,
)
from ingestion.metadata_store import IngestionMetadataStore
from insights.faq import FaqQuestionInput, group_faqs
from insights.faq_classification import (
    ExistingGroup,
    classify_question,
    merge_groups,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter(prefix="/insights/faq", tags=["insights"])


def _to_existing_groups(groups: list[FaqGroupRefSchema]) -> list[ExistingGroup]:
    return [
        ExistingGroup(id=g.id, question=g.question, title=g.title, count=g.count)
        for g in groups
    ]


@router.post(
    "/group",
    response_model=FaqGroupResponse,
    summary="Group recurring questions into FAQ entries",
    description=(
        "PM-only. Semantically groups the given questions, gives each group a "
        "short title naming what it is about, redacts PII from the sample "
        "questions returned per group, and attaches the documents that "
        "answered each group. Called by the backend's insights refresh "
        "(pull-based); the AI service is stateless, so the backend supplies "
        "the full set of questions to group on every request. This is the "
        "full-rebuild path — its cost grows with the question count, so live "
        "updates use /classify instead."
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
            questions,
            llm,
            store,
            metadata_store,
            project_id=body.project_id,
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
                questions=[
                    FaqSampleQuestionSchema(ids=q.ids, text=q.text) for q in g.questions
                ],
                documents=[
                    FaqDocumentSchema(id=d.id, title=d.title, source=d.source)
                    for d in g.documents
                ],
                title=g.title,
                question_ids=g.question_ids,
            )
            for g in groups
        ]
    )


@router.post(
    "/classify",
    response_model=FaqClassifyResponse,
    summary="File a single freshly asked question into the FAQ",
    description=(
        "PM-only. Places one question into the project's existing FAQ: decides "
        "whether it is a real documentation question at all, and whether it "
        "joins an existing entry or opens a new one with its own title. The "
        "returned question text is PII-redacted. Called by the backend on "
        "every chat question, which is why the prompt is bounded by a "
        "candidate list of entries instead of the full question history."
    ),
)
def classify_faq_question(
    body: FaqClassifyRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    metadata_store: IngestionMetadataStore = Depends(get_ingestion_metadata_store),
) -> FaqClassifyResponse:
    try:
        result = classify_question(
            body.question,
            _to_existing_groups(body.groups),
            llm,
            store,
            metadata_store,
            project_id=body.project_id,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return FaqClassifyResponse(
        relevant=result.relevant,
        question=result.question,
        title=result.title,
        group_id=result.group_id,
        documents=[
            FaqDocumentSchema(id=d.id, title=d.title, source=d.source)
            for d in result.documents
        ],
    )


@router.post(
    "/groups/merge",
    response_model=FaqMergeResponse,
    summary="Propose merges of duplicate FAQ entries",
    description=(
        "PM-only. Folds entries that ended up duplicated — the same question "
        "opened twice in different words, which the single-question classify "
        "path can produce — back together. The surviving id is always one of "
        "the submitted ones, since it keeps the stored samples, title and "
        "documents. An empty plan means nothing was safely mergeable."
    ),
)
def merge_faq_groups(
    body: FaqMergeGroupsRequest,
    llm: LLMClient = Depends(get_llm),
) -> FaqMergeResponse:
    try:
        merges = merge_groups(_to_existing_groups(body.groups), body.target_max, llm)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return FaqMergeResponse(
        merges=[FaqMergeSchema(into=m.into, sources=m.sources) for m in merges]
    )
