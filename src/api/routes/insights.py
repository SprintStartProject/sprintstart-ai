from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    FaqClassifyRequest,
    FaqClassifyResponse,
    FaqConsolidateCategoriesRequest,
    FaqDocumentSchema,
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
    ExistingCategory,
    ExistingGroup,
    Merge,
    classify_question,
    consolidate_categories,
    merge_groups,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter(prefix="/insights/faq", tags=["insights"])


def _to_merge_response(merges: list[Merge]) -> FaqMergeResponse:
    return FaqMergeResponse(
        merges=[FaqMergeSchema(into=m.into, sources=m.sources) for m in merges]
    )


@router.post(
    "/group",
    response_model=FaqGroupResponse,
    summary="Group recurring questions into FAQ clusters",
    description=(
        "PM-only. Semantically groups the given questions, sorts the groups "
        "into topic categories, redacts PII from the sample questions returned "
        "per group, and attaches the documents that answered each group. "
        "Called by the backend's insights refresh (pull-based); the AI service "
        "is stateless, so the backend supplies the full set of questions to "
        "group on every request. This is the full-rebuild path — its cost "
        "grows with the question count, so live updates use /classify instead."
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
            max_categories=body.max_categories,
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
                    FaqSampleQuestionSchema(id=q.id, text=q.text) for q in g.questions
                ],
                documents=[
                    FaqDocumentSchema(id=d.id, title=d.title, source=d.source)
                    for d in g.documents
                ],
                category=g.category,
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
        "PM-only. Places one question into the project's existing FAQ "
        "structure: decides whether it is a real documentation question at "
        "all, which category it belongs to, and whether it joins an existing "
        "group or opens a new one. The returned question text is PII-redacted. "
        "Called by the backend on every AI Buddy interaction, which is why the "
        "prompt is bounded by the FAQ's structure (categories plus candidate "
        "groups) instead of its full question history."
    ),
)
def classify_faq_question(
    body: FaqClassifyRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    metadata_store: IngestionMetadataStore = Depends(get_ingestion_metadata_store),
) -> FaqClassifyResponse:
    categories = [
        ExistingCategory(
            name=c.name,
            group_count=c.group_count,
            question_count=c.question_count,
        )
        for c in body.categories
    ]
    groups = [
        ExistingGroup(
            id=g.id,
            question=g.question,
            category=g.category,
            count=g.count,
        )
        for g in body.groups
    ]

    try:
        result = classify_question(
            body.question,
            categories,
            groups,
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
        category=result.category,
        group_id=result.group_id,
        documents=[
            FaqDocumentSchema(id=d.id, title=d.title, source=d.source)
            for d in result.documents
        ],
    )


@router.post(
    "/categories/consolidate",
    response_model=FaqMergeResponse,
    summary="Propose category merges once the category ceiling is exceeded",
    description=(
        "PM-only. Reduces an over-grown category set back under the given "
        "ceiling by proposing merges of related categories. Only category "
        "names and counts are sent — no question text — so this stays cheap "
        "enough to run whenever the ceiling is crossed. The response is a "
        "plan; the backend applies it and remains the owner of the data."
    ),
)
def consolidate_faq_categories(
    body: FaqConsolidateCategoriesRequest,
    llm: LLMClient = Depends(get_llm),
) -> FaqMergeResponse:
    categories = [
        ExistingCategory(
            name=c.name,
            group_count=c.group_count,
            question_count=c.question_count,
        )
        for c in body.categories
    ]

    try:
        merges = consolidate_categories(categories, body.target_max, llm)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return _to_merge_response(merges)


@router.post(
    "/groups/merge",
    response_model=FaqMergeResponse,
    summary="Propose merges of duplicate groups within one category",
    description=(
        "PM-only. Folds groups that ended up duplicated — the same question "
        "opened twice in different words, which the single-question classify "
        "path can produce — back together. The surviving group id is always "
        "one of the submitted ones, since it keeps the stored samples and "
        "documents. An empty plan means nothing was safely mergeable."
    ),
)
def merge_faq_groups(
    body: FaqMergeGroupsRequest,
    llm: LLMClient = Depends(get_llm),
) -> FaqMergeResponse:
    groups = [
        ExistingGroup(
            id=g.id,
            question=g.question,
            category=g.category,
            count=g.count,
        )
        for g in body.groups
    ]

    try:
        merges = merge_groups(groups, body.target_max, llm)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return _to_merge_response(merges)
