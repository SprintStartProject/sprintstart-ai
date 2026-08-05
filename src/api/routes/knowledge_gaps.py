from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from api.schemas import (
    KnowledgeGapSchema,
    KnowledgeGapsRequest,
    KnowledgeGapsResponse,
)
from ingestion.metadata_store import IngestionMetadataStore
from insights.knowledge_gaps import detect_knowledge_gaps
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter(prefix="/insights/knowledge-gaps", tags=["insights"])


@router.post(
    "/detect",
    response_model=KnowledgeGapsResponse,
    summary="Detect documentation-coverage gaps per component",
    description=(
        "PM-only. Enumerates the components of one project known to the "
        "ingestion index and, for each, reports which expected documentation "
        "categories are missing and how severe the gap is. Called by the "
        "backend's Knowledge-Gaps insight refresh (pull-based); the AI service "
        "is stateless and sources everything from its ingestion index, so the "
        "request carries nothing but the project scope.\n\n"
        "`owners` and `relatedQuestions` are intentionally not returned — the "
        "index holds no user/ownership data and this service retains no question "
        "history; the backend enriches the returned `component` with those."
    ),
)
def detect_gaps(
    body: KnowledgeGapsRequest,
    llm: Annotated[LLMClient, Depends(get_llm)],
    store: Annotated[VectorStore, Depends(get_store)],
    metadata_store: Annotated[
        IngestionMetadataStore, Depends(get_ingestion_metadata_store)
    ],
) -> KnowledgeGapsResponse:
    try:
        gaps = detect_knowledge_gaps(
            llm, store, metadata_store, project_id=body.project_id
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return KnowledgeGapsResponse(
        gaps=[
            KnowledgeGapSchema(
                component=g.component,
                missing_types=g.missing_types,
                present_types=g.present_types,
                last_updated=g.last_updated,
                severity=g.severity,
            )
            for g in gaps
        ]
    )
