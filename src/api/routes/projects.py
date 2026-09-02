"""Project-level AI evaluation endpoints."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm, get_store
from api.schemas import (
    IndustryEvaluationRequest,
    IndustryEvaluationResponse,
    ValidationErrorResponse,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from projects.evaluation import evaluate_industry
from store.base import VectorStore

router = APIRouter(prefix="/projects", tags=["projects"])


@router.post(
    "/{project_id}/industry/evaluate",
    response_model=IndustryEvaluationResponse,
    summary="Evaluate project industry/domain from ingested artifacts",
    description=(
        "Retrieves project chunks via RAG and prompts the LLM to identify the "
        "industry/domain as free text, along with a confidence rating and evidence. "
        "Scoping is fail-closed per project_id. Degrades gracefully to an empty "
        "response on parse errors or when no evidence is found."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during evaluation.",
        }
    },
)
def evaluate_project_industry(
    project_id: str,
    request: IndustryEvaluationRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> IndustryEvaluationResponse:
    try:
        return evaluate_industry(llm, store, project_id=project_id)
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Industry evaluation failed: {exc}",
        ) from exc
