"""API endpoints for AI-supported skill suggestions."""

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from api.dependencies import get_llm, get_store
from api.schemas import (
    SkillSuggestionRequest,
    SkillSuggestionResponse,
    ValidationErrorResponse,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from skills.suggestion import suggest_skills
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/skills", tags=["skills"])


@router.post(
    "/suggest",
    response_model=SkillSuggestionResponse,
    summary="Suggest fitting skills for a project role",
    description=(
        "Suggests role-appropriate and project-grounded skills for self-assessment. "
        "Universal skills (soft skills, agile methods) are recommended without "
        "requiring artifact citations; project-specific skills (technologies, "
        "frameworks) are grounded in the project's retrieved artifacts and cite "
        "chunk IDs."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during skill suggestion.",
        }
    },
)
def suggest(
    request: SkillSuggestionRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> SkillSuggestionResponse:
    try:
        return suggest_skills(
            llm=llm,
            store=store,
            project_id=request.project_id,
            role_name=request.role_name,
            role_description=request.role_description,
            project_industry=request.project_industry,
            available_skills=request.available_skills,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        logger.exception("Skill suggestion failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Skill suggestion failed: {exc}",
        ) from exc
