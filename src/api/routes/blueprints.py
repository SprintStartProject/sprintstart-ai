"""Generation API for AI-proposed onboarding blueprints.

The AI service is stateless: this router runs the batch generation job over the
ingested corpus and returns the resulting blueprints. The backend owns
persistence, versioning, and rollback — it passes its currently-active
blueprints in on every request so generation can number versions and skip an
unchanged corpus.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from api.dependencies import get_llm, get_store
from api.schemas import (
    GenerateBlueprintsRequest,
    ValidationErrorResponse,
)
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.generation import GenerationOutcome, generate_blueprints
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/blueprints", tags=["onboarding-blueprints"])


class GenerateResponse(BaseModel):
    outcomes: list[GenerationOutcome]


@router.post(
    "/generate",
    response_model=GenerateResponse,
    summary="Draft/update blueprints from the corpus",
    description=(
        "Runs the batch generation job over the project's ingested corpus and "
        "returns `source: generated` blueprints for the backend to persist. "
        "Generated scopes are project-qualified "
        "(`project:<projectId>|global`, `project:<projectId>|area:<name>`) and "
        "only this project's chunks are used as evidence. The job is "
        "idempotent given the backend's active blueprints: an unchanged corpus "
        "yields an `unchanged` outcome with no new blueprint.\n\n"
        "**Outcome statuses.** `created`/`updated` carry a new blueprint to "
        "persist; `escalated` does too, but the draft would have removed or "
        "downgraded a human-owned step, so it needs review. `unchanged` and "
        "`skipped` mean there is nothing to do — leave the active blueprint as "
        "it is.\n\n"
        "`invalidated` is the one outcome that requires action without "
        "returning a blueprint: none of the evidence the active *generated* "
        "blueprint was grounded in is retrievable for this project any more "
        "(its artifacts were moved to another project, reclassified, or "
        "deleted). Deactivate that scope's generated blueprint and stop serving "
        "its citations — continuing to serve them exposes content the project "
        "can no longer see. Human-authored blueprints are never invalidated "
        "this way; a scope with nothing generated to withdraw reports "
        "`skipped` instead.\n\n"
        "This is a heavyweight, schedulable operation (one retrieval + LLM pass "
        "per scope); it is not on the onboarding request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during generation.",
        }
    },
)
def generate(
    request: GenerateBlueprintsRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> GenerateResponse:
    try:
        active = [b.to_model() for b in request.active]
        outcomes = generate_blueprints(
            llm,
            store,
            project_id=request.project_id,
            scopes=request.scopes,
            active=active,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blueprint generation failed: {exc}",
        ) from exc
    return GenerateResponse(outcomes=outcomes)
