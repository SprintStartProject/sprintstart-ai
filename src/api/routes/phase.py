"""Assembly API for AI-enhanced onboarding phases.

A blueprint phase can be ``AI_ENHANCED``: instead of an authored step list it
carries an author's prompt. This router fills such a phase at personalization
time — retrieve the project's own material, then generate grounded steps
(with tasks/resources) and a small knowledge check. The AI service is
stateless; the backend owns persistence and copies the assembled content onto
the user's onboarding path.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_store
from api.schemas import AssemblePhaseRequest, ValidationErrorResponse
from api.sse import stream_progress
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.phase import assemble_phase, stream_phase
from onboarding.phase_models import PhaseOutcome
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/phase", tags=["onboarding-phase"])


def _require(title: str, value: str, field: str) -> str:
    stripped = value.strip()
    if not stripped:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"{field} must not be empty; a phase is scoped to its prompt.",
        )
    return stripped


@router.post(
    "",
    response_model=PhaseOutcome,
    summary="Assemble one AI-enhanced onboarding phase's content",
    description=(
        "Fills a blueprint phase that carries an author's prompt rather than a "
        "fixed step list: retrieves the project's own material, then builds "
        "actionable steps (with tasks and resources) and a small knowledge "
        "check. Nothing is authored -- a step that cites no retrieved chunk is "
        "dropped, so an empty corpus, no retrieved evidence, or a generation "
        "whose every step was ungrounded all return `skipped` with no content. "
        "The backend must persist that as an honest empty phase."
    ),
    responses={
        422: {"model": ValidationErrorResponse},
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during assembly.",
        },
    },
)
def assemble(
    request: AssemblePhaseRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> PhaseOutcome:
    phase_title = _require("phase_title", request.phase_title, "phase_title")
    phase_prompt = _require("phase_prompt", request.phase_prompt, "phase_prompt")
    project_id = _require("project_id", request.project_id, "project_id")
    try:
        return assemble_phase(
            llm,
            store,
            phase_title=phase_title,
            phase_description=request.phase_description,
            phase_prompt=phase_prompt,
            project_id=project_id,
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Phase assembly failed: {exc}",
        ) from exc


@router.post(
    "/stream",
    response_class=StreamingResponse,
    summary="Assemble one AI-enhanced onboarding phase's content (streaming)",
    description=(
        "The same assembly as `POST /onboarding/phase`, streamed as "
        "Server-Sent Events so a caller can watch the phase fill: a `stage` per "
        "retrieval/generation step, an `item` per step/question as it clears "
        "grounding, and a terminal `done` carrying the whole outcome. The "
        "`done` result is identical to what the non-streaming endpoint returns "
        "-- the stream is a view of the same computation, never a second "
        "answer. An LLM outage arrives as a terminal `error` event, not an "
        "HTTP error."
    ),
    responses={422: {"model": ValidationErrorResponse}},
)
def assemble_stream(
    request: AssemblePhaseRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> StreamingResponse:
    phase_title = _require("phase_title", request.phase_title, "phase_title")
    phase_prompt = _require("phase_prompt", request.phase_prompt, "phase_prompt")
    project_id = _require("project_id", request.project_id, "project_id")
    events = stream_phase(
        llm,
        store,
        phase_title=phase_title,
        phase_description=request.phase_description,
        phase_prompt=phase_prompt,
        project_id=project_id,
        last_fingerprint=request.last_fingerprint,
    )
    return StreamingResponse(
        stream_progress(events, operation="phase"),
        media_type="text/event-stream",
    )
