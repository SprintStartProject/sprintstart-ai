"""Proposal API for shared competency modules.

The AI service is stateless: this router proposes one module for one competency
over the ingested corpus and returns it. The backend owns persistence, versioning
and the PM approval that makes a module live — it passes the fingerprint it last
recorded for this competency so an unchanged corpus doesn't churn a module a PM
has already edited.

Nothing about an individual hire is accepted here, deliberately: one competency
yields one module that everybody reads.
"""

from typing import Annotated, get_args

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_store
from api.schemas import ProposeModuleRequest, ValidationErrorResponse
from api.sse import stream_progress
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.module_models import ModuleLevel, ModuleOutcome
from onboarding.modules import propose_module, stream_module
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/modules", tags=["onboarding-modules"])

_VALID_LEVELS = set(get_args(ModuleLevel))


def _require_level(level: str) -> None:
    if level not in _VALID_LEVELS:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=(
                f"Unknown level {level!r}; expected one of {sorted(_VALID_LEVELS)}"
            ),
        )


@router.post(
    "/propose",
    response_model=ModuleOutcome,
    summary="Propose a shared module for one competency",
    description=(
        "Runs module proposal for a single competency over the ingested corpus "
        "and returns ordered, typed pages plus the module's gating check, for "
        "the backend to store as a proposal awaiting PM approval -- never "
        "auto-applied.\n\n"
        "Grounding is enforced per page: a page making claims about the codebase "
        "without citing evidence is dropped, while the rest of the module is "
        "kept. The job is idempotent given the caller's last recorded "
        "fingerprint: an unchanged corpus yields an `unchanged` outcome, so a "
        "PM's edits are not churned by re-runs.\n\n"
        "This is a heavyweight, schedulable operation (one retrieval + LLM pass); "
        "it is not on the onboarding request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during generation.",
        }
    },
)
def propose(
    request: ProposeModuleRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> ModuleOutcome:
    _require_level(request.level)
    try:
        return propose_module(
            llm,
            store,
            competency_key=request.competency_key,
            competency_label=request.competency_label,
            competency_description=request.competency_description,
            level=request.level,  # type: ignore[arg-type]
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Module proposal failed: {exc}",
        ) from exc


@router.post(
    "/propose/stream",
    response_class=StreamingResponse,
    summary="Propose a shared module for one competency (streaming)",
    description=(
        "The same proposal as `POST /onboarding/modules/propose`, streamed as "
        "Server-Sent Events so a PM can watch it build: `stage` events for "
        "retrieval and generation, an `item` per page as it clears grounding, "
        "and a terminal `done` carrying the whole outcome. The `done` result is "
        "identical to the non-streaming endpoint -- the stream is a view of the "
        "same computation. An LLM outage arrives as a terminal `error` event."
    ),
    responses={422: {"model": ValidationErrorResponse}},
)
def propose_stream(
    request: ProposeModuleRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> StreamingResponse:
    _require_level(request.level)
    events = stream_module(
        llm,
        store,
        competency_key=request.competency_key,
        competency_label=request.competency_label,
        competency_description=request.competency_description,
        level=request.level,  # type: ignore[arg-type]
        last_fingerprint=request.last_fingerprint,
    )
    return StreamingResponse(
        stream_progress(events, operation="module"),
        media_type="text/event-stream",
    )
