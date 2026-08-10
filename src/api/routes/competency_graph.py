"""Generation API for AI-proposed competencies.

The AI service is stateless: this router runs the batch proposal job over the
ingested corpus and returns candidate competencies. The backend owns
persistence — it passes its current live vocabulary in on every request so
proposals can be deduplicated against what already exists.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_store
from api.schemas import (
    GenerateCompetencyGraphRequest,
    GraphProposalOutcomeSchema,
    ValidationErrorResponse,
)
from api.sse import stream_progress
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.graph_generation import (
    generate_competency_graph,
    stream_competency_graph,
)
from store.base import VectorStore

router = APIRouter(
    prefix="/onboarding/competency-graph", tags=["onboarding-competency-graph"]
)


@router.post(
    "/propose",
    response_model=GraphProposalOutcomeSchema,
    summary="Propose competencies from the corpus",
    description=(
        "Runs the batch proposal job over the ingested corpus and returns candidate "
        "`SKILL`/`CONCEPT` competencies for the backend to persist.\n\n"
        "The result is a flat vocabulary: it states no ordering between "
        "competencies. The relationship pass that used to draw prerequisite edges "
        "was retired along with the graph it described.\n\n"
        "The caller's last recorded fingerprint makes the run idempotent -- an "
        "unchanged corpus proposes nothing, because competencies are all this "
        "derives and they are a function of the corpus.\n\n"
        "This is a heavyweight, schedulable operation (one retrieval + one LLM pass "
        "over the whole corpus); it is not on the onboarding request path."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during generation.",
        }
    },
)
def propose(
    request: GenerateCompetencyGraphRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> GraphProposalOutcomeSchema:
    try:
        outcome = generate_competency_graph(
            llm,
            store,
            active_competencies=[c.to_model() for c in request.active_competencies],
            existing_areas=request.existing_areas,
            tombstoned_competencies=[
                c.to_model() for c in request.tombstoned_competencies
            ],
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Competency graph proposal failed: {exc}",
        ) from exc
    return GraphProposalOutcomeSchema(**outcome.model_dump())


@router.post(
    "/propose/stream",
    response_class=StreamingResponse,
    summary="Propose competencies from the corpus (streaming)",
    description=(
        "The same proposal job as `POST /onboarding/competency-graph/propose`, "
        "streamed as Server-Sent Events so a PM can watch the vocabulary build: a "
        "`stage` per phase (retrieving → grounding), an `item` per competency as "
        "it clears grounding, and a terminal `done` carrying the whole outcome. "
        "The `done` result is identical to what the non-streaming endpoint returns "
        "-- the stream is a view of the same computation, never a second answer. "
        "An LLM outage arrives as a terminal `error` event, not an HTTP error."
    ),
    responses={422: {"model": ValidationErrorResponse}},
)
def propose_stream(
    request: GenerateCompetencyGraphRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> StreamingResponse:
    events = stream_competency_graph(
        llm,
        store,
        active_competencies=[c.to_model() for c in request.active_competencies],
        existing_areas=request.existing_areas,
        tombstoned_competencies=[c.to_model() for c in request.tombstoned_competencies],
        last_fingerprint=request.last_fingerprint,
    )
    return StreamingResponse(
        stream_progress(events, operation="competency_graph"),
        media_type="text/event-stream",
    )
