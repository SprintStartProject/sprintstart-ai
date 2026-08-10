"""Assembly API for subject-scoped diagrams.

The AI service is stateless: this router draws one diagram of one subject over
the ingested corpus and returns it. The backend owns caching against the subject
and the corpus fingerprint it was drawn from, and stores only the *question* —
the picture is re-derived per read so a diagram cannot describe code that moved.

Like orientation and unlike module proposal, this sits on a hire's request path:
a board card hydrates on every page load. That is why `last_fingerprint` matters
here more than anywhere else — an unchanged corpus must cost no generation.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_store
from api.schemas import AssembleDiagramRequest, ValidationErrorResponse
from api.sse import stream_progress
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.diagram import assemble_diagram, clamp_subject, stream_diagram
from onboarding.diagram_models import DiagramOutcome
from store.base import VectorStore

router = APIRouter(prefix="/onboarding/diagram", tags=["onboarding-diagram"])


def _require_subject(subject: str) -> str:
    clamped = clamp_subject(subject)
    if not clamped:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="subject must not be empty; a diagram is scoped to a question.",
        )
    return clamped


@router.post(
    "",
    response_model=DiagramOutcome,
    summary="Assemble a subject-scoped diagram",
    description=(
        "Draws what the project's own material already shows about one subject: "
        "typed nodes for the parts, typed edges for how they connect. Nothing is "
        "authored -- every node carries the chunks it came from, and a node that "
        "cites nothing is dropped along with the edges that reached it.\n\n"
        "The subject is the only part a model chooses, and it aims retrieval "
        "rather than being asserted. An empty corpus, no retrieved evidence, an "
        "unreadable generation, or fewer than two connected parts surviving all "
        "return `skipped` with no diagram -- the caller must show that as an "
        "honest empty state and never as an explanation."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable during assembly.",
        }
    },
)
def assemble(
    request: AssembleDiagramRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> DiagramOutcome:
    subject = _require_subject(request.subject)
    try:
        return assemble_diagram(
            llm,
            store,
            subject=subject,
            last_fingerprint=request.last_fingerprint,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Diagram assembly failed: {exc}",
        ) from exc


@router.post(
    "/stream",
    response_class=StreamingResponse,
    summary="Assemble a subject-scoped diagram (streaming)",
    description=(
        "The same assembly as `POST /onboarding/diagram`, streamed as "
        "Server-Sent Events so a caller can watch the picture come together: a "
        "`stage` per facet of retrieval, an `item` per node as it clears "
        "grounding, and a terminal `done` carrying the whole outcome. The `done` "
        "result is identical to what the non-streaming endpoint returns -- the "
        "stream is a view of the same computation, never a second answer. An LLM "
        "outage arrives as a terminal `error` event, not an HTTP error."
    ),
    responses={422: {"model": ValidationErrorResponse}},
)
def assemble_stream(
    request: AssembleDiagramRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> StreamingResponse:
    subject = _require_subject(request.subject)
    events = stream_diagram(
        llm,
        store,
        subject=subject,
        last_fingerprint=request.last_fingerprint,
    )
    return StreamingResponse(
        stream_progress(events, operation="diagram"),
        media_type="text/event-stream",
    )
