import logging
from collections.abc import Iterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from api.dependencies import (
    OrchestratorFactory,
    get_llm,
    get_orchestrator_factory,
    get_source_state_store,
    get_store,
)
from api.schemas import ChatRequest, ValidationErrorResponse
from api.sse import sse_event
from ingestion.source_state_store import SourceStateStore
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from rag.prompt import build_messages
from rag.retriever import retrieve
from rag.types import RetrievalFilters
from store.base import VectorStore

logger = logging.getLogger(__name__)

router = APIRouter()

_FILTERED_TOP_K = 5
_FILTERED_MIN_SCORE = 0.3

_NO_FILTERED_RESULTS_MESSAGE = (
    "I could not find any matching sources for the selected filters, "
    "so I cannot answer this reliably."
)


def _retrieval_filters_from_request(body: ChatRequest) -> RetrievalFilters:
    """Build the retrieval filters for this request.

    The project scope is always present — it is what keeps one project's
    answers and citations out of another's. Source-system and time bounds are
    optional and only set when the caller asked for them.
    """
    if body.filters is None:
        return RetrievalFilters(project_id=body.project_id)

    return RetrievalFilters(
        project_id=body.project_id,
        source_systems=body.filters.source_systems or None,
        time_from=body.filters.time_from,
        time_to=body.filters.time_to,
    )


def _has_narrowing_filters(body: ChatRequest) -> bool:
    """Whether the caller narrowed the corpus beyond the project scope.

    Only a caller-chosen narrowing switches /chat from the agentic path to the
    single-shot filtered retrieval below; the always-present project scope must
    not, or the agent would never run again.
    """
    if body.filters is None:
        return False

    return bool(
        body.filters.source_systems
        or body.filters.time_from is not None
        or body.filters.time_to is not None
    )


@router.post(
    "/chat",
    summary="Ask a question (streaming)",
    response_class=StreamingResponse,
    responses={422: {"model": ValidationErrorResponse}},
)
def chat(
    body: ChatRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    source_state: SourceStateStore = Depends(get_source_state_store),
    build_orchestrator: OrchestratorFactory = Depends(get_orchestrator_factory),
) -> StreamingResponse:
    def event_stream() -> Iterator[str]:
        try:
            filters = _retrieval_filters_from_request(body)

            if not _has_narrowing_filters(body):
                yield from build_orchestrator(filters).stream(
                    body.question,
                    body.history,
                )
                return

            exclusions = source_state.get_exclusions()

            yield sse_event({"type": "tool_use", "name": "retrieve", "kind": "tool"})

            chunks = retrieve(
                body.question,
                llm,
                store,
                top_k=_FILTERED_TOP_K,
                min_score=_FILTERED_MIN_SCORE,
                exclusions=exclusions,
                filters=filters,
            )

            if not chunks:
                yield sse_event(
                    {
                        "type": "token",
                        "content": _NO_FILTERED_RESULTS_MESSAGE,
                    }
                )
                yield sse_event({"type": "done"})
                return

            history: list[Message] = [
                Message(role=h.role, content=h.content) for h in body.history
            ]
            messages = build_messages(body.question, chunks, history)

            for token in llm.stream(messages):
                if token:
                    yield sse_event({"type": "token", "content": token})

            for citation in build_citations(chunks):
                yield sse_event(
                    {
                        "type": "citation",
                        "artifact_id": citation.artifact_id,
                        "start_line": citation.start_line,
                        "start_page": citation.start_page,
                    }
                )

            yield sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in chat stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
