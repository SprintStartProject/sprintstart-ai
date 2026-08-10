from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_source_state_store, get_store
from api.schemas import (
    BuddyAgentMessageSchema,
    BuddyAgentRequest,
    BuddyAgentResponse,
    BuddyCitationSchema,
    BuddyCompactRequest,
    BuddyCompactResponse,
    BuddyOpenRequest,
    BuddyToolCallSchema,
    BuddyToolSpecSchema,
    ValidationErrorResponse,
)
from api.sse import sse_event
from ingestion.source_state_store import SourceStateStore
from llm.base import LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from onboarding.buddy_agent import run_agent_turn
from onboarding.buddy_compact import compact_memory
from onboarding.buddy_open import stream_session
from onboarding.vocabulary import Vocabulary
from store.base import VectorStore

router = APIRouter()


def _to_message(schema: BuddyAgentMessageSchema) -> Message:
    msg = Message(role=schema.role, content=schema.content)
    if schema.tool_calls:
        msg["tool_calls"] = [
            ToolCall(id=call.id, name=call.name, arguments=dict(call.arguments))
            for call in schema.tool_calls
        ]
    if schema.tool_call_id is not None:
        msg["tool_call_id"] = schema.tool_call_id
    return msg


def _from_message(msg: Message) -> BuddyAgentMessageSchema:
    return BuddyAgentMessageSchema(
        role=msg["role"],
        content=msg.get("content") or "",
        tool_calls=[
            BuddyToolCallSchema(
                id=call.id, name=call.name, arguments=dict(call.arguments)
            )
            for call in msg.get("tool_calls") or []
        ],
        tool_call_id=msg.get("tool_call_id"),
    )


def _to_toolspec(schema: BuddyToolSpecSchema) -> ToolSpec:
    return ToolSpec(
        name=schema.name,
        description=schema.description,
        parameters=dict(schema.parameters),
    )


@router.post(
    "/onboarding/buddy/agent",
    response_model=BuddyAgentResponse,
    summary="Run one agentic buddy turn (tool-using, stateless)",
    tags=["onboarding-buddy"],
    responses={422: {"model": ValidationErrorResponse}},
)
def buddy_agent(
    body: BuddyAgentRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    source_state: SourceStateStore = Depends(get_source_state_store),
) -> BuddyAgentResponse:
    """One turn of the tool-using buddy.

    Executes ``search_docs`` locally (retrieval + citations) and returns as soon as it
    either has a final answer or needs a backend-only tool run. The backend carries the
    ``messages`` list back verbatim, each pending tool's result appended as a ``tool``.
    """
    messages = [_to_message(m) for m in body.messages]
    backend_tools = [_to_toolspec(t) for t in body.backend_tools]
    try:
        result = run_agent_turn(
            messages,
            backend_tools,
            llm,
            store,
            exclusions=source_state.get_exclusions(),
            prior_summary=body.prior_summary,
            vocabulary=Vocabulary(
                contribution_noun=body.vocabulary.contribution_noun,
                contribution_noun_plural=body.vocabulary.contribution_noun_plural,
                contribution_verb_past=body.vocabulary.contribution_verb_past,
            ),
            project_ids=frozenset(body.project_ids) or None,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return BuddyAgentResponse(
        final=result.final,
        text=result.text,
        messages=[_from_message(m) for m in result.messages],
        pending_tool_calls=[
            BuddyToolCallSchema(
                id=call.id, name=call.name, arguments=dict(call.arguments)
            )
            for call in result.pending_tool_calls
        ],
        citations=[
            BuddyCitationSchema(
                artifact_id=cit.artifact_id,
                start_line=cit.start_line,
                start_page=cit.start_page,
            )
            for cit in result.citations
        ],
    )


@router.post(
    "/onboarding/buddy/compact",
    response_model=BuddyCompactResponse,
    summary="Fold older turns into the mentor's durable memory note",
    tags=["onboarding-buddy"],
    responses={
        422: {"model": ValidationErrorResponse},
        503: {"description": "The model was unavailable; nothing was folded."},
    },
)
def buddy_compact(
    body: BuddyCompactRequest,
    llm: LLMClient = Depends(get_llm),
) -> BuddyCompactResponse:
    """Rewrite the mentor's memory note to cover ``folded`` as well.

    ⚠️ **Nobody is waiting on this call, and that is why it exists separately.** The
    same fold used to run only as the first step of ``/onboarding/buddy/agent``, ahead
    of the answer -- so once a visit's window outgrew the backend's cap, every turn
    paid an extra serialized model call to compress one exchange before the hire's
    reply started. The caller now runs this *after* a turn finishes.

    Stateless as ever: the caller owns the note, the transcript and the cursor, and
    advances that cursor by exactly the messages it sent here. **503 means nothing was
    folded** -- the caller keeps its cursor where it is and tries again after the next
    turn, which is why an unavailable model is not worth degrading gracefully over.
    """
    memory = compact_memory(
        prior_summary=body.prior_summary,
        folded=[_to_message(m) for m in body.folded],
        llm=llm,
    )
    if memory is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The model was unavailable; the memory note is unchanged.",
        )
    return BuddyCompactResponse(memory=memory)


@router.post(
    "/onboarding/buddy/open/stream",
    summary="Open a buddy visit: greet the hire",
    response_class=StreamingResponse,
    tags=["onboarding-buddy"],
    responses={422: {"model": ValidationErrorResponse}},
)
def buddy_open_stream(
    body: BuddyOpenRequest,
    llm: LLMClient = Depends(get_llm),
) -> StreamingResponse:
    """Greet the hire opening a visit, streaming the greeting as it is written.

    ⚠️ **This used to rewrite the mentor's durable memory note as well, from the same
    model call.** So a hire's memory was composed while the model was busy greeting
    them, and the call did two jobs of which the hire could see one. Folding is
    ``/onboarding/buddy/compact`` now, which the caller runs when nobody is waiting.

    ⚠️ **The greeting comes first and is streamed, and that ordering is still the
    feature.** An earlier version asked for strict JSON whose first field was the note
    the hire never sees, so opening a visit meant waiting on up to 200 words of
    invisible output before the first word addressed to the hire was generated.

    Emits ``token`` events carrying the greeting as it arrives and one terminal
    ``done`` carrying the whole greeting and any suggested action. Degrades to a plain
    welcome rather than erroring: opening the buddy must never fail the page.
    """

    def event_stream() -> Iterator[str]:
        for event in stream_session(
            memory=body.memory,
            recent=[_to_message(m) for m in body.recent],
            state=body.state,
            llm=llm,
        ):
            yield sse_event(event)

    return StreamingResponse(event_stream(), media_type="text/event-stream")
