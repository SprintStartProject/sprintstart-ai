import logging
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
from onboarding.buddy_agent import (
    AgentReasoning,
    AgentToken,
    AgentToolUse,
    AgentTurnResult,
    run_agent_turn,
    stream_agent_turn,
)
from onboarding.buddy_compact import compact_memory
from onboarding.buddy_open import stream_session
from onboarding.vocabulary import Vocabulary
from rag.types import RetrievalFilters
from store.base import VectorStore

logger = logging.getLogger(__name__)

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
    if schema.reasoning:
        msg["reasoning"] = schema.reasoning
    if schema.reasoning_details:
        msg["reasoning_details"] = [dict(d) for d in schema.reasoning_details]
    return msg


def _from_message(msg: Message) -> BuddyAgentMessageSchema:
    # The reasoning rides along so the hop after a backend tool can hand it back:
    # a provider with extended thinking rejects a tool turn that lost its signed
    # thinking blocks.
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
        reasoning=msg.get("reasoning"),
        reasoning_details=list(msg.get("reasoning_details") or []),
    )


def _to_toolspec(schema: BuddyToolSpecSchema) -> ToolSpec:
    return ToolSpec(
        name=schema.name,
        description=schema.description,
        parameters=dict(schema.parameters),
    )


def _vocabulary(body: BuddyAgentRequest) -> Vocabulary:
    return Vocabulary(
        contribution_noun=body.vocabulary.contribution_noun,
        contribution_noun_plural=body.vocabulary.contribution_noun_plural,
        contribution_verb_past=body.vocabulary.contribution_verb_past,
    )


def _narrowing(body: BuddyAgentRequest) -> RetrievalFilters | None:
    """The reader's source-system and time narrowing, if they chose any.

    The project scope is not in here: it is ``project_ids`` and always applies.
    An empty source-system list means "all".
    """
    if body.filters is None:
        return None
    return RetrievalFilters(
        source_systems=body.filters.source_systems or None,
        time_from=body.filters.time_from,
        time_to=body.filters.time_to,
    )


def _agent_response(result: AgentTurnResult) -> BuddyAgentResponse:
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
        reasoning=result.reasoning,
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

    Executes the searches locally (retrieval + citations) and returns as soon as it
    either has a final answer or needs a backend-only tool run. The backend carries the
    ``messages`` list back verbatim, each pending tool's result appended as a ``tool``.

    ``capabilities_enabled`` and ``team_mode`` pick the persona. Both are read on every
    hop rather than only the first: the persona is rebuilt each time, so a resume hop
    that omitted one would finish the turn in the other mode.

    The same turn as ``/onboarding/buddy/agent/stream``, buffered into one response.
    """
    try:
        result = run_agent_turn(
            [_to_message(m) for m in body.messages],
            [_to_toolspec(t) for t in body.backend_tools],
            llm,
            store,
            exclusions=source_state.get_exclusions(),
            prior_summary=body.prior_summary,
            vocabulary=_vocabulary(body),
            project_ids=frozenset(body.project_ids),
            capabilities_enabled=body.capabilities_enabled,
            team_mode=body.team_mode,
            filters=_narrowing(body),
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc

    return _agent_response(result)


@router.post(
    "/onboarding/buddy/agent/stream",
    summary="Run one agentic buddy turn, streaming it as it happens",
    response_class=StreamingResponse,
    tags=["onboarding-buddy"],
    responses={422: {"model": ValidationErrorResponse}},
)
def buddy_agent_stream(
    body: BuddyAgentRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    source_state: SourceStateStore = Depends(get_source_state_store),
) -> StreamingResponse:
    """One turn of the tool-using buddy, with the model's work visible as it happens.

    Takes the same request as ``/onboarding/buddy/agent`` and runs the same turn;
    what differs is that nothing waits for the last hop. Emits ``reasoning`` and
    ``token`` events, each carrying a ``content`` fragment, as the model produces
    them; a ``tool_use`` event (``name``, ``arguments``) before each search this
    service runs itself; and one terminal ``result`` carrying the fields of
    ``BuddyAgentResponse`` (``final``, ``text``, ``messages``, ``pending_tool_calls``,
    ``citations``, ``reasoning``). The ``text`` of ``result`` is authoritative: it is
    what the caller stores and carries back, and the ``token`` events spell it out
    except where a turn that streamed text for one hop then searched again, which
    joins its hops with a blank line.

    A model that is unavailable, or any failure mid-turn, becomes a terminal ``error``
    event rather than a truncated body; the status line is already sent by then.
    """
    messages = [_to_message(m) for m in body.messages]
    backend_tools = [_to_toolspec(t) for t in body.backend_tools]
    exclusions = source_state.get_exclusions()
    narrowing = _narrowing(body)

    def event_stream() -> Iterator[str]:
        try:
            for event in stream_agent_turn(
                messages,
                backend_tools,
                llm,
                store,
                exclusions=exclusions,
                prior_summary=body.prior_summary,
                vocabulary=_vocabulary(body),
                project_ids=frozenset(body.project_ids),
                capabilities_enabled=body.capabilities_enabled,
                team_mode=body.team_mode,
                filters=narrowing,
            ):
                if isinstance(event, AgentReasoning):
                    yield sse_event({"type": "reasoning", "content": event.text})
                elif isinstance(event, AgentToken):
                    yield sse_event({"type": "token", "content": event.text})
                elif isinstance(event, AgentToolUse):
                    yield sse_event(
                        {
                            "type": "tool_use",
                            "name": event.name,
                            "arguments": event.arguments,
                        }
                    )
                else:
                    yield sse_event(
                        {
                            "type": "result",
                            **_agent_response(event).model_dump(mode="json"),
                        }
                    )
        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in buddy agent stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")


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

    Nobody is waiting on this call, and that is why it exists separately. The
    Folding inside ``/onboarding/buddy/agent`` instead, ahead of the answer, makes
    every turn past the backend's window cap pay an extra serialized model call to
    compress one exchange before the hire's reply starts. The caller runs this
    *after* a turn finishes.

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

    This writes no memory note. Rewriting one from the same model call would compose
    a hire's durable memory while the model is busy greeting them, leaving the call
    doing two jobs of which the hire can see one. Folding is
    ``/onboarding/buddy/compact``, which the caller runs when nobody is waiting.

    The greeting comes first and is streamed, and that ordering is the feature.
    Asking for strict JSON whose first field is the note the hire never sees would
    mean waiting on up to 200 words of invisible output before the first word
    addressed to the hire is generated.

    Emits ``token`` events carrying the greeting as it arrives and one terminal
    ``done`` carrying the whole greeting and any suggested action. Degrades to a plain
    welcome rather than erroring: opening the buddy must never fail the page.

    With ``team_mode`` the reader is a project's manager and ``state`` is their team's
    attention list, so both the prompt and that plain welcome address a manager.
    """

    def event_stream() -> Iterator[str]:
        try:
            for event in stream_session(
                memory=body.memory,
                recent=[_to_message(m) for m in body.recent],
                state=body.state,
                llm=llm,
                team_mode=body.team_mode,
            ):
                yield sse_event(event)
        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in buddy open stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )

    return StreamingResponse(event_stream(), media_type="text/event-stream")
