"""Agentic onboarding buddy: one tool-using turn.

The buddy reasons over the hire's question and calls tools -- some it runs itself,
some only the backend can. ``search_docs`` (and ``grep``, when the reader asked the
corpus rather than the mentor) are AI-local -- they own retrieval and citations --
and are executed here, in an internal loop, so a question needing several searches
is answered in one call. They are the chat agent's own tools and evidence budget
(``buddy.tools``), so answering from the buddy loses nothing chat had. A tool only
the backend can run (``get_my_metrics``) cannot be executed here: the turn stops and
hands the pending call back, and the backend re-invokes this endpoint with the tool
result appended.

Every user message is fenced off from the persona's rules (``buddy.query_fence``)
before the model sees it.

Stateless like every other onboarding endpoint: the caller (backend) carries the
running message list between invocations. Nothing about the hire lives here -- their
state arrives only as tool results the backend supplies.

Session memory: the backend bounds an unbounded transcript by sending only a recent
window plus a running summary of everything older (``prior_summary``). The summary
rides the system message in the returned conversation, so resume hops need nothing
re-sent.

Folding older turns into that summary is ``buddy.compact``, on its own
endpoint, which the backend runs after a turn rather than during one -- see
``run_agent_turn`` for what that cost while it lived here.

The loop is ``stream_agent_turn``, a generator, so a reader sees the model think and
write while the turn runs instead of after its last hop. ``run_agent_turn`` drains it
for the callers that only want the outcome.
"""

from collections.abc import Collection, Generator, Iterator
from dataclasses import dataclass, replace

from buddy.events import (
    AgentReasoning,
    AgentStreamEvent,
    AgentToolUse,
    AgentTurnResult,
    HopText,
)
from buddy.evidence import (
    format_result,
    local_tools,
    without_test_material,
)
from buddy.persona import build_persona
from buddy.query_fence import QUERY_FENCE_NOTE, fence_user_messages
from buddy.tools.base import ToolResult, run_tool_calls
from buddy.tools.evidence import limit_evidence
from buddy.vocabulary import DEFAULT_VOCABULARY, Vocabulary
from corpus.filters import NO_FILTERED_RESULTS_MESSAGE, has_narrowing_filters
from corpus.source_filter import SourceExclusions
from corpus.types import Citation, RetrievalFilters
from llm.base import (
    ChatResult,
    LLMClient,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from store.base import VectorStore

# How the summary enters the model's context: appended to the persona in the system
# message, so it rides the running conversation the caller carries between turns.
_SUMMARY_HEADER = "\n\nConversation so far (compressed memory of earlier turns):\n"

# How many internal search hops before we force a final answer, so a confused model
# can't loop forever gathering evidence it never uses.
#
# Raised from 4 after a testing session: only *search-only* hops consume this budget (a
# hop that asks for a backend tool returns immediately), and a model that searched four
# times before deciding to act never got to act at all -- it hit the forced answer
# below, which has no tools, and promised the hire a button that could not exist. Six
# leaves room for a thorough answer and still bounds the loop.
_MAX_STEPS = 6

# What the model is told when the search budget is spent.
#
# It answers the hire's question with the persona still in front of it -- "offer
# `add_path_step`", "offer to claim it" -- and without this it took those at face value
# and told the hire to confirm something no tool call had produced. A model that knows
# it has no tools this turn can only do the honest thing: answer with what it has and
# leave the offer for the next turn.
_NO_TOOLS_THIS_TURN = (
    "You have no tools available for this reply and cannot do anything or offer "
    "anything: no confirm button can appear. Answer with what you already have. Do not "
    "say you have done something, do not tell the hire to confirm, click or check "
    "anything, and do not claim anything is on their screen. If something still needs "
    "doing, say what it is and that you can set it up when they reply."
)


def _persona_prompt(
    summary: str | None,
    tool_names: Collection[str],
    vocabulary: Vocabulary,
    capabilities_enabled: bool,
    team_mode: bool,
) -> str:
    persona = build_persona(
        tool_names,
        vocabulary,
        capabilities_enabled=capabilities_enabled,
        team_mode=team_mode,
    )
    # Every mode and every hop: the fence is applied to every user message, so the
    # rule for reading it has to be in front of the model whenever one is.
    persona = persona.rstrip("\n") + "\n" + QUERY_FENCE_NOTE
    if not summary:
        return persona
    return persona + _SUMMARY_HEADER + summary


def _extract_summary(system_content: str) -> str | None:
    if _SUMMARY_HEADER in system_content:
        return system_content.split(_SUMMARY_HEADER, 1)[1]
    return None


def _ensure_persona(
    messages: list[Message],
    summary: str | None,
    tool_names: Collection[str],
    vocabulary: Vocabulary,
    capabilities_enabled: bool,
    team_mode: bool,
) -> list[Message]:
    # The system message a resume carries is replaced, not kept: only its summary
    # survives. So both modes have to arrive on every hop -- a hop that dropped one
    # would rebuild the default persona mid-turn and answer a manager as a hire.
    effective_summary = summary
    rest = messages
    if messages and messages[0]["role"] == "system":
        if effective_summary is None:
            effective_summary = _extract_summary(messages[0].get("content") or "")
        rest = messages[1:]
    persona_msg = Message(
        role="system",
        content=_persona_prompt(
            effective_summary,
            tool_names,
            vocabulary,
            capabilities_enabled,
            team_mode,
        ),
    )
    return [persona_msg, *rest]


def _assistant_message(result: ChatResult) -> Message:
    msg = Message(role="assistant", content=result.text)
    if result.tool_calls:
        msg["tool_calls"] = result.tool_calls
    # Kept for the next hop: a reasoning provider continues a tool-using thought
    # from these, and one with extended thinking rejects the turn without them.
    # They also go out on the wire message (``api.routes.buddy._from_message``),
    # so the hop after a backend tool gets them back from the caller.
    if result.reasoning:
        msg["reasoning"] = result.reasoning
    if result.reasoning_details:
        msg["reasoning_details"] = result.reasoning_details
    return msg


def _tool_result_message(call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id)


@dataclass(frozen=True)
class _Request:
    messages: list[Message]
    backend_tools: list[ToolSpec]
    llm: LLMClient
    store: VectorStore
    exclusions: SourceExclusions | None
    prior_summary: str | None
    vocabulary: Vocabulary
    project_ids: frozenset[str] | None
    capabilities_enabled: bool
    team_mode: bool
    filters: RetrievalFilters | None


class _Turn:
    """The running state of one agent turn.

    Holds the conversation the model is shown, what has been cited, and whether a
    search found anything, and drives the hops: a hop streams the model, then
    either answers, runs the searches it asked for and goes round again, or hands
    a backend tool back.
    """

    def __init__(self, request: _Request) -> None:
        self._llm = request.llm
        self._filters = request.filters
        scope = replace(
            request.filters if request.filters is not None else RetrievalFilters(),
            project_id=None,
            # Scoped to the projects this hire is on, so the mentor cannot quote
            # another team's material as this team's -- and cannot hide the hire's
            # own second project either. Fail-closed, like every project scope:
            # material belonging to no project matches none (`corpus.filters`).
            project_ids=request.project_ids,
        )
        self._registry = local_tools(
            request.llm,
            request.store,
            request.exclusions
            if request.exclusions is not None
            else SourceExclusions(),
            scope,
            with_grep=not request.capabilities_enabled,
        )
        self._local_names = self._registry.names()
        backend = [
            t for t in request.backend_tools if t["name"] not in self._local_names
        ]
        self._backend_names = {tool["name"] for tool in backend}
        self._tools = [*self._registry.specs(), *backend]
        # The persona describes exactly the tools this hire was mounted, never a fixed
        # catalogue: the backend decides what a given role can even have, and a mentor
        # told about a tool it does not have will offer the hire something impossible.
        self.work = _ensure_persona(
            fence_user_messages(list(request.messages)),
            request.prior_summary,
            {*self._local_names, *self._backend_names},
            request.vocabulary,
            request.capabilities_enabled,
            request.team_mode,
        )
        self._citations: list[Citation] = []
        self._seen_chunk_ids: set[str] = set()
        self._reasoning: list[str] = []
        self._searched = False
        self._found = False
        self._text = HopText()

    def run(self) -> Iterator[AgentStreamEvent]:
        for _ in range(_MAX_STEPS):
            if (yield from self._hop()):
                return
        yield from self._forced_answer()

    # Nothing matched under an explicit narrowing: the reader chose the filter, so an
    # answer composed from no sources is the one thing not to give.
    def _overridable(self) -> bool:
        narrowed = has_narrowing_filters(self._filters)
        return (
            narrowed and self._searched and not self._found and not self._backend_names
        )

    def _answer(self, text: str) -> AgentTurnResult:
        if self._overridable():
            text = NO_FILTERED_RESULTS_MESSAGE
        return AgentTurnResult(
            final=True,
            text=text,
            messages=[*self.work, Message(role="assistant", content=text)],
            citations=self._citations,
            reasoning=self._reasoning,
        )

    def _hop(self) -> Generator[AgentStreamEvent, None, bool]:
        """One round with the model; whether the turn is over."""
        self._text.begin_hop()
        # A hop that can end as the filter message must not have shown the model's
        # own text first, so it is held until the hop says which it is.
        hold = self._overridable()
        result, held = yield from self._stream_model(hold)

        if not result.tool_calls:
            answer = self._answer(result.text)
            if hold:
                yield from self._text.token(answer.text)
            yield answer
            return True
        # Not the answer after all: the held text was narration before a search.
        for fragment in held:
            yield from self._text.token(fragment)
        self.work = [*self.work, _assistant_message(result)]

        outcomes = yield from self._search(result)
        pending = self._take_results(result, outcomes)
        # A tool only the backend can run: stop and hand it back. Any local searches
        # in this same turn already have their results appended, so the message list
        # stays well-formed.
        if pending:
            yield AgentTurnResult(
                final=False,
                text=result.text,
                messages=self.work,
                pending_tool_calls=pending,
                citations=self._citations,
                reasoning=self._reasoning,
            )
            return True
        # Only local searches this turn -- go round again and let the model reason
        # over them.
        return False

    def _stream_model(
        self, hold: bool
    ) -> Generator[AgentStreamEvent, None, tuple[ChatResult, list[str]]]:
        held: list[str] = []
        result: ChatResult | None = None
        for event in self._llm.chat_stream(self.work, self._tools):
            if isinstance(event, ReasoningDelta):
                if event.text:
                    yield AgentReasoning(event.text)
            elif isinstance(event, TextDelta):
                if hold:
                    held.append(event.text)
                else:
                    yield from self._text.token(event.text)
            else:
                result = event
        if result is None:
            raise LLMUnavailableError("The model ended its reply without a result.")
        if result.reasoning and result.reasoning.strip():
            self._reasoning.append(result.reasoning)
        return result, held

    def _search(
        self, result: ChatResult
    ) -> Generator[AgentToolUse, None, dict[str, ToolResult]]:
        """Run the searches the model asked for, each announced before it runs."""
        local_calls = [c for c in result.tool_calls if c.name in self._local_names]
        for call in local_calls:
            yield AgentToolUse(call.name, dict(call.arguments))
        outcomes = dict(
            zip(
                (c.id for c in local_calls),
                run_tool_calls(self._registry, local_calls) if local_calls else [],
                strict=True,
            )
        )
        self._searched = self._searched or bool(local_calls)
        return outcomes

    def _take_results(
        self, result: ChatResult, outcomes: dict[str, ToolResult]
    ) -> list[ToolCall]:
        """Answer every call in order; return the ones only the backend can run."""
        pending: list[ToolCall] = []
        for call in result.tool_calls:
            outcome = outcomes.get(call.id)
            if outcome is not None:
                self._record(call, outcome)
            elif call.name in self._backend_names:
                pending.append(call)
            else:
                self.work = [
                    *self.work,
                    _tool_result_message(call.id, f"Unknown tool: {call.name}."),
                ]
        return pending

    def _record(self, call: ToolCall, outcome: ToolResult) -> None:
        screened = without_test_material(outcome)
        chunks = limit_evidence(screened.chunks)
        # Cited after the drop and the budget, and only the direct matches:
        # nothing the model was not shown, and no context-only neighbour,
        # is offered to the hire as a source. Deduped by chunk id.
        matched = screened.matched_chunks(chunks)
        self._found = self._found or bool(matched)
        fresh = [c for c in matched if c.id not in self._seen_chunk_ids]
        self._seen_chunk_ids.update(c.id for c in fresh)
        self._citations.extend(build_citations(fresh))
        content = format_result(screened, chunks)
        self.work = [*self.work, _tool_result_message(call.id, content)]

    def _forced_answer(self) -> Iterator[AgentStreamEvent]:
        """Step budget spent: force a final answer with no tools, not loop forever.

        The notice goes to the model, not into the returned conversation: a final
        turn's messages are discarded by the caller, and a transcript carrying
        instructions about a budget nobody can see would be a strange thing to keep.
        """
        self._text.begin_hop()
        hold = self._overridable()
        forced = ""
        forced_reasoning = ""
        notice = Message(role="system", content=_NO_TOOLS_THIS_TURN)
        for event in self._llm.stream([*self.work, notice]):
            if isinstance(event, ReasoningDelta):
                if event.text:
                    forced_reasoning += event.text
                    yield AgentReasoning(event.text)
            else:
                forced += event.text
                if not hold:
                    yield from self._text.token(event.text)
        if forced_reasoning.strip():
            self._reasoning.append(forced_reasoning)
        answer = self._answer(forced)
        if hold:
            yield from self._text.token(answer.text)
        yield answer


def stream_agent_turn(
    messages: list[Message],
    backend_tools: list[ToolSpec],
    llm: LLMClient,
    store: VectorStore,
    exclusions: SourceExclusions | None = None,
    prior_summary: str | None = None,
    vocabulary: Vocabulary = DEFAULT_VOCABULARY,
    project_ids: frozenset[str] | None = None,
    capabilities_enabled: bool = True,
    team_mode: bool = False,
    filters: RetrievalFilters | None = None,
) -> Iterator[AgentStreamEvent]:
    """Runs one agent turn, yielding what the model does as it does it.

    Executes the searches locally and pauses on backend tools. Loops internally while
    the model only asks for local searches (``search_docs``, and ``grep`` when
    capabilities are off); finishes as soon as it either produces a final answer or
    requests a tool only the backend can run. A step budget bounds the internal loop;
    if it's exhausted the model is asked once more with no tools, forcing an answer.

    Yields ``AgentReasoning`` and ``AgentToken`` fragments as the model produces them,
    an ``AgentToolUse`` before each search this loop runs, and as the last event the
    ``AgentTurnResult`` -- the same outcome ``run_agent_turn`` returns. Text from a
    hop that went on to search is streamed like any other, and the next hop's text is
    set apart from it by a blank line, written here and nowhere else.

    ``prior_summary`` stands in for everything older than ``messages``.

    ``capabilities_enabled`` and ``team_mode`` choose the persona and must be passed
    on every hop, because the persona is rebuilt on every hop: a resume that lost
    either would answer the rest of the turn as the default mentor.

    ``filters`` narrows every search by source system and time; its project fields
    are ignored, because the scope is ``project_ids``. When it narrows, this hop
    searched, nothing it found survived, and no backend tool is mounted to have
    supplied other evidence, the answer is ``NO_FILTERED_RESULTS_MESSAGE`` rather
    than whatever the model composed from nothing. Whether a hop can end that way is
    known before it starts, so such a hop's text is held back rather than streamed:
    it is released if the hop goes on to search, and replaced by the message if it is
    the answer.

    This turn does not fold anything, and used not to be able to say that. A
    ``summarize_upto`` argument asked it to compact the oldest window messages
    *before* the model began composing a reply, and because the caller's cursor
    advanced by exactly what was folded, the window sat at its cap forever once it
    first filled -- so every turn of a long visit paid an extra serialized model call,
    ahead of the answer, to compress a single exchange. Folding is
    ``POST /onboarding/buddy/compact``, which the backend runs afterwards.
    """
    request = _Request(
        messages,
        backend_tools,
        llm,
        store,
        exclusions,
        prior_summary,
        vocabulary,
        project_ids,
        capabilities_enabled,
        team_mode,
        filters,
    )
    yield from _Turn(request).run()


def run_agent_turn(
    messages: list[Message],
    backend_tools: list[ToolSpec],
    llm: LLMClient,
    store: VectorStore,
    exclusions: SourceExclusions | None = None,
    prior_summary: str | None = None,
    vocabulary: Vocabulary = DEFAULT_VOCABULARY,
    project_ids: frozenset[str] | None = None,
    capabilities_enabled: bool = True,
    team_mode: bool = False,
    filters: RetrievalFilters | None = None,
) -> AgentTurnResult:
    """Runs one agent turn to its outcome; see ``stream_agent_turn`` for the rules.

    Same turn, same arguments, with the fragments discarded: what the buffered
    endpoint and the tests want is the ``AgentTurnResult``.
    """
    outcome: AgentTurnResult | None = None
    for event in stream_agent_turn(
        messages,
        backend_tools,
        llm,
        store,
        exclusions=exclusions,
        prior_summary=prior_summary,
        vocabulary=vocabulary,
        project_ids=project_ids,
        capabilities_enabled=capabilities_enabled,
        team_mode=team_mode,
        filters=filters,
    ):
        if isinstance(event, AgentTurnResult):
            outcome = event
    if outcome is None:
        raise LLMUnavailableError("The agent turn ended without an outcome.")
    return outcome


__all__ = ["run_agent_turn", "stream_agent_turn"]
