"""One agent turn: the hops, the searches they run and the answer they end in."""

from collections.abc import Generator, Iterator
from dataclasses import dataclass, replace

from buddy.conversation import assistant_message, ensure_persona, tool_result_message
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
from buddy.query_fence import fence_user_messages
from buddy.tools.base import ToolResult, run_tool_calls
from buddy.tools.evidence import limit_evidence
from buddy.vocabulary import Vocabulary
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


@dataclass(frozen=True)
class Request:
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


class Turn:
    """The running state of one agent turn.

    Holds the conversation the model is shown, what has been cited, and whether a
    search found anything, and drives the hops: a hop streams the model, then
    either answers, runs the searches it asked for and goes round again, or hands
    a backend tool back.
    """

    def __init__(self, request: Request) -> None:
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
        self.work = ensure_persona(
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
        self.work = [*self.work, assistant_message(result)]

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
                    tool_result_message(call.id, f"Unknown tool: {call.name}."),
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
        self.work = [*self.work, tool_result_message(call.id, content)]

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
