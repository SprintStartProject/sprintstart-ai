"""Agentic onboarding buddy: one tool-using turn.

The buddy reasons over the hire's question and calls tools -- some it runs itself,
some only the backend can. ``search_docs`` (and ``grep``, when the reader asked the
corpus rather than the mentor) are AI-local -- they own retrieval and citations --
and are executed here, in an internal loop, so a question needing several searches
is answered in one call. They are the chat agent's own tools and evidence budget
(``agents.tools``), so answering from the buddy loses nothing chat had. A tool only
the backend can run (``get_my_metrics``) cannot be executed here: the turn stops and
hands the pending call back, and the backend re-invokes this endpoint with the tool
result appended.

Every user message is fenced off from the persona's rules (``onboarding.query_fence``)
before the model sees it.

Stateless like every other onboarding endpoint: the caller (backend) carries the
running message list between invocations. Nothing about the hire lives here -- their
state arrives only as tool results the backend supplies.

Session memory: the backend bounds an unbounded transcript by sending only a recent
window plus a running summary of everything older (``prior_summary``). The summary
rides the system message in the returned conversation, so resume hops need nothing
re-sent.

Folding older turns into that summary is ``onboarding.buddy_compact``, on its own
endpoint, which the backend runs after a turn rather than during one -- see
``run_agent_turn`` for what that cost while it lived here.
"""

from collections.abc import Collection, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace

from agents.tools.base import ToolRegistry, ToolResult
from agents.tools.evidence import format_evidence, limit_evidence
from agents.tools.grep import GrepTool
from agents.tools.retrieve import RetrieveTool
from llm.base import ChatResult, LLMClient, Message, ToolCall, ToolSpec
from llm.errors import LLMUnavailableError
from onboarding.buddy_persona import build_persona
from onboarding.query_fence import QUERY_FENCE_NOTE, fence_user_messages
from onboarding.vocabulary import DEFAULT_VOCABULARY, Vocabulary
from rag.citation import build_citations
from rag.prompt import chunk_header
from rag.source_filter import SourceExclusions
from rag.source_kind import is_test_chunk
from rag.types import Citation, RetrievalFilters, ScoredChunk
from store.base import VectorStore

# How the summary enters the model's context: appended to the persona in the system
# message, so it rides the running conversation the caller carries between turns.
_SUMMARY_HEADER = "\n\nConversation so far (compressed memory of earlier turns):\n"

SEARCH_DOCS = "search_docs"

_SEARCH_TOOL: ToolSpec = {
    "name": SEARCH_DOCS,
    "description": (
        "Search the project's indexed documentation, code, issues and pull requests "
        "for grounded evidence. Use this for any question about how the codebase, "
        "product, or process works."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, phrased as a search query.",
            }
        },
        "required": ["query"],
    },
}

_TOP_K = 5
# Same retrieval floor / confidence line as the legacy buddy: `retrieve` drops
# anything below it, so an empty result means nothing indexed answers with confidence.
_MIN_SCORE = 0.3

GREP = "grep"

# Searches asked for in the same step run together, as in the chat agent. Low
# because each `retrieve` already forks a pair of threads (`rag.hybrid`).
_MAX_PARALLEL_TOOLS = 4

# The persona's no-evidence path reacts to this wording; keep it.
_NO_MATCH = "No indexed material matched this search."

# Answered instead of the model when the reader narrowed the search and nothing
# under the narrowing matched -- the same notice chat gave, word for word, so a
# reader moving from chat to the buddy meets the same honesty.
NO_FILTERED_RESULTS_MESSAGE = (
    "I could not find any matching sources for the selected filters, "
    "so I cannot answer this reliably."
)

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


@dataclass
class AgentTurnResult:
    """The outcome of one agent turn.

    ``final`` distinguishes "here is the answer" (``text`` is it) from "I need the
    backend to run these tools first" (``pending_tool_calls`` are them). ``messages``
    is always the full running conversation the caller must carry back verbatim next
    turn -- it already includes any search steps run here and the tool-use turn the
    pending calls belong to.
    """

    final: bool
    text: str
    messages: list[Message]
    pending_tool_calls: list[ToolCall] = field(default_factory=list[ToolCall])
    citations: list[Citation] = field(default_factory=list[Citation])
    #: The model's reasoning, one entry per call that returned any. Display-only:
    #: it is never written into ``messages`` as text the caller carries back.
    reasoning: list[str] = field(default_factory=list[str])


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
    # Kept for the next internal hop, as the chat agent does: a reasoning provider
    # continues a tool-using thought from these. They never reach the caller --
    # the wire message has no field for them (``api.routes.buddy._from_message``).
    if result.reasoning:
        msg["reasoning"] = result.reasoning
    if result.reasoning_details:
        msg["reasoning_details"] = result.reasoning_details
    return msg


def _tool_result_message(call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id)


def drop_test_material(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Remove test, fixture and sample files from what the mentor may quote.

    Dropped rather than labelled, and the difference matters. A label asks the
    model to hold a distinction while it answers, and this is the exact question
    a model is worst at holding it on: a fixture describing a review process
    *reads* like a review process. Told "this is sample data", it produced a
    branch-naming convention, a CI pipeline and a merge workflow that no team had
    ever used — and defended them when challenged. What a model cannot quote, it
    cannot attribute to the team.

    Emptying the result is a *good* outcome, not a degraded one. The buddy's
    no-evidence path drafts the question to ask a colleague instead of answering,
    which is exactly right here: if the only thing matching was a fixture, the
    thing is not documented, and saying so beats reciting an example as policy.

    The label below stays as the second line, for material that is genuinely
    primary but talks about tests.
    """
    return [
        chunk
        for chunk in chunks
        if not is_test_chunk(chunk.filename, chunk.source_url, chunk.source_role)
    ]


def _chunk_header(chunk: ScoredChunk) -> str:
    # Anything reaching here already survived `drop_test_material`, so this only
    # fires for a file no signal marked -- it is the belt to that braces.
    header = chunk_header(chunk)
    if is_test_chunk(chunk.filename, chunk.source_url, chunk.source_role):
        return (
            f"{header} (test/fixture file -- example or sample data, "
            "not the team's real documentation or process)"
        )
    return header


def _without_test_material(result: ToolResult) -> ToolResult:
    """``result`` with test material dropped from its chunks *and* its matches.

    Both, so a dropped fixture is neither quoted nor counted as an omitted match
    ("3 further matches omitted") the model would go looking for.
    """
    kept = drop_test_material(result.chunks)
    if len(kept) == len(result.chunks):
        return result
    match_ids = result.match_chunk_ids
    if match_ids is not None:
        match_ids = frozenset(chunk.id for chunk in kept if chunk.id in match_ids)
    return ToolResult(summary=result.summary, chunks=kept, match_chunk_ids=match_ids)


def _format_result(result: ToolResult, chunks: list[ScoredChunk]) -> str:
    return format_evidence(result, chunks, header=_chunk_header, empty=_NO_MATCH)


class _SearchDocsTool(RetrieveTool):
    """Chat's ``retrieve`` under the name the buddy's persona and backend know.

    Reused rather than re-implemented so the buddy gets what chat had: each hit
    arrives with its neighbouring chunks, and only the hits are cited.
    """

    name = SEARCH_DOCS
    description = _SEARCH_TOOL["description"]

    def tool_spec(self) -> ToolSpec:
        return _SEARCH_TOOL


def _local_tools(
    llm: LLMClient,
    store: VectorStore,
    exclusions: SourceExclusions,
    filters: RetrievalFilters,
    with_grep: bool,
) -> ToolRegistry:
    search = _SearchDocsTool(
        llm,
        store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        exclusions=exclusions,
        filters=filters,
    )
    if not with_grep:
        return ToolRegistry([search])
    return ToolRegistry(
        [search, GrepTool(store, exclusions=exclusions, filters=filters)]
    )


def _run_local(registry: ToolRegistry, calls: Sequence[ToolCall]) -> list[ToolResult]:
    """Runs a step's local searches, together when there are several.

    Results come back in call order (``map``), so the conversation never depends
    on which search finished first. Safe for the same reasons as the chat agent's
    ``_run_tools``: every tool here only reads.
    """

    def run_one(call: ToolCall) -> ToolResult:
        return registry.execute(call.name, call.arguments)

    if len(calls) == 1:
        return [run_one(calls[0])]
    workers = min(len(calls), _MAX_PARALLEL_TOOLS)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(run_one, calls))


def _narrows(filters: RetrievalFilters | None) -> bool:
    """Whether the reader narrowed the search beyond the always-present scope."""
    if filters is None:
        return False
    return bool(filters.source_systems) or bool(filters.time_from or filters.time_to)


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
    """Runs one agent turn: executes the searches locally, pauses on backend tools.

    Loops internally while the model only asks for local searches (``search_docs``,
    and ``grep`` when capabilities are off); returns as soon as it either produces a
    final answer or requests a tool only the backend can run. A step budget bounds
    the internal loop; if it's exhausted the model is asked once more with no tools,
    forcing an answer.

    ``prior_summary`` stands in for everything older than ``messages``.

    ``capabilities_enabled`` and ``team_mode`` choose the persona and must be passed
    on every hop, because the persona is rebuilt on every hop: a resume that lost
    either would answer the rest of the turn as the default mentor.

    ``filters`` narrows every search by source system and time; its project fields
    are ignored, because the scope is ``project_ids``. When it narrows, this hop
    searched, nothing it found survived, and no backend tool is mounted to have
    supplied other evidence, the answer is ``NO_FILTERED_RESULTS_MESSAGE`` rather
    than whatever the model composed from nothing.

    This turn does not fold anything, and used not to be able to say that. A
    ``summarize_upto`` argument asked it to compact the oldest window messages
    *before* the model began composing a reply, and because the caller's cursor
    advanced by exactly what was folded, the window sat at its cap forever once it
    first filled -- so every turn of a long visit paid an extra serialized model call,
    ahead of the answer, to compress a single exchange. Folding is
    ``POST /onboarding/buddy/compact``, which the backend runs afterwards.
    """
    window = fence_user_messages(list(messages))
    summary = prior_summary
    with_grep = not capabilities_enabled
    scope = replace(
        filters if filters is not None else RetrievalFilters(),
        project_id=None,
        # Scoped to the projects this hire is on, so the mentor cannot quote
        # another team's material as this team's -- and cannot hide the hire's
        # own second project either. Fail-closed, like every project scope:
        # material belonging to no project matches none (`rag.filters`).
        project_ids=project_ids,
    )
    registry = _local_tools(
        llm,
        store,
        exclusions if exclusions is not None else SourceExclusions(),
        scope,
        with_grep,
    )
    local_names = registry.names()

    backend = [t for t in backend_tools if t["name"] not in local_names]
    backend_names = {tool["name"] for tool in backend}
    tools = [*registry.specs(), *backend]
    # The persona describes exactly the tools this hire was mounted, never a fixed
    # catalogue: the backend decides what a given role can even have, and a mentor
    # told about a tool it does not have will offer the hire something impossible.
    work = _ensure_persona(
        window,
        summary,
        {*local_names, *backend_names},
        vocabulary,
        capabilities_enabled,
        team_mode,
    )
    citations: list[Citation] = []
    seen_chunk_ids: set[str] = set()
    reasoning: list[str] = []
    searched = False
    found = False

    def _answer(text: str) -> AgentTurnResult:
        # Nothing matched under an explicit narrowing: the reader chose the filter,
        # so an answer composed from no sources is the one thing not to give.
        if _narrows(filters) and searched and not found and not backend_names:
            text = NO_FILTERED_RESULTS_MESSAGE
        return AgentTurnResult(
            final=True,
            text=text,
            messages=[*work, Message(role="assistant", content=text)],
            citations=citations,
            reasoning=reasoning,
        )

    for _ in range(_MAX_STEPS):
        result = llm.chat(work, tools)
        if result.reasoning and result.reasoning.strip():
            reasoning.append(result.reasoning)

        if not result.tool_calls:
            return _answer(result.text)
        work = [*work, _assistant_message(result)]

        local_calls = [c for c in result.tool_calls if c.name in local_names]
        outcomes = dict(
            zip(
                (c.id for c in local_calls),
                _run_local(registry, local_calls) if local_calls else [],
                strict=True,
            )
        )
        searched = searched or bool(local_calls)

        pending: list[ToolCall] = []
        for call in result.tool_calls:
            outcome = outcomes.get(call.id)
            if outcome is not None:
                screened = _without_test_material(outcome)
                chunks = limit_evidence(screened.chunks)
                # Cited after the drop and the budget, and only the direct matches:
                # nothing the model was not shown, and no context-only neighbour,
                # is offered to the hire as a source. Deduped by chunk id.
                matched = screened.matched_chunks(chunks)
                found = found or bool(matched)
                fresh = [c for c in matched if c.id not in seen_chunk_ids]
                seen_chunk_ids.update(c.id for c in fresh)
                citations.extend(build_citations(fresh))
                content = _format_result(screened, chunks)
                work = [*work, _tool_result_message(call.id, content)]
            elif call.name in backend_names:
                pending.append(call)
            else:
                work = [
                    *work,
                    _tool_result_message(call.id, f"Unknown tool: {call.name}."),
                ]

        # A tool only the backend can run: stop and hand it back. Any local searches
        # in this same turn already have their results appended above, so the message
        # list stays well-formed.
        if pending:
            return AgentTurnResult(
                final=False,
                text=result.text,
                messages=work,
                pending_tool_calls=pending,
                citations=citations,
                reasoning=reasoning,
            )
        # Only local searches this turn -- loop and let the model reason over them.

    # Step budget spent: force a final answer with no tools rather than loop forever.
    #
    # The notice goes to the model, not into the returned conversation: a final turn's
    # messages are discarded by the caller, and a transcript carrying instructions about
    # a budget nobody can see would be a strange thing to keep. `generate` returns
    # text only, so this call adds nothing to ``reasoning``.
    return _answer(
        llm.generate([*work, Message(role="system", content=_NO_TOOLS_THIS_TURN)])
    )


__all__ = [
    "AgentTurnResult",
    "GREP",
    "LLMUnavailableError",
    "NO_FILTERED_RESULTS_MESSAGE",
    "run_agent_turn",
    "SEARCH_DOCS",
]
