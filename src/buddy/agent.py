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

from collections.abc import Iterator

from buddy.events import (
    AgentStreamEvent,
    AgentTurnResult,
)
from buddy.turn import Request, Turn
from buddy.vocabulary import DEFAULT_VOCABULARY, Vocabulary
from corpus.source_filter import SourceExclusions
from corpus.types import RetrievalFilters
from llm.base import (
    LLMClient,
    Message,
    ToolSpec,
)
from llm.errors import LLMUnavailableError
from store.base import VectorStore


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
    request = Request(
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
    yield from Turn(request).run()


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
