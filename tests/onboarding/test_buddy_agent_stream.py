"""The buddy turn as a stream: what a reader sees while the loop is still running."""

from collections.abc import Iterator, Sequence

import pytest

from llm.base import (
    ChatResult,
    LLMChatStreamEvent,
    LLMStreamEvent,
    Message,
    ReasoningDelta,
    TextDelta,
    ToolCall,
    ToolSpec,
)
from llm.errors import LLMUnavailableError
from onboarding.buddy_agent import (
    GREP,
    AgentReasoning,
    AgentStreamEvent,
    AgentToken,
    AgentToolUse,
    AgentTurnResult,
    run_agent_turn,
    stream_agent_turn,
)
from rag.filters import NO_FILTERED_RESULTS_MESSAGE
from rag.types import Chunk, RetrievalFilters
from tests.stubs.llm import ScriptedLLMClient
from tests.stubs.store import StubVectorStore

_EMBEDDING = [1.0] + [0.0] * 767
_METRICS: ToolSpec = {
    "name": "get_my_metrics",
    "description": "The hire's onboarding metrics.",
    "parameters": {"type": "object", "properties": {}},
}
_JIRA_ONLY = RetrievalFilters(source_systems=["JIRA"])

type Hop = Sequence[LLMChatStreamEvent]


def _user(text: str) -> Message:
    return Message(role="user", content=text)


def _store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="auth.md",
                text="the login handler lives in auth.py",
                embedding=_EMBEDDING,
                source_system="GITHUB",
            )
        ]
    )
    return store


def _search(query: str = "login") -> ChatResult:
    return ChatResult(
        text="",
        tool_calls=[ToolCall(id="call_0", name=GREP, arguments={"patterns": [query]})],
    )


def _narrated_search(text: str, query: str = "login") -> Hop:
    """A hop that says something, then asks for a search."""
    call = ToolCall(id="call_0", name=GREP, arguments={"patterns": [query]})
    return [TextDelta(text), ChatResult(text=text, tool_calls=[call])]


def _answer(*fragments: str) -> Hop:
    return [*(TextDelta(f) for f in fragments), ChatResult(text="".join(fragments))]


class _Hops(ScriptedLLMClient):
    """Plays one scripted event sequence per model call."""

    def __init__(self, hops: Sequence[Hop], forced: Sequence[LLMStreamEvent] = ()):
        super().__init__(turns=[])
        self._hops = list(hops)
        self._forced = list(forced)

    def chat_stream(
        self, messages: list[Message], tools: list[ToolSpec] | None = None
    ) -> Iterator[LLMChatStreamEvent]:
        self.chat_calls.append(messages)
        yield from self._hops.pop(0)

    def stream(self, messages: list[Message]) -> Iterator[LLMStreamEvent]:
        self.stream_calls.append(messages)
        yield from self._forced


def _events(
    llm: ScriptedLLMClient,
    *,
    backend_tools: list[ToolSpec] | None = None,
    filters: RetrievalFilters | None = None,
) -> list[AgentStreamEvent]:
    return list(
        stream_agent_turn(
            [_user("where is login?")],
            backend_tools or [],
            llm,
            _store(),
            capabilities_enabled=False,
            filters=filters,
        )
    )


def _text(events: list[AgentStreamEvent]) -> str:
    return "".join(e.text for e in events if isinstance(e, AgentToken))


def _outcome(events: list[AgentStreamEvent]) -> AgentTurnResult:
    last = events[-1]
    assert isinstance(last, AgentTurnResult)
    return last


def test_reasoning_and_tokens_arrive_in_order_and_the_result_comes_last() -> None:
    llm = _Hops(
        [
            [
                ReasoningDelta("Let me "),
                ReasoningDelta("think."),
                TextDelta("In "),
                TextDelta("auth.py."),
                ChatResult(text="In auth.py.", reasoning="Let me think."),
            ]
        ]
    )

    events = _events(llm)

    assert events[:-1] == [
        AgentReasoning("Let me "),
        AgentReasoning("think."),
        AgentToken("In "),
        AgentToken("auth.py."),
    ]
    outcome = _outcome(events)
    assert outcome.final is True
    assert outcome.text == "In auth.py."
    assert outcome.reasoning == ["Let me think."]


def test_fragments_are_yielded_before_the_model_has_finished() -> None:
    state = {"finished": False}

    class _Slow(ScriptedLLMClient):
        def chat_stream(
            self, messages: list[Message], tools: list[ToolSpec] | None = None
        ) -> Iterator[LLMChatStreamEvent]:
            yield TextDelta("first")
            state["finished"] = True
            yield ChatResult(text="first")

    turn = stream_agent_turn([_user("hi")], [], _Slow(turns=[]), _store())

    assert next(turn) == AgentToken("first")
    assert state["finished"] is False


def test_each_local_search_is_announced_before_it_runs() -> None:
    llm = _Hops([[_search("login")], _answer("ok")])

    events = _events(llm)

    uses = [e for e in events if isinstance(e, AgentToolUse)]
    assert uses == [AgentToolUse(GREP, {"patterns": ["login"]})]
    assert events.index(uses[0]) < events.index(_outcome(events))


def test_a_backend_tool_is_not_announced_as_a_search() -> None:
    call = ToolCall(id="m1", name="get_my_metrics", arguments={})
    llm = _Hops([[ChatResult(text="", tool_calls=[call])]])

    events = _events(llm, backend_tools=[_METRICS])

    assert not any(isinstance(e, AgentToolUse) for e in events)
    outcome = _outcome(events)
    assert outcome.final is False
    assert [c.name for c in outcome.pending_tool_calls] == ["get_my_metrics"]


def test_the_text_of_one_hop_is_set_apart_from_the_next() -> None:
    llm = _Hops([_narrated_search("Let me check."), _answer("It is in auth.py.")])

    events = _events(llm)

    assert _text(events) == "Let me check.\n\nIt is in auth.py."
    # Only the answer is the stored text; the narration was streamed, not kept.
    assert _outcome(events).text == "It is in auth.py."


def test_a_hop_that_wrote_nothing_adds_no_separator() -> None:
    llm = _Hops([[_search()], _answer("It is in auth.py.")])

    assert _text(_events(llm)) == "It is in auth.py."


def test_the_separator_survives_a_silent_hop_in_between() -> None:
    llm = _Hops(
        [_narrated_search("Checking."), [_search("again")], _answer("Found it.")]
    )

    assert _text(_events(llm)) == "Checking.\n\nFound it."


def test_a_filtered_answer_with_no_sources_is_replaced_not_streamed() -> None:
    llm = _Hops([[_search()], _answer("an ", "ungrounded ", "guess")])

    events = _events(llm, filters=_JIRA_ONLY)

    assert _text(events) == NO_FILTERED_RESULTS_MESSAGE
    assert _outcome(events).text == NO_FILTERED_RESULTS_MESSAGE


def test_held_text_is_released_when_the_hop_goes_on_to_search() -> None:
    llm = _Hops(
        [
            [_search()],
            _narrated_search("Still looking.", "other"),
            _answer("guess"),
        ]
    )

    events = _events(llm, filters=_JIRA_ONLY)

    assert _text(events) == f"Still looking.\n\n{NO_FILTERED_RESULTS_MESSAGE}"


def test_a_filtered_search_that_found_something_streams_normally() -> None:
    llm = _Hops([[_search()], _answer("In ", "auth.py.")])

    events = _events(llm, filters=RetrievalFilters(source_systems=["GITHUB"]))

    assert _text(events) == "In auth.py."
    assert _outcome(events).text == "In auth.py."


def test_a_spent_search_budget_streams_the_forced_answer() -> None:
    llm = _Hops(
        [[_search()]] * 6,
        forced=[
            ReasoningDelta("wrapping up"),
            TextDelta("Here is "),
            TextDelta("all."),
        ],
    )

    events = _events(llm)

    assert len(llm.stream_calls) == 1
    assert AgentReasoning("wrapping up") in events
    assert _text(events) == "Here is all."
    outcome = _outcome(events)
    assert outcome.final is True
    assert outcome.text == "Here is all."
    assert outcome.reasoning == ["wrapping up"]


def test_a_forced_answer_under_a_filter_with_no_sources_is_replaced() -> None:
    llm = _Hops([[_search()]] * 6, forced=[TextDelta("guess")])

    events = _events(llm, filters=_JIRA_ONLY)

    assert _text(events) == NO_FILTERED_RESULTS_MESSAGE


def test_abandoning_the_stream_stops_the_loop() -> None:
    """The caller hanging up mid-turn must not leave the model running more hops."""
    llm = _Hops([_narrated_search("Let me check."), _answer("never reached")])
    turn = stream_agent_turn(
        [_user("hi")], [], llm, _store(), capabilities_enabled=False
    )

    assert next(turn) == AgentToken("Let me check.")
    turn.close()

    assert len(llm.chat_calls) == 1


def test_a_model_that_ends_without_a_result_is_an_unavailable_model() -> None:
    llm = _Hops([[TextDelta("half an ans")]])

    with pytest.raises(LLMUnavailableError):
        _events(llm)


def test_the_buffered_turn_is_the_streamed_turns_outcome() -> None:
    def make() -> _Hops:
        return _Hops([_narrated_search("Checking."), _answer("In auth.py.")])

    streamed = _outcome(_events(make()))
    buffered = run_agent_turn(
        [_user("where is login?")],
        [],
        make(),
        _store(),
        capabilities_enabled=False,
    )

    assert buffered == streamed
    assert buffered.messages[-1] == Message(role="assistant", content="In auth.py.")
