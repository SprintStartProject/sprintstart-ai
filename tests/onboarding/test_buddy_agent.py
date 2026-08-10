import pytest

from llm.base import Message, ToolSpec
from llm.errors import LLMUnavailableError
from onboarding.buddy_agent import (
    SEARCH_DOCS,
    _format_chunks,
    drop_test_material,
    run_agent_turn,
)
from rag.types import ScoredChunk
from tests.stubs.llm import ScriptedLLMClient
from tests.stubs.store import StubVectorStore

_GET_MY_METRICS: ToolSpec = {
    "name": "get_my_metrics",
    "description": "The hire's onboarding metrics.",
    "parameters": {"type": "object", "properties": {}},
}


def _tool(name: str) -> ToolSpec:
    """A minimal backend tool spec; only its name reaches the persona."""
    return {
        "name": name,
        "description": f"The {name} tool.",
        "parameters": {"type": "object", "properties": {}},
    }


def _user(text: str) -> Message:
    return Message(role="user", content=text)


def _system_of(messages: list[Message]) -> str:
    assert messages[0]["role"] == "system"
    return messages[0].get("content") or ""


class _UnavailableSummarizer(ScriptedLLMClient):
    """Chats fine but cannot summarize -- the degrade path for compaction."""

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        raise LLMUnavailableError("model down")


def test_pauses_and_hands_back_a_backend_tool_call() -> None:
    llm = ScriptedLLMClient(turns=[[("get_my_metrics", {})]])
    store = StubVectorStore()

    result = run_agent_turn([_user("is my PR stuck?")], [_GET_MY_METRICS], llm, store)

    assert result.final is False
    assert [call.name for call in result.pending_tool_calls] == ["get_my_metrics"]
    # The assistant turn carrying the tool call is in the running messages, so the
    # backend can append a matching tool result and resume without it going malformed.
    assert any(
        msg["role"] == "assistant" and msg.get("tool_calls") for msg in result.messages
    )


def test_resumes_with_a_tool_result_and_answers() -> None:
    llm = ScriptedLLMClient(turns=[[("get_my_metrics", {})]])
    store = StubVectorStore()
    first = run_agent_turn([_user("is my PR stuck?")], [_GET_MY_METRICS], llm, store)
    call_id = first.pending_tool_calls[0].id

    # The backend appends the tool result and re-invokes; the scripted model now has
    # no more turns, so it produces its final answer.
    resumed_messages = [
        *first.messages,
        Message(
            role="tool",
            content="openPullRequestCount=1, longestOpenWaitHours=52, stalled=true",
            tool_call_id=call_id,
        ),
    ]
    llm2 = ScriptedLLMClient(
        turns=[], answer="Your PR has waited 52 hours — that's on the reviewer."
    )

    result = run_agent_turn(resumed_messages, [_GET_MY_METRICS], llm2, store)

    assert result.final is True
    assert "52 hours" in result.text


def _chunk(
    filename: str, source_url: str | None = None, source_role: str = "primary"
) -> ScoredChunk:
    return ScoredChunk(
        id="c1",
        artifact_id="a1",
        filename=filename,
        text="Some content.",
        score=0.9,
        source_url=source_url,
        source_role=source_role,  # type: ignore[arg-type]
    )


# Retrieval returns a *basename*, never a path -- the earlier tests here passed a
# path as `filename` and so never reproduced what the buddy actually sees. These
# use the real shape.
_FIXTURE_URL = "https://github.com/o/r/blob/7766d14/tests/rag/demo-corpus/process.md"
_REAL_DOC_URL = "https://github.com/o/r/blob/main/docs/process.md"


def test_a_fixture_never_reaches_the_model_at_all() -> None:
    # The failure this fixes: a demo-corpus file quoted as the team's real process,
    # complete with a branch convention and a CI pipeline nobody had ever used.
    kept = drop_test_material([_chunk("process.md", _FIXTURE_URL)])

    assert kept == []


def test_a_real_document_with_the_same_basename_is_kept() -> None:
    kept = drop_test_material([_chunk("process.md", _REAL_DOC_URL)])

    assert len(kept) == 1


def test_a_search_finding_only_fixtures_finds_nothing() -> None:
    # And that is the right answer: the buddy's no-evidence path drafts the question
    # to ask a colleague, which beats reciting an example as policy.
    kept = drop_test_material([_chunk("process.md", _FIXTURE_URL), _chunk("test_x.py")])

    assert _format_chunks(kept) == "No indexed material matched this search."


def test_ingest_time_role_is_honoured_even_with_no_url() -> None:
    kept = drop_test_material([_chunk("sample.json", None, "test")])

    assert kept == []


def test_still_marks_anything_that_slips_past_the_drop() -> None:
    # Belt to the braces: `_format_chunks` is reachable with un-dropped input.
    formatted = _format_chunks([_chunk("process.md", _FIXTURE_URL)])

    assert "test/fixture file" in formatted


def test_does_not_mark_a_real_source_chunk() -> None:
    formatted = _format_chunks([_chunk("process.md", _REAL_DOC_URL)])

    assert "test/fixture file" not in formatted


def test_runs_search_docs_locally_and_collects_citations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = ScoredChunk(
        id="c1",
        artifact_id="a1",
        filename="README.md",
        text="Run ./gradlew build.",
        score=0.9,
    )

    def _fake_retrieve(*args: object, **kwargs: object) -> list[ScoredChunk]:
        return [chunk]

    monkeypatch.setattr("onboarding.buddy_agent.retrieve", _fake_retrieve)
    llm = ScriptedLLMClient(
        turns=[[(SEARCH_DOCS, {"query": "how to build"})], []],
        answer="Run ./gradlew build.",
    )

    result = run_agent_turn(
        [_user("how do I build?")], [_GET_MY_METRICS], llm, StubVectorStore()
    )

    assert result.final is True
    # search_docs is executed here, not handed back, and its chunk becomes a citation.
    assert result.pending_tool_calls == []
    assert [cit.artifact_id for cit in result.citations] == ["a1"]


def test_search_is_scoped_to_the_project_the_hire_is_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The wiring that stops project scoping being a field nobody sets.

    Without it the mentor searches every project indexed and can quote another
    team's process as this team's.
    """
    seen: list[object] = []

    def _fake_retrieve(*args: object, **kwargs: object) -> list[ScoredChunk]:
        seen.append(kwargs.get("filters"))
        return []

    monkeypatch.setattr("onboarding.buddy_agent.retrieve", _fake_retrieve)
    llm = ScriptedLLMClient(turns=[[(SEARCH_DOCS, {"query": "how do we deploy"})], []])

    run_agent_turn(
        [_user("how do we deploy?")],
        [],
        llm,
        StubVectorStore(),
        project_ids=frozenset({"project-a"}),
    )

    assert [getattr(f, "project_ids", None) for f in seen] == [frozenset({"project-a"})]


def test_a_deployment_serving_one_project_scopes_to_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[object] = []

    def _fake_retrieve(*args: object, **kwargs: object) -> list[ScoredChunk]:
        seen.append(kwargs.get("filters"))
        return []

    monkeypatch.setattr("onboarding.buddy_agent.retrieve", _fake_retrieve)
    llm = ScriptedLLMClient(turns=[[(SEARCH_DOCS, {"query": "how do we deploy"})], []])

    run_agent_turn([_user("how?")], [], llm, StubVectorStore())

    # No project passed means no narrowing, so a single-project deployment is
    # unaffected by any of this.
    assert [getattr(f, "project_ids", None) for f in seen] == [None]


def test_unknown_tool_is_answered_as_such_and_does_not_stall() -> None:
    llm = ScriptedLLMClient(turns=[[("does_not_exist", {})], []], answer="done")
    result = run_agent_turn([_user("hi")], [_GET_MY_METRICS], llm, StubVectorStore())

    assert result.final is True
    assert any(
        msg["role"] == "tool" and "Unknown tool" in (msg.get("content") or "")
        for msg in result.messages
    )


def test_persona_makes_the_buddy_a_plan_aware_mentor() -> None:
    llm = ScriptedLLMClient(turns=[], answer="hi")

    # The persona is assembled from the tools this hire was actually mounted, so a
    # test about the plan-aware directives has to mount the plan tools.
    plan_tools = [
        _GET_MY_METRICS,
        _tool("get_learning_plan"),
        _tool("get_module"),
        _tool("submit_verification"),
        _tool("claim_goal"),
    ]

    run_agent_turn([_user("hello")], plan_tools, llm, StubVectorStore())

    persona = _system_of(llm.chat_calls[0])
    # The load-bearing directives, pinned without over-fitting the wording: plan
    # before recommending, teach from modules, verify through the action, no
    # invented order, no scores.
    assert "get_learning_plan" in persona
    assert "get_module" in persona
    assert "submit_verification" in persona
    assert "claim_goal" in persona
    assert "never invent" in persona
    assert "never mention scores" in persona


def test_persona_omits_tools_this_hire_was_not_mounted() -> None:
    llm = ScriptedLLMClient(turns=[], answer="hi")

    run_agent_turn([_user("hello")], [_GET_MY_METRICS], llm, StubVectorStore())

    persona = _system_of(llm.chat_calls[0])
    # A mentor told about a tool it was not given will offer the hire something
    # impossible -- which is how a Scrum Master's buddy ended up discussing their
    # pull requests.
    assert "get_learning_plan" not in persona
    assert "get_module" not in persona
    assert "`get_my_metrics`" in persona


def test_summary_round_trips_inside_the_system_message_on_resume() -> None:
    llm = ScriptedLLMClient(turns=[], answer="answer")
    first = run_agent_turn(
        [_user("m1"), _user("m2")],
        [_GET_MY_METRICS],
        llm,
        StubVectorStore(),
        prior_summary="Summary of old turns.",
    )

    # A resume hop carries the returned conversation verbatim -- no summary field --
    # and must not get a second persona prepended nor lose the folded memory.
    llm2 = ScriptedLLMClient(turns=[], answer="answer")
    run_agent_turn(first.messages, [_GET_MY_METRICS], llm2, StubVectorStore())

    system_messages = [m for m in llm2.chat_calls[0] if m["role"] == "system"]
    assert len(system_messages) == 1
    assert "Summary of old turns." in (system_messages[0].get("content") or "")


def test_prior_summary_is_standing_context() -> None:
    llm = ScriptedLLMClient(turns=[], answer="answer")

    run_agent_turn(
        [_user("recent")],
        [_GET_MY_METRICS],
        llm,
        StubVectorStore(),
        prior_summary="The hire merged their first PR.",
    )

    assert "The hire merged their first PR." in _system_of(llm.chat_calls[0])


def test_the_turn_never_folds_however_long_the_window_is() -> None:
    """⚠️ Folding used to run *before* the answer, and this pins that it is gone.

    Because the caller's cursor advanced by exactly what was folded, the window sat at
    its cap forever once it first filled -- so every turn of a long visit paid an extra
    serialized model call to compress one exchange, ahead of the reply the hire was
    waiting for. `onboarding.buddy_compact` does it afterwards.

    The model here refuses to `generate` at all, which is what compaction used: a turn
    that still folded would raise instead of answering.
    """
    llm = _UnavailableSummarizer(turns=[], answer="answer anyway")
    history = [_user(f"m{i}") for i in range(1, 40)]

    result = run_agent_turn(history, [_GET_MY_METRICS], llm, StubVectorStore())

    # Nothing is dropped, so nothing needed summarizing: the whole window is sent.
    contents = [msg.get("content") for msg in result.messages]
    assert all(f"m{i}" in contents for i in range(1, 40))
    assert result.final is True
