import json

import pytest

from corpus.ids import ProjectKey
from corpus.types import RetrievalFilters, ScoredChunk
from insights import interview
from insights.interview import InterviewLimits, InterviewRequest, plan_interview
from insights.interview_components import RiskFile
from insights.interview_prompts import Coverage, QuestionType
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from store.base import VectorStore
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


class _SequenceLLM(StubLLMClient):
    """Answers each ``generate`` call with the next scripted text."""

    def __init__(self, *answers: str) -> None:
        super().__init__()
        self.answers = list(answers)
        self.prompts: list[list[Message]] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.prompts.append(messages)
        return self.answers.pop(0)


class _DownLLM(StubLLMClient):
    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        raise LLMUnavailableError("down")


_FILES = [
    RiskFile("github:acme/app", "src/auth/Login.kt", "ORPHANED", 0.9),
    RiskFile("github:acme/app", "docs/setup.md", "HOARDED", 0.4, kind="PROSE"),
]


@pytest.fixture
def searches(monkeypatch: pytest.MonkeyPatch) -> list[RetrievalFilters | None]:
    seen: list[RetrievalFilters | None] = []

    def fake_retrieve(
        question: str,
        llm: LLMClient,
        store: VectorStore,
        top_k: int = 5,
        filters: RetrievalFilters | None = None,
    ) -> list[ScoredChunk]:
        seen.append(filters)
        return [ScoredChunk("c1", "a1", "auth.md", "Login retries thrice.", 0.9)]

    monkeypatch.setattr(interview, "retrieve", fake_retrieve)
    return seen


def _questions(*texts: str) -> str:
    entries = [{"text": t, "type": "gotcha", "triggers": ["orphaned"]} for t in texts]
    return json.dumps({"questions": entries})


def _verdicts(*coverages: str) -> str:
    entries = [
        {"index": i, "coverage": c, "answer": "From the docs.", "sources": ["auth.md"]}
        for i, c in enumerate(coverages)
    ]
    return json.dumps({"verdicts": entries})


def _request(limits: InterviewLimits | None = None) -> InterviewRequest:
    return InterviewRequest(
        project=ProjectKey("p1"), files=_FILES, limits=limits or InterviewLimits()
    )


@pytest.mark.usefixtures("searches")
def test_questions_the_records_answer_are_reported_apart() -> None:
    llm = _SequenceLLM(
        _questions("Why retry?", "Who rotates the key?", "What breaks on deploy?"),
        _verdicts("answered", "partial", "open"),
        _questions("Is the setup page current?"),
        _verdicts("open"),
    )

    plan = plan_interview(llm, StubVectorStore(), _request())

    assert [q.text for q in plan.questions] == [
        "What breaks on deploy?",
        "Who rotates the key?",
        "Is the setup page current?",
    ]
    assert [q.text for q in plan.answered_by_records] == ["Why retry?"]
    answered = plan.answered_by_records[0]
    assert answered.record_answer == "From the docs."
    assert answered.sources == ("auth.md",)
    assert answered.type == QuestionType.GOTCHA


@pytest.mark.usefixtures("searches")
def test_priority_follows_component_risk_finding_and_what_is_left_open() -> None:
    llm = _SequenceLLM(
        _questions("Open?", "Partial?"),
        _verdicts("open", "partial"),
        json.dumps({"questions": [{"text": "Doc?", "type": "gotcha"}]}),
        _verdicts("open"),
    )

    plan = plan_interview(llm, StubVectorStore(), _request())
    by_text = {q.text: q for q in plan.questions}

    assert by_text["Open?"].priority == 0.9
    assert by_text["Partial?"].priority == 0.45
    assert by_text["Partial?"].coverage == Coverage.PARTIAL
    assert by_text["Doc?"].priority == 0.24
    assert by_text["Doc?"].component == "github:acme/app/docs"


def test_every_search_stays_inside_the_project(
    searches: list[RetrievalFilters | None],
) -> None:
    llm = _SequenceLLM(
        _questions("A?"), _verdicts("open"), _questions("B?"), _verdicts("open")
    )

    plan_interview(llm, StubVectorStore(), _request())

    assert searches
    assert all(f is not None and f.project_id == "p1" for f in searches)


@pytest.mark.usefixtures("searches")
def test_the_model_sees_the_records_found_for_the_component() -> None:
    llm = _SequenceLLM(
        _questions("A?"), _verdicts("open"), _questions("B?"), _verdicts("open")
    )

    plan_interview(llm, StubVectorStore(), _request())

    assert "Login retries thrice." in llm.prompts[0][1]["content"]
    assert "Question 0: A?" in llm.prompts[1][1]["content"]


@pytest.mark.usefixtures("searches")
def test_a_malformed_generation_costs_only_that_component() -> None:
    llm = _SequenceLLM("not json", _questions("B?"), _verdicts("open"))

    plan = plan_interview(llm, StubVectorStore(), _request())

    assert [q.text for q in plan.questions] == ["B?"]


@pytest.mark.usefixtures("searches")
def test_a_malformed_judgement_keeps_the_questions_open() -> None:
    llm = _SequenceLLM(
        _questions("A?"), "garbage", _questions("B?"), _verdicts("answered")
    )

    plan = plan_interview(llm, StubVectorStore(), _request())

    assert [(q.text, q.coverage) for q in plan.questions] == [("A?", Coverage.OPEN)]


@pytest.mark.usefixtures("searches")
def test_only_the_riskiest_components_are_planned() -> None:
    llm = _SequenceLLM(_questions("A?"), _verdicts("open"))

    plan = plan_interview(
        llm, StubVectorStore(), _request(InterviewLimits(max_components=1))
    )

    assert {q.component for q in plan.questions} == {"github:acme/app/src/auth"}


@pytest.mark.usefixtures("searches")
def test_an_unavailable_model_is_not_hidden() -> None:
    with pytest.raises(LLMUnavailableError):
        plan_interview(_DownLLM(), StubVectorStore(), _request())
