"""Plans an exit interview from an offboarding risk map.

The backend decides deterministically which files a leaver's departure puts at risk and
why. This module turns that map into the questions worth the leaver's time:

1. group the files into components (:mod:`insights.interview_components`);
2. per component, read excerpts of the project's records and let the model write
   questions that answer the component's findings;
3. per question, search the records again and let a judge decide whether they already
   answer it;
4. keep the questions the records leave open or half answered, most urgent first, and
   report the ones the records answer separately, with the answer and its sources, so a
   manager can see why they were not asked.

A model failure is not hidden: :class:`llm.errors.LLMUnavailableError` propagates. A
malformed model answer only costs that component its questions (generation) or keeps its
questions open (judging), because asking one question too many is cheaper than losing
knowledge.
"""

import logging
from dataclasses import dataclass, field

from corpus.ids import ProjectKey
from corpus.types import RetrievalFilters
from insights.interview_components import (
    TRIGGER_WEIGHT,
    Component,
    RiskFile,
    Trigger,
    components_of,
)
from insights.interview_prompts import (
    Coverage,
    DraftQuestion,
    Excerpt,
    QuestionType,
    Verdict,
    generate_prompt,
    judge_prompt,
    parse_questions,
    parse_verdicts,
)
from llm.base import LLMClient
from rag.retriever import retrieve
from store.base import VectorStore

logger = logging.getLogger(__name__)

_CONTEXT_TOP_K = 6
_EVIDENCE_TOP_K = 4
_EXCERPT_CHARS = 600
_PARTIAL_FACTOR = 0.5


@dataclass(frozen=True)
class InterviewLimits:
    max_components: int = 8
    max_files: int = 6
    per_component: int = 4


@dataclass(frozen=True)
class InterviewRequest:
    project: ProjectKey
    files: list[RiskFile]
    limits: InterviewLimits = field(default_factory=InterviewLimits)


@dataclass(frozen=True)
class InterviewQuestion:
    """One question for the leaver, or one the records already answer."""

    component: str
    text: str
    type: QuestionType
    triggers: tuple[Trigger, ...]
    files: tuple[str, ...]
    priority: float
    coverage: Coverage
    record_answer: str | None = None
    sources: tuple[str, ...] = ()


@dataclass(frozen=True)
class InterviewPlan:
    questions: list[InterviewQuestion]
    answered_by_records: list[InterviewQuestion]


@dataclass(frozen=True)
class _Run:
    llm: LLMClient
    store: VectorStore
    filters: RetrievalFilters
    limits: InterviewLimits


def plan_interview(
    llm: LLMClient, store: VectorStore, request: InterviewRequest
) -> InterviewPlan:
    """The questions to ask the leaver, most urgent first."""
    run = _Run(llm, store, RetrievalFilters(project_id=request.project), request.limits)
    components = components_of(
        request.files, request.limits.max_components, request.limits.max_files
    )
    planned = [q for component in components for q in _plan_component(run, component)]
    asked = [q for q in planned if q.coverage != Coverage.ANSWERED]
    asked.sort(key=lambda q: (-q.priority, q.component, q.text))
    answered = [q for q in planned if q.coverage == Coverage.ANSWERED]
    return InterviewPlan(questions=asked, answered_by_records=answered)


def _plan_component(run: _Run, component: Component) -> list[InterviewQuestion]:
    drafts = _draft(run, component)
    if not drafts:
        return []
    verdicts = _judge(run, drafts)
    return [
        _question(component, draft, verdict)
        for draft, verdict in zip(drafts, verdicts, strict=True)
    ]


def _excerpts(run: _Run, query: str, top_k: int) -> list[Excerpt]:
    chunks = retrieve(query, run.llm, run.store, top_k=top_k, filters=run.filters)
    return [Excerpt(c.filename, c.text[:_EXCERPT_CHARS]) for c in chunks]


def _draft(run: _Run, component: Component) -> list[DraftQuestion]:
    query = " ".join(file.path for file in component.files)
    context = _excerpts(run, query, _CONTEXT_TOP_K)
    prompt = generate_prompt(component, context, run.limits.per_component)
    raw = run.llm.generate(prompt)
    try:
        return parse_questions(raw, component, run.limits.per_component)
    except ValueError as exc:
        logger.warning("No interview questions for %s: %s", component.name, exc)
        return []


def _judge(run: _Run, drafts: list[DraftQuestion]) -> list[Verdict]:
    evidence = [_excerpts(run, draft.text, _EVIDENCE_TOP_K) for draft in drafts]
    raw = run.llm.generate(judge_prompt(drafts, evidence))
    try:
        return parse_verdicts(raw, len(drafts))
    except ValueError as exc:
        logger.warning("Interview questions left unjudged: %s", exc)
        return [Verdict(Coverage.OPEN, None, ()) for _ in drafts]


def _question(
    component: Component, draft: DraftQuestion, verdict: Verdict
) -> InterviewQuestion:
    urgency = max(TRIGGER_WEIGHT[t] for t in draft.triggers)
    factor = _PARTIAL_FACTOR if verdict.coverage == Coverage.PARTIAL else 1.0
    return InterviewQuestion(
        component=component.name,
        text=draft.text,
        type=draft.type,
        triggers=draft.triggers,
        files=draft.files,
        priority=round(component.risk * urgency * factor, 3),
        coverage=verdict.coverage,
        record_answer=verdict.answer,
        sources=verdict.sources,
    )
