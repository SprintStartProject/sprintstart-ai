"""What the model drafts for a phase, and how that draft is read.

The generation models, the prompt, the correction round, and the resolution of a
draft against the retrieved evidence: everything that depends on the model's output
rather than on the order of the pipeline stages in assembly.phase.job.
"""

import json
import logging
from typing import cast

from pydantic import BaseModel, Field, ValidationError, field_validator

from assembly.citations import resolve_citations
from assembly.models import CheckOption, CheckQuestion
from assembly.phase.models import (
    PhaseResource,
    PhaseStep,
    PhaseTask,
)
from assembly.similarity import OVERLAP_THRESHOLD, text_overlap
from assembly.stages import (
    AssemblyError,
    evidence_line,
)
from corpus.types import ScoredChunk
from llm.base import Message
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

TOP_K = 12
_MAX_PROMPT_TEXT = 4000

# How far into the phase block the step list should go; questions are quiz
# extras, not the phase's substance.
_MIN_STEPS = 1
_MAX_STEPS = 8


class _GenTask(BaseModel):
    title: str = ""
    description: str = ""


class _GenResource(BaseModel):
    title: str = ""
    url: str = ""


class _GenStep(BaseModel):
    title: str
    description: str = ""
    tasks: list[_GenTask] = Field(default_factory=list[_GenTask])
    resources: list[_GenResource] = Field(default_factory=list[_GenResource])
    estimated_minutes: int | None = None
    expected_outcome: str = ""
    chunk_ids: list[str] = Field(default_factory=list[str])
    key: str = ""
    blocked_by: list[str] = Field(default_factory=list[str])

    @field_validator("tasks", mode="before")
    @classmethod
    def normalize_string_tasks(cls, value: object) -> object:
        """Accept the compact task form models commonly emit.

        The requested schema uses task objects, but some otherwise valid model
        responses return ``tasks`` as a list of imperative strings. A string
        already contains the task's title; preserving it with an empty
        description is lossless and keeps a harmless shape deviation from
        discarding the whole phase.
        """
        if value is None:
            return []
        if not isinstance(value, list):
            return value
        items = cast(list[object], value)
        return [
            {"title": item, "description": ""} if isinstance(item, str) else item
            for item in items
        ]


class _GenOption(BaseModel):
    label: str = ""
    correct: bool = False


class _GenQuestion(BaseModel):
    type: str = ""
    question: str = ""
    explanation: str | None = None
    correct_answer: str | None = None
    options: list[_GenOption] = Field(default_factory=list[_GenOption])
    key: str = ""
    blocked_by: list[str] = Field(default_factory=list[str])

    @field_validator("options", mode="before")
    @classmethod
    def normalize_null_options(cls, value: object) -> object:
        return [] if value is None else value


class GenPayload(BaseModel):
    steps: list[_GenStep] = Field(default_factory=list[_GenStep])
    questions: list[_GenQuestion] = Field(default_factory=list[_GenQuestion])


def phase_query(
    title: str,
    description: str,
    prompt: str,
    industry: str | None = None,
) -> str:
    parts: list[str] = []
    if industry and industry.strip():
        parts.append(f"Industry: {industry.strip()}")
    parts.extend([title, description, prompt])
    return " ".join(p for p in parts if p.strip())[:_MAX_PROMPT_TEXT]


def build_prompt(
    title: str,
    description: str,
    prompt: str,
    chunks: list[ScoredChunk],
    industry: str | None = None,
) -> list[Message]:
    evidence = "\n".join(evidence_line(c) for c in chunks)
    description_line = f"\nPhase description: {description}" if description else ""
    industry_line = (
        f"\nProject industry/domain: {industry.strip()}"
        if industry and industry.strip()
        else ""
    )
    system = (
        "You are authoring one phase of a software-team onboarding path from "
        "the team's own knowledge base. The phase already has a title and an "
        "author's instruction (the prompt) prescribing what it should teach. "
        "The evidence below is the material this team actually has; produce the "
        "phase's content so a new member can complete it in this project.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences):\n"
        '{"steps": [{"key": str, "title": str, "description": str, "tasks": '
        '[{"title": str, "description": str}], "resources": [{"title": str, '
        '"url": str}], "estimated_minutes": int|null, "expected_outcome": str, '
        '"chunk_ids": [str], "blocked_by": [str]}], "questions": [{"key": str, '
        '"type": "MULTIPLE_CHOICE"|"SHORT_TEXT", "question": str, "explanation": '
        'str, "correct_answer": str|null, "options": [{"label": str, "correct": '
        'bool}], "blocked_by": [str]}]}\n\n'
        "Rules:\n"
        "1. Make every step actionable and specific to this project: exact "
        "paths, exact commands, exact conventions. Generic software advice is "
        "worse than nothing because it costs the newcomer attention.\n"
        "2. EVERY step MUST reference at least one chunk id from the evidence; "
        "do not invent sources. A step that cites nothing will be dropped.\n"
        f"3. Write {_MIN_STEPS}-{_MAX_STEPS} steps, ordered as a newcomer would "
        "do them, covering exactly what the prompt prescribes.\n"
        "4. Under each step give 0-4 small concrete tasks and at most 2 "
        "resources lifted from the evidence (a resource url must come from the "
        "evidence, if any).\n"
        "5. 'estimated_minutes' is the time a newcomer should expect to spend; "
        "'expected_outcome' says what they can do once the step is done.\n"
        "6. Write 2-4 short knowledge-check questions about this phase's "
        "content, measuring understanding rather than recall. "
        "MULTIPLE_CHOICE: 3-4 options, exactly one correct (never mark most or "
        "all correct), plausible misconceptions as distractors, and an "
        "explanation. SHORT_TEXT: ask the learner to explain a consequence, "
        "trade-off or reasoning in their own words, with a non-empty "
        "'correct_answer' capturing the expected idea and 'options' set to [].\n"
        "7. Base everything strictly on the evidence below; never invent facts "
        "it does not support. If the evidence says nothing about part of the "
        "prompt, omit that part — an honest gap beats invented content.\n"
        "8. Dependency edges. Give every step and question a short, unique key "
        '(e.g. "s1", "q2"). Set "blocked_by" to the keys of the items a '
        "newcomer must finish first for this one to make sense — e.g. read the "
        "README before running its build command, understand the domain before "
        "reviewing its conventions, or answer a knowledge check after the step "
        'that teaches it. Leave "blocked_by" empty only when the item is '
        "genuinely independent and can be done at any point; never use it just "
        "to restate the normal reading order, and only reference keys you "
        "defined.\n"
    )
    user = (
        f"Phase: {title}{description_line}{industry_line}\n\n"
        f"Author's prompt:\n{prompt[:_MAX_PROMPT_TEXT] or '(none)'}\n\n"
        f"Evidence:\n{evidence}"
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def parse_payload(raw: str) -> GenPayload:
    try:
        payload = json.loads(extract_json_object(raw), strict=False)
        return GenPayload.model_validate(payload)
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise AssemblyError(f"invalid phase output: {exc}") from exc


def correction_prompt(
    messages: list[Message], raw: str, error: AssemblyError
) -> list[Message]:
    """Ask once for the same answer with only its JSON/shape corrected."""
    return [
        *messages,
        Message(role="assistant", content=raw),
        Message(
            role="user",
            content=(
                "That response could not be validated: "
                f"{error}. Return the same grounded content again as one valid "
                "JSON object matching the schema exactly. Task entries must be "
                'objects like {"title": "...", "description": "..."}. Return '
                "JSON only, with all quotes escaped and all commas present."
            ),
        ),
    ]


def resolve_steps(
    payload: GenPayload, chunks: list[ScoredChunk]
) -> tuple[list[PhaseStep], int]:
    """Keep the grounded, non-duplicate steps; return them with the count dropped.

    A step goes when it has no title, when it cites nothing real, or when it
    restates a step already kept. Kept steps keep the model's ordering.
    """
    chunks_by_id = {c.id: c for c in chunks}
    kept: list[PhaseStep] = []
    dropped = 0
    seen_texts: list[str] = []
    seen_titles: list[str] = []

    for item in payload.steps:
        if not item.title.strip():
            dropped += 1
            continue

        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if not citations:
            logger.info("Dropped ungrounded phase step %r", item.title)
            dropped += 1
            continue

        full = f"{item.title} {item.description}".strip()
        if any(
            text_overlap(full, prior) > OVERLAP_THRESHOLD
            or text_overlap(item.title, prior_title) > OVERLAP_THRESHOLD
            for prior, prior_title in zip(seen_texts, seen_titles, strict=False)
        ):
            logger.info("Dropped phase step restating %r", item.title)
            dropped += 1
            continue

        seen_texts.append(full)
        seen_titles.append(item.title)
        tasks = [
            PhaseTask(title=t.title, description=t.description)
            for t in item.tasks
            if t.title.strip()
        ]
        resources = [
            PhaseResource(title=r.title, url=r.url)
            for r in item.resources
            if r.title.strip() and r.url.strip()
        ]
        minutes = item.estimated_minutes if (item.estimated_minutes or 0) > 0 else None
        kept.append(
            PhaseStep(
                title=item.title,
                description=item.description,
                tasks=tasks,
                resources=resources,
                estimated_minutes=minutes,
                expected_outcome=item.expected_outcome,
                key=item.key.strip(),
                blocked_by=[k for k in item.blocked_by if k.strip()],
            )
        )

    return kept, dropped


def _validate_question(item: _GenQuestion) -> CheckQuestion | None:
    """Drop a question that doesn't meet the check-question constraints.

    The phase's quiz obeys the shape the backend's knowledge-check UI expects.
    Positions are assigned later over survivors only.
    """
    if not item.question.strip():
        return None

    if item.type == "MULTIPLE_CHOICE":
        options = [o for o in item.options if o.label.strip()]
        correct_count = sum(1 for o in options if o.correct)
        # Below the prompt's own guidance against near-all-correct questions:
        # a question where every option is "correct" discriminates nothing and
        # is dropped regardless of how well the model followed the prompt.
        if len(options) < 2 or correct_count == 0 or correct_count == len(options):
            return None
        return CheckQuestion(
            position=0,
            type="MULTIPLE_CHOICE",  # type: ignore[arg-type]
            question=item.question,
            explanation=item.explanation,
            options=[
                CheckOption(position=i, label=o.label, correct=o.correct)
                for i, o in enumerate(options)
            ],
            key=item.key.strip(),
            blocked_by=[k for k in item.blocked_by if k.strip()],
        )

    if item.type == "SHORT_TEXT":
        if not item.correct_answer or not item.correct_answer.strip():
            return None
        return CheckQuestion(
            position=0,
            type="SHORT_TEXT",  # type: ignore[arg-type]
            question=item.question,
            explanation=item.explanation,
            correct_answer=item.correct_answer,
            key=item.key.strip(),
            blocked_by=[k for k in item.blocked_by if k.strip()],
        )

    return None


def resolve_questions(payload: GenPayload) -> tuple[list[CheckQuestion], int]:
    candidates = (_validate_question(item) for item in payload.questions)
    questions = [q for q in candidates if q is not None]
    dropped = len(payload.questions) - len(questions)
    for position, question in enumerate(questions):
        question.position = position
    return questions, dropped


def resolve_dependencies(
    steps: list[PhaseStep], questions: list[CheckQuestion]
) -> None:
    """Rewrite each kept item's ``blocked_by`` keys against the kept items.

    Steps and questions share one key space, ordered steps first then
    questions — the order a newcomer meets them. A reference survives only when
    it points at an item *earlier* in that order (so the assembled graph is
    always acyclic), the referenced key is non-empty, and the target is not the
    item itself. References to a step that grounding dropped, or to an unknown
    key, resolve to nothing and are dropped with the edge.
    """
    order: list[str] = [s.key for s in steps if s.key]
    order += [q.key for q in questions if q.key]
    index = {key: position for position, key in enumerate(order)}

    def resolve(item_key: str, blocked_by: list[str]) -> list[str]:
        if not item_key:
            return []
        return [
            key
            for key in blocked_by
            if key != item_key and key in index and index[key] < index[item_key]
        ]

    for step in steps:
        step.blocked_by = resolve(step.key, step.blocked_by)
    for question in questions:
        question.blocked_by = resolve(question.key, question.blocked_by)
