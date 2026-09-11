"""AI assembly of one blueprint phase's content from the project's corpus.

A blueprint phase can be ``AI_ENHANCED``: instead of an authored step list it
carries a prompt describing what the phase should teach. During personalization
the backend asks this service to fill such a phase — retrieve the project's own
material, then generate actionable steps (with tasks and resources) and a small
knowledge check, grounded in that evidence.

The pipeline is deliberately conservative, matching the sibling jobs in this
package:

* **Grounded** — a step must cite at least one retrieved chunk or it is dropped.
  Nothing is authored; a claim about the codebase that no retrieved chunk
  supports is dropped, not softened.
* **Idempotent** — :func:`stream_phase` short-circuits via the shared corpus
  fingerprint when ``last_fingerprint`` matches.
* **Honest** — an empty corpus, no retrieved evidence, or a generation whose
  every step was ungrounded yields ``skipped``, never invented content.

The backend owns persistence; the AI service is stateless and only returns the
assembled content.
"""

import json
import logging
from collections.abc import Generator
from datetime import UTC, datetime
from typing import cast

from pydantic import BaseModel, Field, ValidationError, field_validator

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.corpus import fingerprint_gate
from onboarding.models import CheckOption, CheckQuestion
from onboarding.phase_models import (
    PhaseOutcome,
    PhaseProvenance,
    PhaseResource,
    PhaseStep,
    PhaseTask,
)
from onboarding.progress import ProgressEvent, ProgressStream, drain
from onboarding.similarity import OVERLAP_THRESHOLD, text_overlap
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import RetrievalFilters, ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 12
_MIN_SCORE = 0.3
_MAX_PROMPT_TEXT = 4000
# Deterministic so two assemblies against the same corpus and prompt give
# materially the same phase — a hire who re-personalizes should not read
# different instructions than their teammates.
_TEMPERATURE = 0.0

# How far into the phase block the step list should go; questions are quiz
# extras, not the phase's substance.
_MIN_STEPS = 1
_MAX_STEPS = 8
_MAX_GENERATION_ATTEMPTS = 2


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


class _GenPayload(BaseModel):
    steps: list[_GenStep] = Field(default_factory=list[_GenStep])
    questions: list[_GenQuestion] = Field(default_factory=list[_GenQuestion])


# --- evidence ------------------------------------------------------------------


def _phase_query(title: str, description: str, prompt: str) -> str:
    parts = [title, description, prompt]
    return " ".join(p for p in parts if p.strip())[:_MAX_PROMPT_TEXT]


def _collapse_duplicates(chunks: list[ScoredChunk]) -> tuple[list[ScoredChunk], int]:
    """Drop chunks that restate one already kept, best-scoring first.

    The README and the wiki repeating each other is load the model pays for
    twice. Comparison is token overlap rather than embeddings: it needs no
    extra call, and a phase must be reproducible.
    """
    kept: list[ScoredChunk] = []
    collapsed = 0
    for chunk in sorted(chunks, key=lambda c: (-c.score, c.id)):
        if any(text_overlap(chunk.text, k.text) > OVERLAP_THRESHOLD for k in kept):
            collapsed += 1
            continue
        kept.append(chunk)
    return kept, collapsed


def _evidence_line(chunk: ScoredChunk) -> str:
    meta = chunk.artifact_type or "FILE"
    if chunk.language:
        meta += f"/{chunk.language}"
    return f"  [{chunk.id}] ({chunk.filename} | {meta}) {chunk.text}"


# --- prompt / parsing ----------------------------------------------------------


def _build_prompt(
    title: str,
    description: str,
    prompt: str,
    chunks: list[ScoredChunk],
) -> list[Message]:
    evidence = "\n".join(_evidence_line(c) for c in chunks)
    description_line = f"\nPhase description: {description}" if description else ""
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
        f"Phase: {title}{description_line}\n\n"
        f"Author's prompt:\n{prompt[:_MAX_PROMPT_TEXT] or '(none)'}\n\n"
        f"Evidence:\n{evidence}"
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _GenPayload:
    try:
        payload = json.loads(extract_json_object(raw), strict=False)
        return _GenPayload.model_validate(payload)
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid phase output: {exc}") from exc


def _correction_prompt(
    messages: list[Message], raw: str, error: ValueError
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


# --- resolution ----------------------------------------------------------------


def _resolve_steps(
    payload: _GenPayload, chunks: list[ScoredChunk]
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

    Mirrors ``onboarding.checks._validate_question`` so the phase's quiz obeys
    the same shape the backend's knowledge-check UI expects. Positions are
    assigned later over survivors only.
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


def _resolve_questions(payload: _GenPayload) -> tuple[list[CheckQuestion], int]:
    candidates = (_validate_question(item) for item in payload.questions)
    questions = [q for q in candidates if q is not None]
    dropped = len(payload.questions) - len(questions)
    for position, question in enumerate(questions):
        question.position = position
    return questions, dropped


def _resolve_dependencies(
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


# --- job -----------------------------------------------------------------------


def stream_phase(
    llm: LLMClient,
    store: VectorStore,
    *,
    phase_title: str,
    phase_description: str = "",
    phase_prompt: str,
    project_id: str,
    last_fingerprint: str | None = None,
) -> Generator[ProgressEvent, None, PhaseOutcome]:
    """Assemble one phase's content, yielding live progress and returning the outcome.

    This is the single implementation: :func:`assemble_phase` drives it to
    completion for the non-streaming path, and the streaming route relays its
    events — so the phase a user watches assemble is byte-for-byte the phase
    the cached call would return; the stream is a view, never a second answer.

    Retrieval is announced as a ``stage``; every step is emitted as an ``item``
    only after it clears the grounding gate, so nothing ungrounded is ever
    shown. ``skipped`` (empty corpus, no evidence, or nothing grounded) is an
    honest answer for the backend to persist as an empty phase.
    """
    progress = ProgressStream("phase")
    fingerprint, early_events, early_outcome = fingerprint_gate(
        progress,
        store,
        last_fingerprint,
        make_unchanged=lambda: PhaseOutcome(
            status="unchanged", notes=["corpus unchanged since the last assembly"]
        ),
        make_empty=lambda: PhaseOutcome(status="skipped", notes=["corpus is empty"]),
        unchanged_label="Nothing changed — the cached phase content is current",
        empty_warning_label="The project has no indexed material yet",
        empty_done_label="No phase content could be assembled",
    )
    if early_outcome is not None:
        yield from early_events
        return early_outcome

    yield progress.stage("retrieving", f"Searching the project for: {phase_title}")
    bm25_cache = BM25IndexCache()
    chunks = hybrid_retrieve(
        question=_phase_query(phase_title, phase_description, phase_prompt),
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=bm25_cache,
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
        filters=RetrievalFilters(project_id=project_id),
    )
    chunks, collapsed = _collapse_duplicates(chunks)

    if not chunks:
        outcome = PhaseOutcome(
            status="skipped",
            notes=["no grounding evidence retrieved for this phase"],
        )
        yield progress.warning("Nothing in the project matched this phase")
        yield progress.done("No phase content could be assembled", _dump(outcome))
        return outcome

    yield progress.stage(
        "generating", f"Writing the phase from {len(chunks)} source(s)"
    )
    messages = _build_prompt(phase_title, phase_description, phase_prompt, chunks)
    parse_error: ValueError | None = None
    payload: _GenPayload | None = None
    for attempt in range(_MAX_GENERATION_ATTEMPTS):
        raw = llm.generate(messages, temperature=_TEMPERATURE)
        try:
            payload = _parse_payload(raw)
            break
        except ValueError as exc:
            parse_error = exc
            logger.warning(
                "Phase assembly attempt %d failed for %r: %s",
                attempt + 1,
                phase_title,
                exc,
            )
            if attempt + 1 < _MAX_GENERATION_ATTEMPTS:
                yield progress.stage("generating", "Correcting invalid generated JSON")
                messages = _correction_prompt(messages, raw, exc)

    if payload is None:
        assert parse_error is not None
        outcome = PhaseOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            chunks_collapsed=collapsed,
            notes=[str(parse_error)],
        )
        yield progress.warning("The generated content could not be read")
        yield progress.done("No phase content could be assembled", _dump(outcome))
        return outcome

    yield progress.stage("grounding", "Checking every step cites its source")
    steps, steps_dropped = _resolve_steps(payload, chunks)
    questions, questions_dropped = _resolve_questions(payload)
    _resolve_dependencies(steps, questions)
    for step in steps:
        yield progress.item(step.model_dump(mode="json"), f"Step: {step.title}")
    for question in questions:
        yield progress.item(
            question.model_dump(mode="json"),
            f"Check question: {question.question[:60]}",
        )

    if steps_dropped:
        yield progress.warning(
            f"Dropped {steps_dropped} ungrounded or duplicate step(s)"
        )
    if questions_dropped:
        yield progress.warning(f"Dropped {questions_dropped} invalid question(s)")

    if not steps and not questions:
        outcome = PhaseOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            chunks_collapsed=collapsed,
            steps_dropped=steps_dropped,
            questions_dropped=questions_dropped,
            notes=["no grounded steps or valid questions assembled"],
        )
        yield progress.warning("Nothing usable survived grounding")
        yield progress.done("No phase content could be assembled", _dump(outcome))
        return outcome

    notes: list[str] = []
    if collapsed:
        notes.append(f"collapsed {collapsed} redundant source chunk(s)")
    if steps_dropped:
        notes.append(f"dropped {steps_dropped} ungrounded or duplicate step(s)")
    if questions_dropped:
        notes.append(f"dropped {questions_dropped} invalid question(s)")

    outcome = PhaseOutcome(
        status="assembled",
        steps=steps,
        check_questions=questions,
        provenance=PhaseProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
            notes=notes,
        ),
        chunks_retrieved=len(chunks),
        chunks_collapsed=collapsed,
        steps_dropped=steps_dropped,
        questions_dropped=questions_dropped,
        notes=notes,
    )
    yield progress.done("Phase content ready", _dump(outcome))
    return outcome


def _dump(outcome: PhaseOutcome) -> dict[str, object]:
    """The outcome as a JSON-safe dict for a ``done`` event's ``result``."""
    return outcome.model_dump(mode="json")


def assemble_phase(
    llm: LLMClient,
    store: VectorStore,
    *,
    phase_title: str,
    phase_description: str = "",
    phase_prompt: str,
    project_id: str,
    last_fingerprint: str | None = None,
) -> PhaseOutcome:
    """Assemble one phase's content for the non-streaming path.

    Backed by :func:`stream_phase` so the sync and streaming paths are the same
    computation.
    """
    return drain(
        stream_phase(
            llm,
            store,
            phase_title=phase_title,
            phase_description=phase_description,
            phase_prompt=phase_prompt,
            project_id=project_id,
            last_fingerprint=last_fingerprint,
        )
    )
