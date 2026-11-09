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

import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime

from assembly.fingerprint import fingerprint_gate
from assembly.models import CheckQuestion
from assembly.phase.draft import (
    TOP_K,
    GenPayload,
    build_prompt,
    correction_prompt,
    parse_payload,
    phase_query,
    resolve_dependencies,
    resolve_questions,
    resolve_steps,
)
from assembly.phase.models import (
    PhaseOutcome,
    PhaseProvenance,
    PhaseStep,
)
from assembly.progress import ProgressEvent, ProgressStream, drain
from assembly.stages import (
    AssemblyError,
    Generation,
    Retrieval,
    Search,
    dump,
    generate_payload,
    give_up,
    retrieve_evidence,
)
from corpus.types import RetrievalFilters, ScoredChunk
from llm.base import LLMClient
from store.base import VectorStore

logger = logging.getLogger(__name__)

_DONE_EMPTY = "No phase content could be assembled"


@dataclass(frozen=True)
class _Phase:
    """What one run assembles: the phase and the project it is assembled for."""

    title: str
    description: str
    prompt: str
    filters: RetrievalFilters
    industry: str | None


@dataclass(frozen=True)
class _Counts:
    """How much evidence a run drew on and how much of the draft it dropped."""

    chunks_retrieved: int = 0
    chunks_collapsed: int = 0
    steps_dropped: int = 0
    questions_dropped: int = 0


@dataclass(frozen=True)
class _Grounded:
    steps: list[PhaseStep]
    questions: list[CheckQuestion]
    counts: _Counts


def _skipped(note: str, counts: _Counts = _Counts()) -> PhaseOutcome:
    return PhaseOutcome(
        status="skipped",
        chunks_retrieved=counts.chunks_retrieved,
        chunks_collapsed=counts.chunks_collapsed,
        steps_dropped=counts.steps_dropped,
        questions_dropped=counts.questions_dropped,
        notes=[note],
    )


def _gate(
    progress: ProgressStream,
    store: VectorStore,
    last_fingerprint: str | None,
    industry: str | None,
) -> tuple[str | None, list[ProgressEvent], PhaseOutcome | None]:
    """Answer from the fingerprint alone when the corpus is empty or unchanged."""
    return fingerprint_gate(
        progress,
        store,
        last_fingerprint,
        make_unchanged=lambda: PhaseOutcome(
            status="unchanged", notes=["corpus unchanged since the last assembly"]
        ),
        make_empty=lambda: PhaseOutcome(status="skipped", notes=["corpus is empty"]),
        unchanged_label="Nothing changed — the cached phase content is current",
        empty_warning_label="The project has no indexed material yet",
        empty_done_label=_DONE_EMPTY,
        extra_fingerprint_material=[f"industry:{industry}"] if industry else None,
    )


def _stop(
    progress: ProgressStream, outcome: PhaseOutcome, warning: str
) -> Generator[ProgressEvent, None, PhaseOutcome]:
    return (yield from give_up(progress, outcome, warning=warning, done=_DONE_EMPTY))


def _gather(
    progress: ProgressStream, llm: LLMClient, store: VectorStore, phase: _Phase
) -> Generator[ProgressEvent, None, tuple[list[ScoredChunk], int]]:
    question = phase_query(phase.title, phase.description, phase.prompt, phase.industry)
    return (
        yield from retrieve_evidence(
            progress,
            llm,
            store,
            Retrieval(
                searches=[
                    Search(
                        label=f"Searching the project for: {phase.title}",
                        question=question,
                    )
                ],
                top_k=TOP_K,
                filters=phase.filters,
            ),
        )
    )


def _write(
    progress: ProgressStream, llm: LLMClient, phase: _Phase, chunks: list[ScoredChunk]
) -> Generator[ProgressEvent, None, GenPayload | AssemblyError]:
    yield progress.stage(
        "generating", f"Writing the phase from {len(chunks)} source(s)"
    )
    return (
        yield from generate_payload(
            progress,
            llm,
            build_prompt(
                phase.title, phase.description, phase.prompt, chunks, phase.industry
            ),
            Generation(
                parse=parse_payload,
                correct=correction_prompt,
                logger=logger,
                failure="Phase assembly attempt %d failed for %r: %s",
                subject=phase.title,
            ),
        )
    )


def _ground(
    progress: ProgressStream,
    payload: GenPayload,
    chunks: list[ScoredChunk],
    collapsed: int,
) -> Generator[ProgressEvent, None, _Grounded]:
    """Keep what the evidence supports, announcing each survivor and each drop."""
    yield progress.stage("grounding", "Checking every step cites its source")
    steps, steps_dropped = resolve_steps(payload, chunks)
    questions, questions_dropped = resolve_questions(payload)
    resolve_dependencies(steps, questions)
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
    counts = _Counts(len(chunks), collapsed, steps_dropped, questions_dropped)
    return _Grounded(steps, questions, counts)


def _notes(counts: _Counts) -> list[str]:
    notes: list[str] = []
    if counts.chunks_collapsed:
        notes.append(f"collapsed {counts.chunks_collapsed} redundant source chunk(s)")
    if counts.steps_dropped:
        notes.append(f"dropped {counts.steps_dropped} ungrounded or duplicate step(s)")
    if counts.questions_dropped:
        notes.append(f"dropped {counts.questions_dropped} invalid question(s)")
    return notes


def _assembled(
    llm: LLMClient, fingerprint: str | None, grounded: _Grounded
) -> PhaseOutcome:
    counts = grounded.counts
    notes = _notes(counts)
    return PhaseOutcome(
        status="assembled",
        steps=grounded.steps,
        check_questions=grounded.questions,
        provenance=PhaseProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
            notes=notes,
        ),
        chunks_retrieved=counts.chunks_retrieved,
        chunks_collapsed=counts.chunks_collapsed,
        steps_dropped=counts.steps_dropped,
        questions_dropped=counts.questions_dropped,
        notes=notes,
    )


def stream_phase(
    llm: LLMClient,
    store: VectorStore,
    *,
    phase_title: str,
    phase_description: str = "",
    phase_prompt: str,
    project_id: str,
    last_fingerprint: str | None = None,
    industry: str | None = None,
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
    normalized = industry.strip() if industry and industry.strip() else None
    phase = _Phase(
        phase_title,
        phase_description,
        phase_prompt,
        RetrievalFilters(project_id=project_id),
        normalized,
    )

    progress = ProgressStream("phase")
    fingerprint, early_events, early_outcome = _gate(
        progress, store, last_fingerprint, normalized
    )
    if early_outcome is not None:
        yield from early_events
        return early_outcome

    chunks, collapsed = yield from _gather(progress, llm, store, phase)
    if not chunks:
        return (
            yield from _stop(
                progress,
                _skipped("no grounding evidence retrieved for this phase"),
                "Nothing in the project matched this phase",
            )
        )

    payload = yield from _write(progress, llm, phase, chunks)
    if isinstance(payload, AssemblyError):
        skipped = _skipped(str(payload), _Counts(len(chunks), collapsed))
        return (
            yield from _stop(
                progress, skipped, "The generated content could not be read"
            )
        )

    grounded = yield from _ground(progress, payload, chunks, collapsed)
    if not grounded.steps and not grounded.questions:
        skipped = _skipped(
            "no grounded steps or valid questions assembled", grounded.counts
        )
        return (
            yield from _stop(progress, skipped, "Nothing usable survived grounding")
        )

    outcome = _assembled(llm, fingerprint, grounded)
    yield progress.done("Phase content ready", dump(outcome))
    return outcome


def assemble_phase(
    llm: LLMClient,
    store: VectorStore,
    *,
    phase_title: str,
    phase_description: str = "",
    phase_prompt: str,
    project_id: str,
    last_fingerprint: str | None = None,
    industry: str | None = None,
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
            industry=industry,
        )
    )
