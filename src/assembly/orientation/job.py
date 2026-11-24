"""Assembly of a task-scoped orientation packet from existing material.

Material the project already has, extracted and restructured by the step
somebody is on. Nothing is authored: a claim about this codebase that no
retrieved chunk supports is dropped, not softened.

Three mechanics carry that guarantee:

* **Retrieval is per step, not per task.** One query per step of the path to a
  pull request (set up → find the code → make the change → check locally → open
  the PR), so the evidence pool contains the setup docs *and* the review
  conventions rather than five restatements of the task's own subject.
* **The pool is deduplicated before the model sees it** — token overlap, no
  extra LLM call.
* **Grounding is per section, with no exempt kinds.** Every section of a packet
  asserts something about this project, so every section must cite.
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime

from assembly.fingerprint import fingerprint_gate
from assembly.orientation.draft import (
    STEP_LABEL,
    STEP_QUERIES,
    TOP_K_PER_STEP,
    GenPayload,
    build_prompt,
    correction_prompt,
    parse_payload,
    resolve_sections,
    task_query,
)
from assembly.orientation.models import (
    STEP_ORDER,
    OrientationOutcome,
    OrientationPacket,
    OrientationProvenance,
    OrientationSection,
    OrientationSource,
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
    sources_drawn_on,
)
from corpus.types import RetrievalFilters, ScoredChunk
from llm.base import LLMClient
from store.base import VectorStore

logger = logging.getLogger(__name__)

_DONE_EMPTY = "No orientation could be assembled"


@dataclass(frozen=True)
class _Task:
    """The task a packet is assembled for, and the projects it may draw on."""

    title: str
    body: str
    labels: list[str]
    touched_paths: list[str]
    project_ids: frozenset[str]


@dataclass(frozen=True)
class _Counts:
    """How much evidence a run drew on and how many sections it dropped."""

    chunks_retrieved: int = 0
    chunks_collapsed: int = 0
    sections_dropped: int = 0


@dataclass(frozen=True)
class _Grounded:
    summary: str
    sections: list[OrientationSection]
    chunks: list[ScoredChunk]
    counts: _Counts


def _skipped(note: str, counts: _Counts = _Counts()) -> OrientationOutcome:
    return OrientationOutcome(
        status="skipped",
        chunks_retrieved=counts.chunks_retrieved,
        chunks_collapsed=counts.chunks_collapsed,
        sections_dropped=counts.sections_dropped,
        notes=[note],
    )


def _gate(
    progress: ProgressStream,
    store: VectorStore,
    last_fingerprint: str | None,
    project_ids: frozenset[str],
) -> tuple[str | None, list[ProgressEvent], OrientationOutcome | None]:
    """Answer from the fingerprint alone when the corpus is empty or unchanged."""
    return fingerprint_gate(
        progress,
        store,
        last_fingerprint,
        make_unchanged=lambda: OrientationOutcome(
            status="unchanged", notes=["corpus unchanged since the cached packet"]
        ),
        make_empty=lambda: OrientationOutcome(
            status="skipped", notes=["corpus is empty"]
        ),
        unchanged_label="Nothing changed — the cached packet is current",
        empty_warning_label="The project has no indexed material yet",
        empty_done_label=_DONE_EMPTY,
        project_ids=project_ids,
    )


def _stop(
    progress: ProgressStream, outcome: OrientationOutcome, warning: str
) -> Generator[ProgressEvent, None, OrientationOutcome]:
    return (yield from give_up(progress, outcome, warning=warning, done=_DONE_EMPTY))


def _gather(
    progress: ProgressStream, llm: LLMClient, store: VectorStore, task: _Task
) -> Generator[ProgressEvent, None, tuple[list[ScoredChunk], int]]:
    """One search per step of the packet, pooled."""
    query = task_query(task.title, task.body, task.labels, task.touched_paths)
    searches = [
        Search(
            label=f"Searching the project: {STEP_LABEL[step]}",
            question=f"{query} {STEP_QUERIES[step]}",
        )
        for step in STEP_ORDER
    ]
    return (
        yield from retrieve_evidence(
            progress,
            llm,
            store,
            Retrieval(
                searches=searches,
                top_k=TOP_K_PER_STEP,
                filters=RetrievalFilters(project_ids=task.project_ids),
            ),
        )
    )


def _write(
    progress: ProgressStream, llm: LLMClient, task: _Task, chunks: list[ScoredChunk]
) -> Generator[ProgressEvent, None, GenPayload | AssemblyError]:
    yield progress.stage(
        "generating", f"Writing the packet from {len(chunks)} source(s)"
    )
    return (
        yield from generate_payload(
            progress,
            llm,
            build_prompt(
                task.title, task.body, task.labels, task.touched_paths, chunks
            ),
            Generation(
                parse=parse_payload,
                correct=correction_prompt,
                logger=logger,
                failure="Orientation assembly attempt %d failed for task %r: %s",
                subject=task.title,
            ),
        )
    )


def _ground(
    progress: ProgressStream,
    payload: GenPayload,
    chunks: list[ScoredChunk],
    collapsed: int,
) -> Generator[ProgressEvent, None, _Grounded]:
    """Keep the sections the evidence supports, announcing each survivor."""
    yield progress.stage("grounding", "Checking every section cites its source")
    sections, dropped = resolve_sections(payload, chunks)
    for section in sections:
        yield progress.item(
            section.model_dump(mode="json"),
            f"{STEP_LABEL[section.step]}: {section.title}",
        )
    # A dropped section is a gap the watcher should see, not a silent omission --
    # unless every section went, which the caller states more directly.
    if sections and dropped:
        yield progress.warning(f"Dropped {dropped} section(s) with no source")
    counts = _Counts(len(chunks), collapsed, dropped)
    return _Grounded(payload.summary.strip(), sections, chunks, counts)


def _notes(counts: _Counts) -> list[str]:
    notes: list[str] = []
    if counts.chunks_collapsed:
        notes.append(f"collapsed {counts.chunks_collapsed} redundant source chunk(s)")
    if counts.sections_dropped:
        notes.append(
            f"dropped {counts.sections_dropped} ungrounded or duplicate section(s)"
        )
    return notes


def _assembled(
    llm: LLMClient,
    fingerprint: str | None,
    task: _Task,
    grounded: _Grounded,
) -> OrientationOutcome:
    counts = grounded.counts
    notes = _notes(counts)
    return OrientationOutcome(
        status="assembled",
        packet=OrientationPacket(
            task_title=task.title,
            summary=grounded.summary,
            sections=grounded.sections,
            sources=sources_drawn_on(
                (c for s in grounded.sections for c in s.citations),
                grounded.chunks,
                OrientationSource,
            ),
        ),
        provenance=OrientationProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
            notes=notes,
        ),
        chunks_retrieved=counts.chunks_retrieved,
        chunks_collapsed=counts.chunks_collapsed,
        sections_dropped=counts.sections_dropped,
        notes=notes,
    )


def stream_orientation(
    llm: LLMClient,
    store: VectorStore,
    *,
    task_title: str,
    project_ids: frozenset[str],
    task_body: str = "",
    labels: list[str] | None = None,
    touched_paths: list[str] | None = None,
    last_fingerprint: str | None = None,
) -> Generator[ProgressEvent, None, OrientationOutcome]:
    """Assemble a packet, yielding live progress and returning the final outcome.

    This is the single implementation: :func:`assemble_orientation` drives it to
    completion for the non-streaming path, and the streaming route relays its
    events. So the packet a hire watches assemble is byte-for-byte the packet the
    cached call would have produced — the stream is a view, never a second answer.

    Retrieval runs once per step (set up → find the code → … → open the PR), each
    announced as a ``stage``. Sections are emitted as ``item`` events only after
    they clear the per-section grounding gate, so nothing ungrounded is ever shown.
    """
    task = _Task(task_title, task_body, labels or [], touched_paths or [], project_ids)
    progress = ProgressStream("orientation")
    fingerprint, early_events, early_outcome = _gate(
        progress, store, last_fingerprint, project_ids
    )
    if early_outcome is not None:
        yield from early_events
        return early_outcome

    chunks, collapsed = yield from _gather(progress, llm, store, task)
    if not chunks:
        return (
            yield from _stop(
                progress,
                _skipped("no grounding evidence retrieved for this task"),
                "Nothing in the project matched this task closely enough",
            )
        )

    payload = yield from _write(progress, llm, task, chunks)
    if isinstance(payload, AssemblyError):
        skipped = _skipped(str(payload), _Counts(len(chunks), collapsed))
        return (
            yield from _stop(
                progress, skipped, "The generated packet could not be read"
            )
        )

    grounded = yield from _ground(progress, payload, chunks, collapsed)
    if not grounded.sections:
        skipped = _skipped(
            "no grounded sections in the assembled packet", grounded.counts
        )
        return (
            yield from _stop(
                progress, skipped, "Every section was dropped for lack of a source"
            )
        )

    outcome = _assembled(llm, fingerprint, task, grounded)
    yield progress.done("Orientation ready", dump(outcome))
    return outcome


def assemble_orientation(
    llm: LLMClient,
    store: VectorStore,
    *,
    task_title: str,
    project_ids: frozenset[str],
    task_body: str = "",
    labels: list[str] | None = None,
    touched_paths: list[str] | None = None,
    last_fingerprint: str | None = None,
) -> OrientationOutcome:
    """Assemble a step-segmented orientation packet for one task.

    ``last_fingerprint`` is the corpus fingerprint the caller recorded the last
    time it assembled a packet *for this task*: a packet is cached against the
    task and the corpus it was built from, so a corpus that has moved on
    regenerates rather than serving guidance that no longer matches the code.

    Takes nothing about the individual hire, deliberately — orientation is a
    property of the task, so two people claiming the same task read the same
    packet and can talk about it.

    Backed by :func:`stream_orientation` so the non-streaming and streaming paths
    are the same computation.
    """
    return drain(
        stream_orientation(
            llm,
            store,
            task_title=task_title,
            project_ids=project_ids,
            task_body=task_body,
            labels=labels,
            touched_paths=touched_paths,
            last_fingerprint=last_fingerprint,
        )
    )
