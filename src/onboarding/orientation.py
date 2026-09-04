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

import json
import logging
from collections.abc import Generator
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.corpus import fingerprint_gate
from onboarding.orientation_models import (
    STEP_ORDER,
    OrientationOutcome,
    OrientationPacket,
    OrientationProvenance,
    OrientationSection,
    OrientationSource,
    OrientationStep,
)
from onboarding.progress import ProgressEvent, ProgressStream, drain
from onboarding.similarity import OVERLAP_THRESHOLD, text_overlap
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K_PER_STEP = 6
_MIN_SCORE = 0.3
_MAX_TASK_TEXT = 4000

# Re-running against an unchanged corpus and the same task must give materially
# the same packet. A packet is disposable, but a hire who reloads the page and
# reads different instructions has no reason to trust either version.
_TEMPERATURE = 0.0

# What each step's retrieval is looking for. These are *queries*, not prose the
# hire ever sees — they exist so the evidence pool spans the whole path to a PR
# instead of five rephrasings of the task's own subject.
_STEP_QUERIES: dict[OrientationStep, str] = {
    "SET_UP": (
        "development environment setup prerequisites install dependencies "
        "build and run the project locally configuration"
    ),
    "FIND_THE_CODE": (
        "project structure where modules live architecture packages layout "
        "which file is responsible for"
    ),
    "MAKE_THE_CHANGE": (
        "coding conventions style guide patterns used in this codebase "
        "how changes are usually implemented"
    ),
    "CHECK_LOCALLY": (
        "run the tests lint format type check verification commands before "
        "pushing definition of done"
    ),
    "OPEN_THE_PR": (
        "contributing guide pull request process branch naming commit message "
        "code review expectations"
    ),
}

# Short human labels for the live progress stream — what the watcher sees a step
# called while it is searched / grounded. Distinct from `_STEP_GUIDE` (prompt text)
# and `_STEP_QUERIES` (retrieval text): this is display only.
_STEP_LABEL: dict[OrientationStep, str] = {
    "SET_UP": "setting up",
    "FIND_THE_CODE": "finding the code",
    "MAKE_THE_CHANGE": "making the change",
    "CHECK_LOCALLY": "checking locally",
    "OPEN_THE_PR": "opening the PR",
}

# What the model is told each step is *for*, in the packet the hire reads.
_STEP_GUIDE = (
    "  SET_UP          - what must be working before touching this task\n"
    "  FIND_THE_CODE   - where in this repository the work lives\n"
    "  MAKE_THE_CHANGE - the conventions that apply to a change like this\n"
    "  CHECK_LOCALLY   - what to run, and what a good result looks like\n"
    "  OPEN_THE_PR     - how a change gets proposed and reviewed here\n"
)


class AssemblyError(Exception):
    """Raised when the LLM output for a packet cannot be parsed/validated."""


class _GenSection(BaseModel):
    step: str = ""
    title: str = ""
    body: str = ""
    chunk_ids: list[str] = Field(default_factory=list[str])


class _GenPayload(BaseModel):
    summary: str = ""
    sections: list[_GenSection] = Field(default_factory=list[_GenSection])


# --- evidence ------------------------------------------------------------------


def _task_query(
    task_title: str, task_body: str, labels: list[str], touched_paths: list[str]
) -> str:
    parts = [task_title, " ".join(labels), " ".join(touched_paths), task_body]
    return " ".join(p for p in parts if p.strip())[:_MAX_TASK_TEXT]


def _collapse_duplicates(chunks: list[ScoredChunk]) -> tuple[list[ScoredChunk], int]:
    """Drop chunks that restate one already kept, best-scoring first.

    The README and the wiki repeating each other is load the reader pays for
    twice. Comparison is token overlap rather than embeddings: it needs no
    extra call, and a packet must be reproducible.
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
    task_title: str,
    task_body: str,
    labels: list[str],
    touched_paths: list[str],
    chunks: list[ScoredChunk],
) -> list[Message]:
    evidence = "\n".join(_evidence_line(c) for c in chunks)
    steps = "|".join(STEP_ORDER)

    system = (
        "You prepare a new team member to do one specific task in a codebase "
        "they have never worked in. You do NOT write documentation: everything "
        "you produce is a restatement of material this team already has, "
        "reorganised around the task and the order somebody actually does "
        "things in.\n\n"
        "The packet is segmented by step:\n"
        f"{_STEP_GUIDE}\n"
        "Rules:\n"
        "1. Write only what the evidence below supports. Every section MUST "
        "list the evidence chunk ids it draws on; a section citing nothing is "
        "dropped, however useful it sounds. If the evidence says nothing about "
        "a step, omit that step -- an honest gap beats an invented answer.\n"
        "2. Be concrete and specific to this task: exact paths, exact commands, "
        "exact conventions. Generic software advice is worse than nothing "
        "because it costs the reader attention.\n"
        "3. Do not repeat yourself across sections. When two sources say the "
        "same thing, say it once and cite both.\n"
        "4. Keep each section short: a reader is doing the task, not studying.\n"
        "5. You may emit more than one section for a step, or none at all.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences):\n"
        '{"summary": str, "sections": [{"step": '
        f"{steps}"
        ', "title": str, "body": str (markdown), "chunk_ids": [str]}]}'
    )
    label_line = f"Labels: {', '.join(labels)}\n" if labels else ""
    paths_line = (
        f"Paths this task is expected to touch: {', '.join(touched_paths)}\n"
        if touched_paths
        else ""
    )
    user = (
        f"Task: {task_title}\n"
        f"{label_line}"
        f"{paths_line}"
        f"\n{task_body[:_MAX_TASK_TEXT] or '(no description)'}\n\n"
        f"Evidence:\n{evidence}"
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _GenPayload:
    try:
        return _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise AssemblyError(f"invalid orientation output: {exc}") from exc


# --- resolution ----------------------------------------------------------------


def _resolve_sections(
    payload: _GenPayload, chunks: list[ScoredChunk]
) -> tuple[list[OrientationSection], int]:
    """Keep the usable sections in step order; return them with the count dropped.

    A section goes when its step is not one of ours, when it has no body, when
    it cites nothing real, or when it restates a section already kept for the
    same step.
    """
    chunks_by_id = {c.id: c for c in chunks}
    kept: list[OrientationSection] = []
    dropped = 0

    for item in payload.sections:
        step = item.step.strip().upper()
        if step not in STEP_ORDER:
            logger.info("Dropped orientation section of unknown step %r", item.step)
            dropped += 1
            continue
        if not item.body.strip():
            dropped += 1
            continue

        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if not citations:
            logger.info("Dropped ungrounded orientation section %r", item.title)
            dropped += 1
            continue

        if any(
            s.step == step and text_overlap(s.body, item.body) > OVERLAP_THRESHOLD
            for s in kept
        ):
            logger.info("Dropped orientation section restating %r", item.title)
            dropped += 1
            continue

        kept.append(
            OrientationSection(
                step=step,  # type: ignore[arg-type]
                title=item.title.strip() or step.replace("_", " ").title(),
                body=item.body,
                citations=citations,
            )
        )

    kept.sort(key=lambda s: STEP_ORDER.index(s.step))
    return kept, dropped


def _sources_drawn_on(
    sections: list[OrientationSection], chunks: list[ScoredChunk]
) -> list[OrientationSource]:
    """The distinct material the kept sections actually cite, in first-use order."""
    chunks_by_id = {c.id: c for c in chunks}
    sources: list[OrientationSource] = []
    seen: set[tuple[str, str | None]] = set()
    for section in sections:
        for citation in section.citations:
            chunk = chunks_by_id.get(citation.chunk_id or "")
            key = (citation.filename, citation.source_url)
            if key in seen:
                continue
            seen.add(key)
            sources.append(
                OrientationSource(
                    filename=citation.filename,
                    source_url=citation.source_url,
                    artifact_type=chunk.artifact_type if chunk else None,
                )
            )
    return sources


# --- job -----------------------------------------------------------------------


def stream_orientation(
    llm: LLMClient,
    store: VectorStore,
    *,
    task_title: str,
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
    progress = ProgressStream("orientation")
    fingerprint, early_events, early_outcome = fingerprint_gate(
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
        empty_done_label="No orientation could be assembled",
    )
    if early_outcome is not None:
        yield from early_events
        return early_outcome

    task_query = _task_query(task_title, task_body, labels or [], touched_paths or [])
    bm25_cache = BM25IndexCache()
    by_id: dict[str, ScoredChunk] = {}
    for step in STEP_ORDER:
        yield progress.stage(
            "retrieving", f"Searching the project: {_STEP_LABEL[step]}"
        )
        for chunk in hybrid_retrieve(
            question=f"{task_query} {_STEP_QUERIES[step]}",
            llm=llm,
            store=store,
            top_k=_TOP_K_PER_STEP,
            min_score=_MIN_SCORE,
            bm25_cache=bm25_cache,
            exclude_roles=GROUNDING_EXCLUDED_ROLES,
        ):
            # A chunk retrieved for two steps keeps its better score; the model
            # decides which step it belongs under, and it appears once either way.
            existing = by_id.get(chunk.id)
            if existing is None or chunk.score > existing.score:
                by_id[chunk.id] = chunk
    chunks, collapsed = _collapse_duplicates(list(by_id.values()))

    if not chunks:
        outcome = OrientationOutcome(
            status="skipped", notes=["no grounding evidence retrieved for this task"]
        )
        yield progress.warning(
            "Nothing in the project matched this task closely enough"
        )
        yield progress.done("No orientation could be assembled", _dump(outcome))
        return outcome

    yield progress.stage(
        "generating", f"Writing the packet from {len(chunks)} source(s)"
    )
    raw = llm.generate(
        _build_prompt(task_title, task_body, labels or [], touched_paths or [], chunks),
        temperature=_TEMPERATURE,
    )
    try:
        payload = _parse_payload(raw)
    except AssemblyError as exc:
        logger.warning("Orientation assembly failed for task %r: %s", task_title, exc)
        outcome = OrientationOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            chunks_collapsed=collapsed,
            notes=[str(exc)],
        )
        yield progress.warning("The generated packet could not be read")
        yield progress.done("No orientation could be assembled", _dump(outcome))
        return outcome

    yield progress.stage("grounding", "Checking every section cites its source")
    sections, dropped = _resolve_sections(payload, chunks)
    for section in sections:
        yield progress.item(
            section.model_dump(mode="json"),
            f"{_STEP_LABEL[section.step]}: {section.title}",
        )
    # A dropped section is a gap the watcher should see, not a silent omission --
    # unless every section went, which the branch below states more directly.
    if sections and dropped:
        yield progress.warning(f"Dropped {dropped} section(s) with no source")

    if not sections:
        outcome = OrientationOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            chunks_collapsed=collapsed,
            sections_dropped=dropped,
            notes=["no grounded sections in the assembled packet"],
        )
        yield progress.warning("Every section was dropped for lack of a source")
        yield progress.done("No orientation could be assembled", _dump(outcome))
        return outcome

    notes: list[str] = []
    if collapsed:
        notes.append(f"collapsed {collapsed} redundant source chunk(s)")
    if dropped:
        notes.append(f"dropped {dropped} ungrounded or duplicate section(s)")

    outcome = OrientationOutcome(
        status="assembled",
        packet=OrientationPacket(
            task_title=task_title,
            summary=payload.summary.strip(),
            sections=sections,
            sources=_sources_drawn_on(sections, chunks),
        ),
        provenance=OrientationProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
            notes=notes,
        ),
        chunks_retrieved=len(chunks),
        chunks_collapsed=collapsed,
        sections_dropped=dropped,
        notes=notes,
    )
    yield progress.done("Orientation ready", _dump(outcome))
    return outcome


def _dump(outcome: OrientationOutcome) -> dict[str, object]:
    """The outcome as a JSON-safe dict for a ``done`` event's ``result``."""
    return outcome.model_dump(mode="json")


def assemble_orientation(
    llm: LLMClient,
    store: VectorStore,
    *,
    task_title: str,
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
            task_body=task_body,
            labels=labels,
            touched_paths=touched_paths,
            last_fingerprint=last_fingerprint,
        )
    )
