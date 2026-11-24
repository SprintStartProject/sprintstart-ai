"""AI-authoring of starter-work pool candidates from open tracker issues.

A batch, re-runnable job -- offline, not on the hire's request path -- that
mines the ingested corpus's open issues for safely-scoped starter tasks and
proposes them as a candidate pool. Idempotent through
:func:`assembly.fingerprint.corpus_fingerprint`.

Candidate sourcing is deterministic, not LLM-driven: only ``ISSUE``
artifacts with ``state == "OPEN"`` are ever considered, so closed issues are
excluded before the LLM sees them rather than relying on it to notice.

Which tracker the issue came from is not asked, and must not become a
condition. An issue is a candidate because it is open, not because of where it
lives -- that is what lets a project whose tracker is Jira mine a pool at all. A
test pins it, because "only GitHub issues" is the assumption most likely to be
re-added by somebody reading this file rather than the filter.

The LLM's role is judgment, not filtering: given each open issue's own
text, it assesses whether the issue is *safely scoped* for a new hire (small,
clear acceptance criteria, no cross-cutting blast radius) and tags the
competencies it exercises. There is no free-form retrieval step here, since the
source of truth for each candidate is the issue itself, not a synthesized
answer -- grounding is
"this proposal came from this issue's own chunks", not citation resolution
against a broader corpus.

"TODOs in code" and "small-surface modules" (the other candidate sources
issue #5 asks for) are out of scope for this slice: unlike an issue, a TODO
comment or a small module has no ingested owner or acceptance criteria to
ground a proposal in, so mining them needs its own grounding strategy.
Deferred rather than guessed at.
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime

from assembly.fingerprint import fingerprint_gate
from assembly.models import CitationRef
from assembly.progress import ProgressEvent, ProgressStream, drain
from assembly.starter_work.draft import (
    MAX_CANDIDATES_PER_RUN,
    GenPayload,
    build_prompt,
    load_candidates,
    mining_fingerprint_material,
    parse_payload,
)
from assembly.starter_work.models import (
    GenerationError,
    ProposedStarterTask,
    StarterTaskCandidate,
    StarterWorkOutcome,
    StarterWorkProvenance,
)
from llm.base import LLMClient
from store.base import VectorStore
from store.metadata_store import IngestionMetadataStore

logger = logging.getLogger(__name__)

_DONE_EMPTY = "No starter tasks could be mined"


@dataclass(frozen=True)
class _Scope:
    """What the corpus fingerprint of a mining run is computed over."""

    project_ids: frozenset[str]
    extra_material: list[str]
    last_fingerprint: str | None


def _gate(
    progress: ProgressStream, store: VectorStore, scope: _Scope
) -> tuple[str | None, list[ProgressEvent], StarterWorkOutcome | None]:
    """Answer from the fingerprint alone when the corpus is empty or unchanged."""
    return fingerprint_gate(
        progress,
        store,
        scope.last_fingerprint,
        make_unchanged=lambda: StarterWorkOutcome(
            status="unchanged", notes=["corpus unchanged since last mining run"]
        ),
        make_empty=lambda: StarterWorkOutcome(
            status="skipped", notes=["corpus is empty"]
        ),
        unchanged_label="Nothing changed since the last mining run",
        empty_warning_label="The project has no indexed material yet",
        empty_done_label=_DONE_EMPTY,
        extra_fingerprint_material=scope.extra_material,
        project_ids=scope.project_ids,
    )


def _skipped(considered: int, *notes: str) -> StarterWorkOutcome:
    return StarterWorkOutcome(
        status="skipped", candidates_considered=considered, notes=list(notes)
    )


def _end(
    progress: ProgressStream,
    outcome: StarterWorkOutcome,
    done: str,
    warning: str | None = None,
) -> Generator[ProgressEvent, None, StarterWorkOutcome]:
    """Finish a run without a proposed pool: say why, then emit the outcome."""
    if warning is not None:
        yield progress.warning(warning)
    yield progress.done(done, _dump(outcome))
    return outcome


def _judge(
    progress: ProgressStream,
    llm: LLMClient,
    candidates: list[StarterTaskCandidate],
    competency_keys: list[str],
) -> Generator[ProgressEvent, None, GenPayload | GenerationError]:
    yield progress.stage(
        "grounding", f"Judging {len(candidates)} open issue(s) for scope safety"
    )
    raw = llm.generate(build_prompt(candidates, competency_keys))
    try:
        return parse_payload(raw)
    except GenerationError as exc:
        logger.warning("Starter-work mining generation failed: %s", exc)
        return exc


def _announce_cap(
    progress: ProgressStream, judged: int, eligible_total: int, skipped_no_chunks: int
) -> Generator[ProgressEvent, None, list[str]]:
    """Say what this run does not judge, and return the notes that record it.

    What did not fit is counted, never silently dropped: the next run reaches it,
    and until then a PM can tell "capped" from "the corpus holds nothing else".
    """
    deferred = max(eligible_total - judged - skipped_no_chunks, 0)
    notes = (
        [
            f"{deferred} more eligible issue(s) were not judged this run "
            f"(cap is {MAX_CANDIDATES_PER_RUN}); the next run picks them up"
        ]
        if deferred
        else []
    )
    if skipped_no_chunks:
        notes.append(f"{skipped_no_chunks} issue(s) had no chunks and were skipped")
    if deferred:
        yield progress.warning(
            f"Judging {judged} of {eligible_total} open issues this run; "
            f"{deferred} wait for the next"
        )
    return notes


def _vet(
    progress: ProgressStream,
    payload: GenPayload,
    candidates: list[StarterTaskCandidate],
    known_keys: set[str] | None,
) -> Generator[ProgressEvent, None, list[ProposedStarterTask]]:
    """Keep the tasks judged safely scoped, each announced the instant it clears."""
    candidates_by_id = {c.source_id: c for c in candidates}
    tasks: list[ProposedStarterTask] = []
    seen: set[str] = set()
    for item in payload.tasks:
        if not item.safely_scoped:
            continue
        candidate = candidates_by_id.get(item.source_id)
        if candidate is None or candidate.source_id in seen:
            continue
        seen.add(candidate.source_id)

        keys = [
            k for k in item.competency_keys if known_keys is None or k in known_keys
        ]
        citation = CitationRef(
            filename=candidate.title,
            chunk_id=candidate.source_id,
            source_url=candidate.source_url,
        )
        task = ProposedStarterTask(
            source_id=candidate.source_id,
            title=candidate.title,
            summary=item.summary or candidate.title,
            competency_keys=keys,
            rationale=item.rationale,
            citations=[citation],
        )
        tasks.append(task)
        yield progress.item(task.model_dump(mode="json"), f"Task: {task.title}")
    return tasks


def stream_starter_work_pool(
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    *,
    project_ids: frozenset[str],
    active_source_ids: list[str] | None = None,
    active_competency_keys: list[str] | None = None,
    last_fingerprint: str | None = None,
) -> Generator[ProgressEvent, None, StarterWorkOutcome]:
    """Mine the pool, yielding live progress and returning the final outcome.

    This is the single implementation: :func:`generate_starter_work_pool` drives
    it to completion for the non-streaming path, and the streaming route relays
    its events — so a PM watching the pool fill sees exactly what the batch call
    would have produced. Each task is emitted as an ``item`` the instant it clears
    the scope-safety judgement, so nothing unsafe is ever shown, and an ``item``
    is a promise the task is in the persisted pool.
    """
    extra_material = mining_fingerprint_material(
        metadata_store, active_source_ids, project_ids=project_ids
    )
    progress = ProgressStream("starter_work")
    scope = _Scope(project_ids, extra_material, last_fingerprint)
    fingerprint, early_events, early_outcome = _gate(progress, store, scope)
    if early_outcome is not None:
        yield from early_events
        return early_outcome

    yield progress.stage("retrieving", "Collecting open issues from the corpus")
    candidates, eligible_total, skipped_no_chunks = load_candidates(
        store,
        metadata_store,
        exclude_source_ids=set(active_source_ids or []),
        project_ids=project_ids,
    )
    if not candidates:
        outcome = _skipped(0, "no open, unpooled issues found")
        return (
            yield from _end(
                progress, outcome, _DONE_EMPTY, "No open, unpooled issues to mine"
            )
        )

    cap_notes = yield from _announce_cap(
        progress, len(candidates), eligible_total, skipped_no_chunks
    )

    payload = yield from _judge(progress, llm, candidates, active_competency_keys or [])
    if isinstance(payload, GenerationError):
        outcome = _skipped(len(candidates), str(payload), *cap_notes)
        return (
            yield from _end(
                progress, outcome, _DONE_EMPTY, "The mined tasks could not be read"
            )
        )

    known_keys = set(active_competency_keys) if active_competency_keys else None
    tasks = yield from _vet(progress, payload, candidates, known_keys)
    if not tasks:
        outcome = _skipped(
            len(candidates), "no candidate judged safely scoped", *cap_notes
        )
        return (yield from _end(progress, outcome, "No issue was judged safely scoped"))

    outcome = StarterWorkOutcome(
        status="proposed",
        tasks=tasks,
        provenance=StarterWorkProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
        ),
        candidates_considered=len(candidates),
        notes=cap_notes,
    )
    yield progress.done(f"Proposed {len(tasks)} starter task(s)", _dump(outcome))
    return outcome


def _dump(outcome: StarterWorkOutcome) -> dict[str, object]:
    """The outcome as a JSON-safe dict for a ``done`` event's ``result``."""
    return outcome.model_dump(mode="json")


def generate_starter_work_pool(
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    *,
    project_ids: frozenset[str],
    active_source_ids: list[str] | None = None,
    active_competency_keys: list[str] | None = None,
    last_fingerprint: str | None = None,
) -> StarterWorkOutcome:
    """Propose safely-scoped starter tasks from open tracker issues.

    ``active_source_ids`` are issues already in the backend's starter-work
    pool -- drives dedup, an issue already pooled is never mined again.
    ``active_competency_keys`` are the backend's live competency keys; a proposed
    task's competency tags are grounded against this set (dropped, not invented,
    when the tag falls outside it) --
    when no keys are supplied there is nothing to validate against, so tags
    are kept as the LLM proposed them. ``last_fingerprint`` carries corpus-wide
    idempotency: an unchanged corpus produces no new proposals.

    Backed by :func:`stream_starter_work_pool` so the non-streaming and streaming
    paths are the same computation.
    """
    return drain(
        stream_starter_work_pool(
            llm,
            store,
            metadata_store,
            project_ids=project_ids,
            active_source_ids=active_source_ids,
            active_competency_keys=active_competency_keys,
            last_fingerprint=last_fingerprint,
        )
    )
