"""AI-authoring of starter-work pool candidates from open tracker issues.

A batch, re-runnable job -- offline, not on the hire's request path -- that
mines the ingested corpus's open issues for safely-scoped starter tasks and
proposes them as a candidate pool. Idempotent through
:func:`onboarding.corpus.corpus_fingerprint`.

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

import json
import logging
from collections.abc import Generator
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.corpus import corpus_fingerprint
from onboarding.models import CitationRef
from onboarding.progress import ProgressEvent, ProgressStream, drain
from rag.types import Chunk
from store.base import VectorStore

ProposalStatus = Literal["proposed", "unchanged", "skipped"]

logger = logging.getLogger(__name__)

_MAX_CHUNKS_PER_ISSUE = 20

# How many issues one mining run may judge.
#
# Every candidate goes into a *single* prompt, so without a cap the prompt
# grows linearly with the corpus: ingesting a whole organisation once produced
# one call slow enough to block the person who triggered it. A crawl is not a
# moment anybody is watching, and an unbounded loop is a bill nobody authorised.
#
# It also protects the judgement: a model asked to rank hundreds of issues in one
# prompt reasons about each of them worse than one asked about twenty-five.
#
# What does not fit is not dropped silently -- see ``notes`` on the outcome.
_MAX_CANDIDATES_PER_RUN = 25


class GenerationError(Exception):
    """Raised when the LLM output for a starter-work proposal can't be parsed."""


# --- domain models -------------------------------------------------------------


class StarterTaskCandidate(BaseModel):
    """One open issue considered for the starter-work pool."""

    source_id: str
    title: str
    text: str
    source_url: str | None = None
    labels: list[str] = Field(default_factory=list[str])


class ProposedStarterTask(BaseModel):
    """A candidate starter task grounded in one open issue, for PM curation."""

    source_id: str
    title: str
    summary: str = ""
    competency_keys: list[str] = Field(default_factory=list[str])
    rationale: str = ""
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class StarterWorkProvenance(BaseModel):
    """Why a mining run looks the way it does; mirrors ``GraphProvenance``."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class StarterWorkOutcome(BaseModel):
    """Result of one starter-work mining run."""

    status: ProposalStatus
    tasks: list[ProposedStarterTask] = Field(default_factory=list[ProposedStarterTask])
    provenance: StarterWorkProvenance | None = None
    candidates_considered: int = 0
    notes: list[str] = Field(default_factory=list[str])


# --- LLM payload -----------------------------------------------------------------


class _GenTask(BaseModel):
    source_id: str = ""
    safely_scoped: bool = False
    summary: str = ""
    competency_keys: list[str] = Field(default_factory=list[str])
    rationale: str = ""


class _GenPayload(BaseModel):
    tasks: list[_GenTask] = Field(default_factory=list[_GenTask])


# --- prompt / parsing --------------------------------------------------------------


def _derive_title(text: str, fallback: str) -> str:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip() or fallback
    return fallback


def _issue_block(candidate: StarterTaskCandidate) -> str:
    label_note = f" [labels: {', '.join(candidate.labels)}]" if candidate.labels else ""
    return f"[{candidate.source_id}]{label_note}\n{candidate.text}"


def _build_prompt(
    candidates: list[StarterTaskCandidate], known_competency_keys: list[str]
) -> list[Message]:
    issues = "\n\n---\n\n".join(_issue_block(c) for c in candidates)
    known = ""
    if known_competency_keys:
        known = (
            "\nKnown competency keys you may tag a task with (do not invent new "
            f"ones): {', '.join(sorted(known_competency_keys))}\n"
        )
    system = (
        "You review a software team's open GitHub issues to build a curated "
        "pool of starter tasks for new hires. You are given each issue's full "
        "text, prefixed with its source id in square brackets.\n\n"
        "For each issue, decide whether it is safely scoped for a first-time "
        "contributor:\n"
        "- 'safely_scoped' is true only if the issue has a small, well-defined "
        "surface (touches one module/area, not a cross-cutting change), a "
        "clear description of what 'done' looks like, and no dependency on "
        "context a new hire wouldn't have.\n"
        "- Reject (safely_scoped=false) anything vague, large, architectural, "
        "or that reads as blocked/needs-discussion.\n"
        "- 'summary' is a one or two sentence restatement of the task for a "
        "hire browsing the pool -- do not just repeat the issue title.\n"
        "- 'competency_keys' tags the skills/concepts this task would "
        "exercise, chosen only from the known keys below when a list is "
        "given.\n"
        "- 'rationale' is one sentence on why the task is (or isn't) safely "
        "scoped.\n"
        f"{known}\n"
        "Return STRICT JSON only (no prose, no markdown fences), one entry "
        "per issue you were given, correlated by 'source_id':\n"
        '{"tasks": [{"source_id": str, "safely_scoped": bool, "summary": str, '
        '"competency_keys": [str], "rationale": str}]}'
    )
    user = f"Open issues:\n\n{issues}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _GenPayload:
    try:
        return _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid starter-work output: {exc}") from exc


# --- candidate sourcing ------------------------------------------------------------


def _issue_text(chunks: list[Chunk]) -> str:
    ordered = sorted(chunks, key=lambda c: c.position or 0)
    return "\n".join(c.text for c in ordered)


def _eligible_artifacts(
    metadata_store: IngestionMetadataStore,
    *,
    exclude_source_ids: set[str],
) -> list[ArtifactRecord]:
    """Every artifact that could become a starter task, in a stable order.

    Metadata only: no chunk reads, so capping happens before the expensive part.
    Sorted by ``source_id`` because **a stable order is what makes the cap fair**
    -- an arbitrary one would re-judge the same issues every run while others
    were never reached at all.
    """
    eligible: list[ArtifactRecord] = []
    for artifact in metadata_store.list_artifacts(status="completed"):
        if artifact.artifact_type != "ISSUE":
            continue
        if (artifact.state or "").upper() != "OPEN":
            continue
        # Only a *definite* True withholds it. Starter work is work a hire can
        # take, and an issue somebody else is already on is not available however
        # open it is -- a Jira board assigns its in-progress tickets, so without
        # this the pool offered new hires work other people were doing. ``None``
        # means the connector cannot tell (GitHub's assignees are not ingested)
        # and must never be read as "nobody", or engineering behaviour would
        # change on a fact nobody established.
        if artifact.has_assignee is True:
            continue
        if artifact.source_id is None or artifact.source_id in exclude_source_ids:
            continue

        eligible.append(artifact)

    return sorted(eligible, key=lambda a: a.source_id or "")


def _load_candidates(
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    *,
    exclude_source_ids: set[str],
    limit: int = _MAX_CANDIDATES_PER_RUN,
) -> tuple[list[StarterTaskCandidate], int, int]:
    """The issues this run will judge, and how many eligible ones there were.

    Chunks are read only for the capped slice, so an organisation-sized corpus
    costs one metadata scan rather than a chunk lookup per issue.

    Returns ``(candidates, eligible_total, skipped_no_chunks)``. ``eligible_total``
    is what the caller reports as left over: a pool silently missing most of the
    corpus reads as "there is no good first work here", which is a different claim
    entirely.
    """
    eligible = _eligible_artifacts(
        metadata_store, exclude_source_ids=exclude_source_ids
    )

    candidates: list[StarterTaskCandidate] = []
    skipped_no_chunks = 0
    for artifact in eligible:
        if len(candidates) >= limit:
            break
        chunks = store.list_chunks_by_artifact(artifact.id, limit=_MAX_CHUNKS_PER_ISSUE)
        if not chunks:
            skipped_no_chunks += 1
            continue

        text = _issue_text(chunks)
        if not text.strip():
            skipped_no_chunks += 1
            continue

        candidates.append(
            StarterTaskCandidate(
                source_id=artifact.source_id or "",
                title=_derive_title(text, artifact.filename),
                text=text,
                source_url=artifact.source_url,
                labels=artifact.labels,
            )
        )
    return candidates, len(eligible), skipped_no_chunks


# --- job -----------------------------------------------------------------------


def stream_starter_work_pool(
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    *,
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
    progress = ProgressStream("starter_work")
    fingerprint = corpus_fingerprint(store)
    if last_fingerprint is not None and last_fingerprint == fingerprint:
        outcome = StarterWorkOutcome(
            status="unchanged", notes=["corpus unchanged since last mining run"]
        )
        yield progress.done("Nothing changed since the last mining run", _dump(outcome))
        return outcome

    if store.count() == 0:
        outcome = StarterWorkOutcome(status="skipped", notes=["corpus is empty"])
        yield progress.warning("The project has no indexed material yet")
        yield progress.done("No starter tasks could be mined", _dump(outcome))
        return outcome

    yield progress.stage("retrieving", "Collecting open issues from the corpus")
    candidates, eligible_total, skipped_no_chunks = _load_candidates(
        store, metadata_store, exclude_source_ids=set(active_source_ids or [])
    )
    if not candidates:
        outcome = StarterWorkOutcome(
            status="skipped", notes=["no open, unpooled issues found"]
        )
        yield progress.warning("No open, unpooled issues to mine")
        yield progress.done("No starter tasks could be mined", _dump(outcome))
        return outcome

    # What did not fit is counted, never silently dropped: the next run reaches it,
    # and until then a PM can tell "capped" from "the corpus holds nothing else".
    deferred = max(eligible_total - len(candidates) - skipped_no_chunks, 0)
    cap_notes = (
        [
            f"{deferred} more eligible issue(s) were not judged this run "
            f"(cap is {_MAX_CANDIDATES_PER_RUN}); the next run picks them up"
        ]
        if deferred
        else []
    )
    if skipped_no_chunks:
        cap_notes.append(f"{skipped_no_chunks} issue(s) had no chunks and were skipped")
    if deferred:
        yield progress.warning(
            f"Judging {len(candidates)} of {eligible_total} open issues this run; "
            f"{deferred} wait for the next"
        )

    yield progress.stage(
        "grounding", f"Judging {len(candidates)} open issue(s) for scope safety"
    )
    raw = llm.generate(_build_prompt(candidates, active_competency_keys or []))
    try:
        payload = _parse_payload(raw)
    except GenerationError as exc:
        logger.warning("Starter-work mining generation failed: %s", exc)
        outcome = StarterWorkOutcome(
            status="skipped",
            candidates_considered=len(candidates),
            notes=[str(exc), *cap_notes],
        )
        yield progress.warning("The mined tasks could not be read")
        yield progress.done("No starter tasks could be mined", _dump(outcome))
        return outcome

    candidates_by_id = {c.source_id: c for c in candidates}
    known_keys = set(active_competency_keys) if active_competency_keys else None

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

    if not tasks:
        outcome = StarterWorkOutcome(
            status="skipped",
            candidates_considered=len(candidates),
            notes=["no candidate judged safely scoped", *cap_notes],
        )
        yield progress.done("No issue was judged safely scoped", _dump(outcome))
        return outcome

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
            active_source_ids=active_source_ids,
            active_competency_keys=active_competency_keys,
            last_fingerprint=last_fingerprint,
        )
    )
