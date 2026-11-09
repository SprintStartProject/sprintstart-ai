"""What the model judges when mining starter work, and the candidates it judges.

Choosing the open issues to put in front of the model, the prompt, and reading its
judgement: everything that depends on the corpus's issues and the model's output
rather than on the order of the stages in assembly.starter_work.job.
"""

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from assembly.starter_work.models import (
    GenerationError,
    StarterTaskCandidate,
)
from corpus.types import Chunk
from llm.base import Message
from llm.parsing import extract_json_object
from store.base import VectorStore
from store.metadata_store import ArtifactRecord, IngestionMetadataStore

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
MAX_CANDIDATES_PER_RUN = 25


class _GenTask(BaseModel):
    source_id: str = ""
    safely_scoped: bool = False
    summary: str = ""
    competency_keys: list[str] = Field(default_factory=list[str])
    rationale: str = ""


class GenPayload(BaseModel):
    tasks: list[_GenTask] = Field(default_factory=list[_GenTask])


def _derive_title(text: str, fallback: str) -> str:
    first_line = text.splitlines()[0].strip() if text else ""
    if first_line.startswith("#"):
        return first_line.lstrip("#").strip() or fallback
    return fallback


def _issue_block(candidate: StarterTaskCandidate) -> str:
    label_note = f" [labels: {', '.join(candidate.labels)}]" if candidate.labels else ""
    return f"[{candidate.source_id}]{label_note}\n{candidate.text}"


def build_prompt(
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


def parse_payload(raw: str) -> GenPayload:
    try:
        return GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid starter-work output: {exc}") from exc


def _issue_text(chunks: list[Chunk]) -> str:
    ordered = sorted(chunks, key=lambda c: c.position or 0)
    return "\n".join(c.text for c in ordered)


def _eligible_artifacts(
    metadata_store: IngestionMetadataStore,
    *,
    exclude_source_ids: set[str],
    project_ids: frozenset[str],
) -> list[ArtifactRecord]:
    """Every artifact that could become a starter task, in a stable order.

    Metadata only: no chunk reads, so capping happens before the expensive part.
    Sorted by ``source_id`` because **a stable order is what makes the cap fair**
    -- an arbitrary one would re-judge the same issues every run while others
    were never reached at all.
    """
    eligible: list[ArtifactRecord] = []
    for artifact in metadata_store.list_artifacts(
        status="completed", project_ids=list(project_ids)
    ):
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


def load_candidates(
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    *,
    exclude_source_ids: set[str],
    project_ids: frozenset[str],
    limit: int = MAX_CANDIDATES_PER_RUN,
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
        metadata_store, exclude_source_ids=exclude_source_ids, project_ids=project_ids
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


def mining_fingerprint_material(
    metadata_store: IngestionMetadataStore,
    active_source_ids: list[str] | None = None,
    *,
    project_ids: frozenset[str],
) -> list[str]:
    """Derive fingerprint material from issue metadata and the active pool.

    Includes the issue tracker metadata (state, has_assignee, labels) that
    drives starter-work eligibility, plus the active pool source ids, so an
    issue being closed, assigned, or added to the pool invalidates the cache
    even if the chunk texts did not change.
    """
    parts: list[str] = []
    for artifact in metadata_store.list_artifacts(
        status="completed", project_ids=list(project_ids)
    ):
        if artifact.artifact_type == "ISSUE":
            labels_str = ",".join(sorted(artifact.labels))
            parts.append(
                f"issue:{artifact.id}:{artifact.state}:{artifact.has_assignee}:{labels_str}"
            )
    for source_id in sorted(active_source_ids or []):
        parts.append(f"pool:{source_id}")
    return parts
