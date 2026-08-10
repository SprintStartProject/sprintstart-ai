"""Proposal of a shared, reviewable module for one competency.

A batch, authoring-time job — never on a hire's request path. It reuses the same
retrieval layer (:func:`rag.hybrid.hybrid_retrieve`) and idempotency mechanism
(:func:`onboarding.corpus.corpus_fingerprint`) as the rest of the offline
generation jobs.

Two properties are load-bearing and easy to lose:

* **Stateless with respect to any individual hire.** Nothing about a person
  enters this function — no profile, no ledger, no transcript. One competency
  yields one module that everybody reads. Personalization is *which* nodes are
  on a path and what a hire can test out of, not a private paraphrase each.
* **Grounding is per page, not per module.** A page making claims about the
  codebase that cites nothing is dropped, even when the rest of the module is
  well grounded — otherwise one hallucinated page rides along on the strength of
  its neighbours.
"""

import json
import logging
from collections.abc import Generator
from datetime import UTC, datetime
from typing import get_args

from pydantic import BaseModel, Field, ValidationError

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.corpus import corpus_fingerprint
from onboarding.module_models import (
    GROUNDED_PAGE_KINDS,
    ModuleLevel,
    ModuleOutcome,
    ModulePageKind,
    ModuleProvenance,
    ProposedModule,
    ProposedModulePage,
    ProposedModuleVerification,
)
from onboarding.progress import ProgressEvent, ProgressStream, drain
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 16
_MIN_SCORE = 0.3

# Re-running against an unchanged corpus must give materially the same module:
# a PM's edits should never be churned by sampling noise. Nothing else in this
# job is stochastic, so the temperature is the whole guarantee.
_TEMPERATURE = 0.0

# Derived from the Literal rather than restated, so the prompt, the validation
# and the backend's enum cannot drift apart.
_ALLOWED_KINDS: tuple[str, ...] = get_args(ModulePageKind)

# How deep the prose goes, and how much scaffolding the module needs. An expert
# node is not four pages of basics, and a beginner node is not three terse
# paragraphs that assume the reader already knows the shape of the system.
_SHAPE_BY_LEVEL: dict[str, str] = {
    "beginner": (
        "Explain from first principles; assume little prior context and spell "
        "out *why*, not just *how*. Include a CONTEXT page and a WALKTHROUGH: "
        "somebody at this level needs to see the thing done once. Aim for 4-5 "
        "pages."
    ),
    "intermediate": (
        "Use a balanced level of detail; assume basic familiarity with the "
        "stack but not this codebase. Aim for 3-4 pages."
    ),
    "advanced": (
        "Keep it concise and focus on this codebase's specific choices and "
        "trade-offs; assume strong general context. Skip introductory framing. "
        "Aim for 2-3 pages."
    ),
    "expert": (
        "Be terse. Cover only what is non-obvious or specific to this "
        "codebase's conventions; an expert does not need the concept "
        "explained. Aim for 1-2 pages, and prefer a TASK over more prose."
    ),
}

_KIND_GUIDE = (
    "  CONTEXT     - why this competency matters *in this codebase*\n"
    "  LESSON      - how it actually works here\n"
    "  WALKTHROUGH - one real example traced end to end\n"
    "  TASK        - something to do, hands on. Not graded.\n"
    "  RESOURCE    - pointers to the real material worth reading\n"
    "  CHECK       - a quick self-check to try before the graded one\n"
)


class GenerationError(Exception):
    """Raised when the LLM output for a module cannot be parsed/validated."""


class _GenPage(BaseModel):
    kind: str = ""
    title: str = ""
    body: str = ""
    chunk_ids: list[str] = Field(default_factory=list[str])


class _GenVerification(BaseModel):
    prompt: str = ""
    rubric: str = ""


class _GenPayload(BaseModel):
    title: str = ""
    summary: str = ""
    pages: list[_GenPage] = Field(default_factory=list[_GenPage])
    verification: _GenVerification | None = None


def _evidence_line(chunk: ScoredChunk) -> str:
    meta = chunk.artifact_type or "FILE"
    if chunk.language:
        meta += f"/{chunk.language}"
    return f"  [{chunk.id}] ({chunk.filename} | {meta}) {chunk.text}"


def _build_prompt(
    competency_label: str,
    competency_description: str,
    level: ModuleLevel,
    chunks: list[ScoredChunk],
) -> list[Message]:
    evidence = "\n".join(_evidence_line(c) for c in chunks)
    kinds = ", ".join(_ALLOWED_KINDS)

    system = (
        "You write one onboarding module teaching a single competency for a "
        "software team, from that team's own codebase and docs. The module is "
        "read by everybody who needs this competency, and a project manager "
        "reviews and edits it before anyone sees it -- so write it for a team, "
        "not for one person.\n\n"
        "A module is an ordered list of typed pages. Use only these kinds:\n"
        f"{_KIND_GUIDE}\n"
        "Rules:\n"
        "1. Order the pages the way somebody should read them.\n"
        "2. Teach *why* before *how*. Every factual claim about this codebase "
        "must come from the evidence below; never invent a detail it does not "
        "support.\n"
        "3. Each CONTEXT / LESSON / WALKTHROUGH / RESOURCE page MUST list the "
        "evidence chunk ids *that page* draws on. Per page, not per module: a "
        "page that cites nothing is dropped, however good the rest is.\n"
        "4. TASK and CHECK pages need no chunk ids -- they are exercises built "
        "on the pages above them, not claims of their own.\n"
        "5. Finish with a graded check: a question that cannot be answered by "
        "somebody who did not read the module, plus a rubric describing what a "
        "passing answer contains. It is not a page; return it separately.\n\n"
        f"Target level: {level}. {_SHAPE_BY_LEVEL[level]}\n\n"
        "Return STRICT JSON only (no prose, no markdown fences):\n"
        '{"title": str, "summary": str, "pages": [{"kind": '
        f"{kinds.replace(', ', '|')}"
        ', "title": str, "body": str (markdown), "chunk_ids": [str]}], '
        '"verification": {"prompt": str, "rubric": str}}'
    )
    user = (
        f"Competency: {competency_label}\n"
        f"Description: {competency_description or '(none)'}\n\n"
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
        raise GenerationError(f"invalid module output: {exc}") from exc


def _resolve_pages(
    payload: _GenPayload, chunks: list[ScoredChunk]
) -> tuple[list[ProposedModulePage], int]:
    """Keeps the pages that are usable; returns them with the number dropped.

    A page is dropped when its kind is not one the backend can store, when it
    has no body, or when it makes claims about the codebase without citing
    anything. Dropping the page rather than the module is the point: one
    ungrounded page should not cost a PM the four good ones next to it.
    """
    chunks_by_id = {c.id: c for c in chunks}
    kept: list[ProposedModulePage] = []
    dropped = 0

    for item in payload.pages:
        kind = item.kind.strip().upper()
        if kind not in _ALLOWED_KINDS:
            logger.info("Dropped module page of unknown kind %r", item.kind)
            dropped += 1
            continue
        if not item.body.strip():
            dropped += 1
            continue

        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if kind in GROUNDED_PAGE_KINDS and not citations:
            logger.info("Dropped ungrounded %s page %r", kind, item.title)
            dropped += 1
            continue

        kept.append(
            ProposedModulePage(
                kind=kind,  # type: ignore[arg-type]
                title=item.title.strip() or kind.title(),
                body=item.body,
                citations=citations,
            )
        )

    return kept, dropped


def _resolve_verification(payload: _GenPayload) -> ProposedModuleVerification | None:
    check = payload.verification
    if check is None or not check.prompt.strip():
        return None
    # KNOWLEDGE is the only type this job can propose: EXACT needs a canonical
    # answer, ARTIFACT needs a repository connection, and ATTEST is not a check
    # at all. A PM raises the rigor when the module deserves it.
    return ProposedModuleVerification(
        type="KNOWLEDGE",
        prompt=check.prompt.strip(),
        rubric=check.rubric.strip() or None,
    )


def stream_module(
    llm: LLMClient,
    store: VectorStore,
    *,
    competency_key: str,
    competency_label: str,
    competency_description: str = "",
    level: ModuleLevel = "beginner",
    last_fingerprint: str | None = None,
) -> Generator[ProgressEvent, None, ModuleOutcome]:
    """Propose a module, yielding live progress and returning the final outcome.

    The single implementation: :func:`propose_module` drives it to completion for
    the non-streaming path, and the streaming route relays its events — so the
    module a PM watches assemble is exactly the module the batch call produces.

    Pages are emitted as ``item`` events only after they clear the per-page
    grounding gate (:data:`GROUNDED_PAGE_KINDS`), so an ungrounded page never
    appears live; a dropped page is reported as a ``warning``.
    """
    progress = ProgressStream("module")
    fingerprint = corpus_fingerprint(store)

    if last_fingerprint is not None and last_fingerprint == fingerprint:
        outcome = ModuleOutcome(
            status="unchanged", notes=["corpus unchanged since last proposal run"]
        )
        yield progress.done(
            "Nothing changed — the current module still stands", _dump(outcome)
        )
        return outcome

    if store.count() == 0:
        outcome = ModuleOutcome(status="skipped", notes=["corpus is empty"])
        yield progress.warning("The project has no indexed material yet")
        yield progress.done("No module could be proposed", _dump(outcome))
        return outcome

    yield progress.stage(
        "retrieving", f"Searching the project for “{competency_label}”"
    )
    query = f"{competency_label}: {competency_description}".strip(": ")
    chunks = hybrid_retrieve(
        question=query,
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=BM25IndexCache(),
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
    )
    if not chunks:
        outcome = ModuleOutcome(
            status="skipped", notes=["no grounding evidence retrieved"]
        )
        yield progress.warning("Nothing in the project grounds this competency")
        yield progress.done("No module could be proposed", _dump(outcome))
        return outcome

    yield progress.stage(
        "generating", f"Writing the module from {len(chunks)} source(s)"
    )
    raw = llm.generate(
        _build_prompt(competency_label, competency_description, level, chunks),
        temperature=_TEMPERATURE,
    )
    try:
        payload = _parse_payload(raw)
    except GenerationError as exc:
        logger.warning(
            "Module proposal failed for competency %r: %s", competency_key, exc
        )
        outcome = ModuleOutcome(
            status="skipped", chunks_retrieved=len(chunks), notes=[str(exc)]
        )
        yield progress.warning("The generated module could not be read")
        yield progress.done("No module could be proposed", _dump(outcome))
        return outcome

    yield progress.stage("grounding", "Checking every page cites its source")
    pages, dropped = _resolve_pages(payload, chunks)
    for page in pages:
        yield progress.item(page.model_dump(mode="json"), f"{page.kind}: {page.title}")
    # A page dropped for lack of a source is a gap the watcher should see, unless
    # every page went -- which the branch below states more directly.
    if pages and dropped:
        yield progress.warning(f"Dropped {dropped} page(s) with no source")

    if not pages:
        outcome = ModuleOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            pages_dropped=dropped,
            notes=["no grounded pages in the generated module"],
        )
        yield progress.warning("Every page was dropped for lack of a source")
        yield progress.done("No module could be proposed", _dump(outcome))
        return outcome

    notes = [f"dropped {dropped} ungrounded or unusable page(s)"] if dropped else []
    outcome = ModuleOutcome(
        status="proposed",
        module=ProposedModule(
            competency_key=competency_key,
            level=level,
            title=payload.title.strip() or competency_label,
            summary=payload.summary.strip(),
            pages=pages,
            verification=_resolve_verification(payload),
        ),
        provenance=ModuleProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
            notes=notes,
        ),
        chunks_retrieved=len(chunks),
        pages_dropped=dropped,
        notes=notes,
    )
    yield progress.done("Module ready for review", _dump(outcome))
    return outcome


def _dump(outcome: ModuleOutcome) -> dict[str, object]:
    """The outcome as a JSON-safe dict for a ``done`` event's ``result``."""
    return outcome.model_dump(mode="json")


def propose_module(
    llm: LLMClient,
    store: VectorStore,
    *,
    competency_key: str,
    competency_label: str,
    competency_description: str = "",
    level: ModuleLevel = "beginner",
    last_fingerprint: str | None = None,
) -> ModuleOutcome:
    """Propose a shared module for one competency at one target level.

    ``last_fingerprint`` is whatever fingerprint the caller recorded from the
    previous run for this competency: idempotency is per module, not
    corpus-wide, since the backend proposes modules one node at a time.

    Takes no argument describing any individual hire, and that is intentional --
    see the module docstring. Backed by :func:`stream_module` so the streaming and
    non-streaming paths are the same computation.
    """
    return drain(
        stream_module(
            llm,
            store,
            competency_key=competency_key,
            competency_label=competency_label,
            competency_description=competency_description,
            level=level,
            last_fingerprint=last_fingerprint,
        )
    )
