"""AI-authoring of onboarding blueprints from the ingested corpus.

A batch, re-runnable job that drafts/updates one project's blueprints
(``scope: project:<id>|global`` and ``scope: project:<id>|area:<name>``) as
``source: generated`` artifacts. It reuses the
existing retrieval layer (:func:`rag.hybrid.hybrid_retrieve`) and the
``LLMClient`` abstraction — there is no separate ingest or retrieval path.

The job is deliberately conservative:

* **Grounded** — every proposed step must cite at least one retrieved chunk;
  ungrounded steps are dropped (consistent with the path-generation grounding
  gate in ``onboarding/pipeline.py``).
* **Idempotent** — a blueprint records the ``corpus_fingerprint`` it was drafted
  from. Re-running against an unchanged corpus is a no-op.
* **Invariant-safe** — it may not remove or downgrade a human-owned step
  (``required`` or ``invariant`` in the active blueprint). Such changes are
  blocked: the protected step is re-injected and the outcome is escalated.

The backend owns blueprint persistence; the AI service is stateless and only
returns generated data.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.models import (
    Blueprint,
    BlueprintProvenance,
    BlueprintStep,
    Skeleton,
    SkeletonRef,
    StepRecord,
)
from onboarding.registry import resolve, upsert_step
from onboarding.scope import Scope
from onboarding.similarity import (
    SIMILARITY_THRESHOLD,
    cosine_similarity,
    step_text,
)
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import Chunk, RetrievalFilters, ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

OutcomeStatus = Literal["created", "updated", "unchanged", "escalated", "skipped"]

_TOP_K = 12
_MIN_SCORE = 0.3


class GenerationError(Exception):
    """Raised when the LLM output for a scope cannot be parsed/validated."""


class GenerationOutcome(BaseModel):
    scope: str
    status: OutcomeStatus
    blueprint: Blueprint | None = None
    draft_version: str | None = None
    chunks_retrieved: int = 0
    steps_drafted: int = 0
    model: str | None = None
    notes: list[str] = Field(default_factory=list)


# --- LLM payload -----------------------------------------------------------


class _GenStep(BaseModel):
    title: str
    description: str = ""
    requirement: Literal["required", "recommended"] = "recommended"
    tags: list[str] = Field(default_factory=list)
    chunk_ids: list[str] = Field(default_factory=list)


class _GenPayload(BaseModel):
    steps: list[_GenStep] = []


# --- corpus fingerprint (idempotency) --------------------------------------


def _fingerprint_fields(chunk: Chunk) -> tuple[str, ...]:
    """Everything that decides whether a chunk can ground a blueprint.

    Chunk ids are content-hashed over ``artifact_id:position:content`` only, so
    re-ingesting the same text under a different source role (``primary`` →
    ``test``, which makes it ineligible for onboarding grounding) or a different
    filename (which appears in citations) leaves the id untouched. A field left
    out here is one whose change silently keeps a stale blueprint alive — extend
    this record when a new field starts affecting eligibility or output.
    """
    return (chunk.id, chunk.text, chunk.source_role, chunk.filename)


def corpus_fingerprint(store: VectorStore, project_id: str) -> str:
    """Stable hash of the project's corpus; changes iff that corpus changes.

    Scoped per project so a change in project A cannot invalidate (or, worse,
    silently validate) project B's generated blueprints.
    """
    digest = hashlib.sha256()
    for chunk in sorted(store.all_chunks(), key=lambda c: c.id):
        if project_id not in chunk.project_ids:
            continue
        for value in _fingerprint_fields(chunk):
            digest.update(value.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest()


# --- scope helpers ---------------------------------------------------------


def default_scopes(active: list[Blueprint], project_id: str) -> list[str]:
    """The project's ``global`` scope plus its active blueprints' area scopes.

    Active blueprints belonging to another project (or to none, i.e. from
    before project separation) do not contribute a scope.
    """
    scopes = {b.scope for b in active if Scope.parse(b.scope).project == project_id}
    scopes.add(Scope.build(project_id).raw)
    return sorted(scopes)


def qualify_scopes(scopes: list[str], project_id: str) -> list[str]:
    """Qualify caller-supplied scope names with the project.

    The backend can keep passing plain ``global`` / ``area:<name>`` names;
    already-qualified scopes for the same project pass through unchanged.
    """
    qualified = [Scope.parse(s).with_project(project_id).raw for s in scopes]
    return list(dict.fromkeys(qualified))


def _scope_label(scope: str) -> str:
    parsed = Scope.parse(scope)
    if parsed.is_global:
        return "everyone on the team, regardless of working area"
    if parsed.area is not None:
        return f"new team members working in {parsed.area}"
    return scope


def _scope_query(scope: str) -> str:
    parsed = Scope.parse(scope)
    if parsed.is_global:
        return "team onboarding essentials getting started setup access"
    if parsed.area is not None:
        return f"{parsed.area} onboarding essentials setup workflow"
    return f"{scope} onboarding"


# --- prompt / parsing ------------------------------------------------------


_DUPLICATE_EXAMPLES = (
    "Examples of what counts as a duplicate (WRONG) vs."
    " an area-specific step (RIGHT):\n"
    "  WRONG: 'Install packages'"
    "          — same concept as a global 'Install dependencies' step\n"
    "  WRONG: 'Set up Python environment'"
    " — same concept as a global 'Verify prerequisites' step\n"
    "  WRONG: 'Run the server'"
    "            — same concept as a global 'Start the service locally' step\n"
    "  WRONG: 'Configure .env file'"
    "       — same concept as a global 'Configure environment variables' step\n"
    "  RIGHT: 'Understand the RAG pipeline architecture'"
    "  — area-specific concept not in global\n"
    "  RIGHT: 'Explore the agent tool registry'"
    "           — area-specific concept not in global\n"
)


def _build_prompt(
    scope: str,
    chunks: list[ScoredChunk],
    global_steps: list[BlueprintStep] | None = None,
) -> list[Message]:
    evidence = "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)
    exclusion = ""
    if global_steps:
        covered = "\n".join(
            f"- {s.title}" + (f": {s.description}" if s.description else "")
            for s in global_steps
        )
        exclusion = (
            "\n3. ALREADY COVERED — the global onboarding blueprint contains the "
            "steps below. You MUST NOT propose any step whose concept overlaps "
            "with them, even when phrased differently.\n\n"
            f"Global steps (do not repeat these concepts):\n{covered}\n\n"
            'Before writing each step, ask: "Is this concept already covered '
            'above, even under a different title?" If yes, skip it.\n\n'
            f"{_DUPLICATE_EXAMPLES}"
            "Only propose steps that are genuinely specific to this area.\n"
        )
    system = (
        "You draft an onboarding blueprint for a software team from its knowledge "
        "base. You are given evidence snippets, each prefixed with its chunk id in "
        "square brackets.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences). Propose a concise, "
        f"ordered list of onboarding steps for {_scope_label(scope)}. Rules:\n"
        "1. Every step MUST reference at least one chunk id from the evidence; do "
        "not invent sources.\n"
        "2. Mark foundational/mandatory steps as 'required', others as "
        "'recommended'.\n"
        f"{exclusion}"
        'JSON schema: {"steps": [{"title": str, "description": str, '
        '"requirement": "required"|"recommended", "tags": [str], '
        '"chunk_ids": [str]}]}'
    )
    user = f"Scope: {scope}\n\nEvidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _draft_steps(
    scope: str,
    chunks: list[ScoredChunk],
    llm: LLMClient,
    pool: dict[str, StepRecord],
    global_steps: list[BlueprintStep] | None = None,
) -> list[SkeletonRef]:
    """Ask the LLM for steps, keep the grounded ones, and upsert them.

    Grounded steps are written into ``pool`` (deduplicating by content
    fingerprint) and returned as ordered skeleton references carrying the
    proposed ``requirement``.
    """
    raw = llm.generate(_build_prompt(scope, chunks, global_steps))
    try:
        payload = _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid generation output: {exc}") from exc

    chunks_by_id = {c.id: c for c in chunks}

    refs: list[SkeletonRef] = []
    seen_ids: set[str] = set()
    for item in payload.steps:
        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if not citations:
            continue  # grounding gate: drop ungrounded steps
        step_id = upsert_step(
            pool,
            title=item.title,
            description=item.description,
            tags=item.tags,
            citations=citations,
        )
        if step_id in seen_ids:
            continue  # identical-title proposals collapse to one ref
        seen_ids.add(step_id)
        refs.append(SkeletonRef(id=step_id, requirement=item.requirement))

    refs = filter_semantic_duplicates(refs, pool, global_steps or [], llm)
    return refs


# --- embedding-based semantic dedup ----------------------------------------


def filter_semantic_duplicates(
    refs: list[SkeletonRef],
    pool: dict[str, StepRecord],
    global_steps: list[BlueprintStep],
    llm: LLMClient,
) -> list[SkeletonRef]:
    """Drop steps that duplicate a prior step by embedding similarity.

    A step is dropped if its embedding is too close to (a) any ``global_steps``
    step (cross-scope: an area step rehashing a global one) or (b) any step
    already kept earlier in this same list (within-scope: two near-identical
    steps in one scope). The first occurrence wins; later duplicates are
    dropped. Passing an empty ``global_steps`` performs pure within-scope dedup,
    which is how the global scope itself is deduplicated.
    """
    # Seed the "already seen" embeddings with the global steps so area steps are
    # compared against them; subsequent kept steps extend this list.
    seen_embeddings: list[list[float]] = [
        llm.embed(step_text(s.title, s.description)) for s in global_steps
    ]

    kept: list[SkeletonRef] = []
    for ref in refs:
        record = pool.get(ref.id)
        if record is None:
            kept.append(ref)
            continue
        embedding = llm.embed(step_text(record.title, record.description))
        max_sim = max(
            (cosine_similarity(embedding, prior) for prior in seen_embeddings),
            default=0.0,
        )
        if max_sim >= SIMILARITY_THRESHOLD:
            logger.info(
                "Dropped duplicate step %r (sim=%.2f with an earlier step)",
                record.title,
                max_sim,
            )
            continue
        kept.append(ref)
        seen_embeddings.append(embedding)
    return kept


# --- invariant gate --------------------------------------------------------


def _enforce_invariants(
    draft: Skeleton, active: Blueprint | None
) -> tuple[Skeleton, list[str]]:
    """Re-inject any human-owned step a draft skeleton would remove or downgrade.

    Protected = ``required`` or ``invariant`` in the active (resolved) blueprint.
    Such steps are never silently dropped: their reference is restored on the
    draft skeleton and the change is reported as an escalation. The referenced
    step already lives in the pool (it is an active step), so the restored ref
    resolves.

    Returns a *new* Skeleton — the input ``draft`` is never mutated.
    """
    if active is None:
        return draft, []

    patched_refs = [r.model_copy(deep=True) for r in draft.steps]
    by_id = {r.id: r for r in patched_refs}
    notes: list[str] = []
    for prev in active.steps:
        protected = prev.requirement == "required" or prev.invariant
        if not protected:
            continue
        existing = by_id.get(prev.id)
        if existing is None:
            patched_refs.append(
                SkeletonRef(
                    id=prev.id,
                    requirement=prev.requirement,
                    invariant=prev.invariant,
                )
            )
            notes.append(
                f"re-injected protected step removed by draft: {prev.title} ({prev.id})"
            )
        elif existing.requirement != prev.requirement:
            existing.requirement = prev.requirement
            existing.invariant = existing.invariant or prev.invariant
            notes.append(
                f"restored requirement of protected step: {prev.title} ({prev.id})"
            )
    patched = draft.model_copy(update={"steps": patched_refs})
    return patched, notes


# --- job -------------------------------------------------------------------


def _next_version(active: Blueprint | None) -> str:
    if active is None:
        return "1"
    try:
        return str(int(active.version) + 1)
    except ValueError:
        return f"{active.version}-next"


def _generate_scope(
    scope: str,
    *,
    fingerprint: str,
    project_id: str,
    llm: LLMClient,
    store: VectorStore,
    bm25_cache: BM25IndexCache,
    model: str | None,
    active: Blueprint | None = None,
    global_steps: list[BlueprintStep] | None = None,
) -> GenerationOutcome:
    # Idempotency: skip if the active blueprint already reflects this corpus.
    if (
        active is not None
        and active.source == "generated"
        and active.provenance is not None
        and active.provenance.corpus_fingerprint == fingerprint
    ):
        return GenerationOutcome(
            scope=scope, status="unchanged", notes=["corpus unchanged since active"]
        )

    if store.count() == 0:
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["corpus is empty"]
        )

    chunks = hybrid_retrieve(
        question=_scope_query(scope),
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=bm25_cache,
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
        filters=RetrievalFilters(project_id=project_id),
    )
    if not chunks:
        return GenerationOutcome(
            scope=scope,
            status="skipped",
            notes=["no grounding evidence retrieved for this project"],
        )

    # Strip chunks already cited by global steps from the area evidence pool.
    # The grounding gate requires every step to cite at least one chunk, so
    # removing global-owned chunks makes it structurally impossible for the LLM
    # to generate setup/install steps that merely rephrase what global covers.
    if global_steps:
        global_chunk_ids = {ref.chunk_id for s in global_steps for ref in s.citations}
        chunks = [c for c in chunks if c.id not in global_chunk_ids]
    if not chunks:
        return GenerationOutcome(
            scope=scope,
            status="skipped",
            notes=["no area-specific evidence after excluding global citations"],
        )

    pool: dict[str, StepRecord] = {}
    # Seed the pool with the active blueprint's steps so protected steps that
    # _enforce_invariants re-injects can be resolved (the backend owns the pool;
    # the active blueprint is the only state passed in).
    for step in active.steps if active is not None else []:
        pool.setdefault(
            step.id,
            StepRecord(
                id=step.id,
                title=step.title,
                description=step.description,
                audience=step.audience,
                min_experience=step.min_experience,
                tags=step.tags,
                resources=step.resources,
                citations=step.citations,
            ),
        )
    refs = _draft_steps(scope, chunks, llm, pool, global_steps)
    if not refs:
        return GenerationOutcome(
            scope=scope, status="skipped", notes=["no grounded steps proposed"]
        )

    draft = Skeleton(
        scope=scope,
        version=_next_version(active),
        source="generated",
        steps=refs,
        provenance=BlueprintProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=model,
        ),
    )

    draft, invariant_notes = _enforce_invariants(draft, active)
    if draft.provenance is not None:
        draft.provenance.notes = invariant_notes

    resolved = resolve(draft, pool)

    status: OutcomeStatus
    if invariant_notes:
        status = "escalated"
    elif active is None:
        status = "created"
    else:
        status = "updated"
    return GenerationOutcome(
        scope=scope,
        status=status,
        blueprint=resolved,
        draft_version=draft.version,
        chunks_retrieved=len(chunks),
        steps_drafted=len(refs),
        model=model,
        notes=invariant_notes,
    )


def generate_blueprints(
    llm: LLMClient,
    store: VectorStore,
    *,
    project_id: str,
    scopes: list[str] | None = None,
    active: list[Blueprint] | None = None,
) -> list[GenerationOutcome]:
    """Draft/update the project's blueprints; returns data without persisting.

    Generation is project-scoped end to end: evidence is retrieved with a
    project filter, the produced scopes are project-qualified
    (``project:<id>|global``, ``project:<id>|area:<name>``), and the corpus
    fingerprint that drives idempotency covers only this project's chunks.

    ``active`` is the set of currently-active blueprints owned by the backend.
    They drive idempotency (skip when the corpus fingerprint is unchanged) and
    version numbering; the AI service holds no state of its own. Active
    blueprints from other projects are ignored — they cannot match a
    project-qualified scope.
    """
    fingerprint = corpus_fingerprint(store, project_id)
    bm25_cache = BM25IndexCache()
    model = llm.model_name

    active_by_scope = {b.scope: b for b in (active or [])}
    outcomes: list[GenerationOutcome] = []
    resolved_scopes = (
        qualify_scopes(scopes, project_id)
        if scopes
        else default_scopes(active or [], project_id)
    )

    # Generate global first so area scopes can exclude its steps.
    global_scope = Scope.build(project_id).raw
    if global_scope in resolved_scopes:
        resolved_scopes = [global_scope] + [
            s for s in resolved_scopes if s != global_scope
        ]

    global_steps: list[BlueprintStep] | None = None
    for scope in resolved_scopes:
        is_global = scope == global_scope
        try:
            outcome = _generate_scope(
                scope,
                fingerprint=fingerprint,
                project_id=project_id,
                llm=llm,
                store=store,
                bm25_cache=bm25_cache,
                model=model,
                active=active_by_scope.get(scope),
                global_steps=global_steps if not is_global else None,
            )
            outcomes.append(outcome)
            # After global is generated, capture its steps for area scopes
            # from the outcome (no disk reads).
            if is_global and global_steps is None and outcome.blueprint is not None:
                global_steps = outcome.blueprint.steps
        except GenerationError as exc:
            logger.warning("Generation failed for scope %s: %s", scope, exc)
            outcomes.append(
                GenerationOutcome(scope=scope, status="skipped", notes=[str(exc)])
            )
    return outcomes
