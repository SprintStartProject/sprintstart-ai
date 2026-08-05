"""Deterministic staged pipeline that assembles a personalized onboarding path.

Stages: (1) select blueprints by scope, (2) filter/order steps by proficiency,
(3) retrieve grounding evidence across the corpus, (4) LLM synthesis of the
personalized layer, (5) validation & quality gates, (6) emit. The pipeline is a
generator that yields :class:`StageProgress` markers (the orchestrator turns
these into SSE events) and returns the final :class:`OnboardingPath`.

Gates guarantee a good path independent of the (current or future) blueprint
source: schema (invalid LLM output -> blueprint-only fallback), coverage
(required steps always present), grounding (LLM steps must cite evidence), and
human-owned invariants (e.g. a security-policy step is always present).
"""

import logging
from collections.abc import Generator
from dataclasses import dataclass

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.blueprints import select_blueprints
from onboarding.checks import generate_phase_check
from onboarding.models import (
    Blueprint,
    BlueprintStep,
    CitationRef,
    OnboardingPath,
    PathPhase,
    PathStep,
    PersonProfile,
    Requirement,
    Task,
    experience_rank,
    proficiency_rank,
)
from onboarding.quality import evaluate
from onboarding.scope import Scope
from onboarding.similarity import OVERLAP_THRESHOLD, step_text, text_overlap
from onboarding.synthesis import SynthesisError, SynthesisResult, synthesize
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import RetrievalFilters, ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

STAGES = ("select", "filter", "retrieve", "synthesize", "validate", "emit")

# Human-owned invariants: steps the assembler guarantees are present in every
# path, regardless of blueprint source. Enforced in code, not in blueprint data.
# Currently empty — add mandatory/compliance steps here to force them in.
_INVARIANTS: list[PathStep] = []


@dataclass(frozen=True)
class StageProgress:
    name: str
    detail: str = ""


def _phase_title(scope: str) -> str:
    parsed = Scope.parse(scope)
    if parsed.is_global:
        return "Getting started"
    if parsed.area is not None:
        return f"{parsed.area.capitalize()} essentials"
    return scope


def _step_applies(step: BlueprintStep, profile: PersonProfile) -> bool:
    """Filter a step by audience and proficiency.

    Required steps always apply (they are guaranteed by the coverage gate);
    recommended steps are gated by ``audience`` and ``min_experience``.
    """
    if step.requirement == "required":
        return True

    # A skill the person listed that matches this step's tags surfaces it
    # regardless of the step's audience — skills are role-orthogonal, so a
    # listed skill can pull in a step outside one's working area.
    skill_names = {s.name.lower() for s in profile.skills}
    skill_match = bool(skill_names & {t.lower() for t in step.tags})

    if step.audience and not skill_match:
        haystack = {profile.working_area.lower(), *(t.lower() for t in profile.tags)}
        haystack.update(skill_names)
        if not haystack & {a.lower() for a in step.audience}:
            return False

    return proficiency_rank(profile.skills) >= experience_rank(step.min_experience)


def _order_steps(
    steps: list[BlueprintStep], required_ids: set[str]
) -> list[BlueprintStep]:
    """Required steps before recommended ones, preserving authored order.

    ``required_ids`` is the cross-scope merge: a step required in *any* selected
    skeleton is treated as required here, so a step that's recommended globally
    but mandatory for an area sorts (and renders) as required.
    """
    return [s for s in steps if s.id in required_ids] + [
        s for s in steps if s.id not in required_ids
    ]


class OnboardingPipeline:
    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        *,
        min_score: float = 0.3,
        bm25_cache: BM25IndexCache | None = None,
    ) -> None:
        self._llm = llm
        self._store = store
        self._min_score = min_score
        # A shared cache lets the BM25 index persist across requests (it
        # self-invalidates when the corpus size changes); falls back to a
        # private cache when none is injected (e.g. in tests).
        self._bm25_cache = bm25_cache or BM25IndexCache()

    def run(
        self,
        profile: PersonProfile,
        blueprints: list[Blueprint],
        project_id: str,
    ) -> Generator[StageProgress, None, OnboardingPath]:
        # (1) select — blueprints are always provided by the backend, and only
        # those scoped to this project are eligible
        selected = select_blueprints(blueprints, profile, project_id)
        total_steps = sum(len(b.steps) for b in selected)
        scopes = ", ".join(b.scope for b in selected) or "none"
        yield StageProgress(
            "select", f"{len(selected)} blueprint(s) [{scopes}],{total_steps} step(s)"
        )
        blueprint_versions = {b.scope: b.version for b in selected}
        required_ids = {
            s.id for b in selected for s in b.steps if s.requirement == "required"
        }

        # (2) filter / order
        phases, kept_steps = _build_phases(selected, profile)
        yield StageProgress("filter", f"{len(kept_steps)} step(s) after filtering")

        # (3) retrieve grounding evidence per step, scoped to the project
        chunks_per_step = self._retrieve_per_step(kept_steps, project_id)
        total_chunks = sum(len(c) for c in chunks_per_step.values())
        yield StageProgress(
            "retrieve",
            f"{total_chunks} chunk(s) retrieved across {len(kept_steps)} step(s)",
        )

        # (4) synthesize the personalized layer (schema gate -> fallback)
        notes: list[str] = []
        try:
            result = synthesize(profile, kept_steps, chunks_per_step, self._llm)
            rewritten_count = len(result.rewrites)
            enriched_count = len(result.enrichments)
            added_count = len(result.added_steps)
            yield StageProgress(
                "synthesize",
                f"{rewritten_count} rewritten, {enriched_count} enriched,"
                f" {added_count} added step(s)",
            )
        except SynthesisError as exc:
            logger.warning("Synthesis failed, using blueprint-only fallback: %s", exc)
            result = SynthesisResult()
            notes.append("LLM synthesis unavailable; blueprint-only fallback")
            yield StageProgress("synthesize", "fallback: LLM synthesis unavailable")

        # (5) validate & quality gates
        _apply_synthesis(phases, result.enrichments, result.rewrites, result.tasks)
        gate_notes = _apply_grounding_gate(phases, result.added_steps)
        notes.extend(gate_notes)
        _enforce_coverage(phases, selected, profile)
        _enforce_invariants(phases)
        total_out = sum(len(p.steps) for p in phases)
        gate_summary = "; ".join(gate_notes) if gate_notes else "all gates passed"
        yield StageProgress("validate", f"{total_out} step(s) in path; {gate_summary}")

        # (5.5) generate a knowledge check per phase, grounded in that phase's
        # step evidence. Best-effort per phase (see generate_phase_check); does
        # not add a stage of its own so the SSE stage contract is unchanged.
        for phase in phases:
            seen_chunk_ids: set[str] = set()
            phase_chunks: list[ScoredChunk] = []
            for step in phase.steps:
                for chunk in chunks_per_step.get(step.id, []):
                    if chunk.id not in seen_chunk_ids:
                        seen_chunk_ids.add(chunk.id)
                        phase_chunks.append(chunk)
            phase.check = generate_phase_check(phase, phase_chunks, self._llm)

        # (6) emit
        yield StageProgress("emit")
        path = OnboardingPath(
            working_area=profile.working_area,
            phases=phases,
            blueprint_versions=blueprint_versions,
        )
        path.quality = evaluate(path, required_ids, notes)
        return path

    def _retrieve_per_step(
        self, steps: list[BlueprintStep], project_id: str
    ) -> dict[str, list[ScoredChunk]]:
        if self._store.count() == 0:
            return {}
        filters = RetrievalFilters(project_id=project_id)
        result: dict[str, list[ScoredChunk]] = {}
        for step in steps:
            query = (
                f"{step.title}: {step.description}" if step.description else step.title
            )
            try:
                result[step.id] = hybrid_retrieve(
                    question=query,
                    llm=self._llm,
                    store=self._store,
                    top_k=4,
                    min_score=self._min_score,
                    bm25_cache=self._bm25_cache,
                    exclude_roles=GROUNDING_EXCLUDED_ROLES,
                    filters=filters,
                )
            except LLMUnavailableError:
                raise
            except Exception:
                logger.exception(
                    "Retrieval failed for step %s; continuing without grounding",
                    step.id,
                )
                result[step.id] = []
        return result


def _is_semantic_duplicate(
    step: BlueprintStep, seen_texts: list[str], seen_titles: list[str]
) -> bool:
    """Return True if *step* overlaps too much with an already-seen step.

    Two signals are combined with MAX so that either one alone is enough:
    - Full-text Jaccard (title + description) catches broad thematic overlap.
    - Title-only Jaccard catches rewrites like "Run Service Locally" vs
      "Run the Service Locally" where different descriptions would dilute the
      combined score below the threshold.
    """
    full_text = step_text(step.title, step.description)
    for seen_full, seen_title in zip(seen_texts, seen_titles, strict=False):
        score = max(
            text_overlap(full_text, seen_full),
            text_overlap(step.title, seen_title),
        )
        if score >= OVERLAP_THRESHOLD:
            return True
    return False


def _build_phases(
    blueprints: list[Blueprint], profile: PersonProfile
) -> tuple[list[PathPhase], list[BlueprintStep]]:
    phases: list[PathPhase] = []
    kept_steps: list[BlueprintStep] = []
    seen: set[str] = set()
    seen_texts: list[str] = []
    seen_titles: list[str] = []

    # Cross-scope status merge: a step required in any selected skeleton is
    # required in the path (contributors can mark a shared step mandatory for
    # their area). This also feeds ordering and the rendered requirement.
    required_ids = {
        s.id for b in blueprints for s in b.steps if s.requirement == "required"
    }

    def _req(step_id: str) -> Requirement:
        return "required" if step_id in required_ids else "recommended"

    for blueprint in blueprints:
        applicable = _order_steps(
            [s for s in blueprint.steps if _step_applies(s, profile)], required_ids
        )
        # Cross-source dedup: a step contributed by an earlier scope (global
        # before area) wins; the same step id from a later scope is skipped.
        # Semantic dedup: steps with different ids but overlapping content are
        # dropped across scopes — required or not. The coverage gate below only
        # re-injects a required step if no semantically equivalent step is
        # already present, so genuinely unique required steps are never lost.
        applicable = [
            s
            for s in applicable
            if s.id not in seen
            and not _is_semantic_duplicate(s, seen_texts, seen_titles)
        ]
        if not applicable:
            continue
        seen.update(s.id for s in applicable)
        seen_texts.extend(step_text(s.title, s.description) for s in applicable)
        seen_titles.extend(s.title for s in applicable)
        kept_steps.extend(applicable)
        phases.append(
            PathPhase(
                title=_phase_title(blueprint.scope),
                scope=blueprint.scope,
                steps=[
                    PathStep(
                        id=s.id,
                        title=s.title,
                        description=s.description,
                        requirement=_req(s.id),
                        origin="blueprint",
                        tags=s.tags,
                        resources=s.resources,
                        citations=s.citations,
                        tasks=s.tasks,
                    )
                    for s in applicable
                ],
            )
        )

    return phases, kept_steps


def _apply_synthesis(
    phases: list[PathPhase],
    enrichments: dict[str, list[CitationRef]],
    rewrites: dict[str, str],
    tasks: dict[str, list[Task]] | None = None,
) -> None:
    for phase in phases:
        for step in phase.steps:
            if step.id in rewrites and rewrites[step.id].strip():
                step.description = rewrites[step.id]
            refs = enrichments.get(step.id)
            if refs:
                step.citations = refs
            if tasks and step.id in tasks:
                step.tasks = tasks[step.id]


def _apply_grounding_gate(
    phases: list[PathPhase], added_steps: list[PathStep]
) -> list[str]:
    """Drop LLM-added steps without a citation; append grounded, novel ones."""
    present = {s.id for phase in phases for s in phase.steps}
    grounded: list[PathStep] = []
    dropped = 0
    for step in added_steps:
        if not step.citations:
            dropped += 1  # grounding gate
            continue
        if step.id in present:
            continue  # already covered by a blueprint step (cross-source dedup)
        present.add(step.id)
        grounded.append(step)
    notes: list[str] = []
    if dropped:
        notes.append(f"dropped {dropped} ungrounded LLM step(s)")
    if grounded:
        phases.append(PathPhase(title="Recommended for you", steps=grounded))
    return notes


def _enforce_coverage(
    phases: list[PathPhase], blueprints: list[Blueprint], profile: PersonProfile
) -> None:
    """Guarantee every required blueprint step is present; re-inject if missing.

    A step is only re-injected if it is absent both by ID *and* by semantic
    equivalence — i.e. the concept it covers isn't already represented in the
    path by a different step from an earlier scope.
    """
    present = {s.id for p in phases for s in p.steps}
    covered_texts = [step_text(s.title, s.description) for p in phases for s in p.steps]
    covered_titles = [s.title for p in phases for s in p.steps]
    for blueprint in blueprints:
        missing = [
            s
            for s in blueprint.steps
            if s.requirement == "required"
            and s.id not in present
            and not _is_semantic_duplicate(s, covered_texts, covered_titles)
        ]
        if not missing:
            continue
        target = _phase_for_scope(phases, blueprint.scope)
        for step in missing:
            target.steps.insert(
                0,
                PathStep(
                    id=step.id,
                    title=step.title,
                    description=step.description,
                    requirement="required",
                    origin="blueprint",
                    tags=step.tags,
                    resources=step.resources,
                    citations=step.citations,
                    tasks=step.tasks,
                ),
            )
            present.add(step.id)
            covered_texts.append(step_text(step.title, step.description))
            covered_titles.append(step.title)


def _phase_for_scope(phases: list[PathPhase], scope: str) -> PathPhase:
    for phase in phases:
        if phase.scope == scope:
            return phase
    # Fallback: match by title for phases without a scope (e.g. "Recommended").
    title = _phase_title(scope)
    for phase in phases:
        if phase.scope is None and phase.title == title:
            return phase
    phase = PathPhase(title=title, scope=scope, steps=[])
    phases.insert(0, phase)
    return phase


def _enforce_invariants(phases: list[PathPhase]) -> None:
    """Force every human-owned invariant step into the path if absent.

    Enforced regardless of blueprint source. ``_INVARIANTS`` is currently empty
    (no mandatory steps), so this is a no-op until invariants are added.
    """
    present = {s.id for p in phases for s in p.steps}
    missing = [step for step in _INVARIANTS if step.id not in present]
    if not missing:
        return
    if not phases:
        phases.append(PathPhase(title="Getting started", steps=[]))
    for step in reversed(missing):
        phases[0].steps.insert(0, step.model_copy(deep=True))
