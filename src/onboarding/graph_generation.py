"""AI-authoring of competency proposals from the ingested corpus.

A batch, re-runnable job that proposes new ``Competency`` nodes (``SKILL``/
``CONCEPT``) grounded in the ingested repo. It reuses the same retrieval layer
(:func:`rag.hybrid.hybrid_retrieve`) and idempotency mechanism
(:func:`onboarding.corpus.corpus_fingerprint`) as every other offline job.

One pass, one LLM call. A second pass used to run after this one to draw edges
between the nodes, added because a single call spent its budget on nodes and
left the graph a scatter. It is gone with the structure it built: the backend
retired prerequisite edges entirely (see
``forks/SKILL_MAP_RETIREMENT_DESIGN.md``). ``RELATED`` -- the kind this
generator was told most relationships belonged to -- was read by nothing, and
where edges *were* read the kind was dropped, so a ``RELATED`` edge could be
spoken to a hire as "usually comes after". Competencies are a flat vocabulary
now: what the ledger keys off, what a module teaches, what the matcher counts.

Because nodes are all this derives, and nodes are a function of the corpus, an
unchanged corpus short-circuits the whole run.

Deliberately simple: the backend's vocabulary can only grow through this job --
a proposal run only ever adds new candidates, it never redrafts or replaces
what already exists. The real constraint is dedup: a competency already live
must not be re-proposed (exact key match), and a near-duplicate must not slip
through either (embedding similarity).

Idempotency is driven by an explicit ``last_fingerprint`` the caller passes in
and is responsible for persisting: this service is stateless, so there is no
"active proposal" object here to carry it the way every other job's provenance
fingerprint does.
"""

import json
import logging
import re
from collections.abc import Generator
from datetime import UTC, datetime

from pydantic import BaseModel, Field, ValidationError

from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from onboarding.citations import resolve_citations
from onboarding.corpus import corpus_fingerprint
from onboarding.graph_models import (
    ActiveCompetency,
    GraphProposalOutcome,
    GraphProvenance,
    ProposedCompetency,
    TombstonedCompetency,
)
from onboarding.models import CitationRef
from onboarding.progress import ProgressEvent, ProgressStream, drain
from onboarding.similarity import SIMILARITY_THRESHOLD, cosine_similarity, step_text
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 20
_MIN_SCORE = 0.3
_QUERY = (
    "core skills, concepts, technologies, architecture patterns and domain "
    "knowledge required to work productively in this codebase"
)
_KEY_RE = re.compile(r"[^a-z0-9]+")


class GenerationError(Exception):
    """Raised when the LLM output for a graph proposal cannot be parsed/validated."""


# --- LLM payload -------------------------------------------------------------


class _GenCompetency(BaseModel):
    key: str
    label: str
    description: str = ""
    kind: str = "SKILL"
    area: str | None = None
    repo_ref: str | None = None
    chunk_ids: list[str] = Field(default_factory=list)


class _GenPayload(BaseModel):
    competencies: list[_GenCompetency] = Field(default_factory=list[_GenCompetency])


# --- prompt / parsing ---------------------------------------------------------


def _normalize_key(raw: str) -> str:
    """Kebab-case a proposed key so it matches the backend's stable-key convention."""
    return _KEY_RE.sub("-", raw.strip().lower()).strip("-")


def _resolve_area(raw: str | None, existing: list[str]) -> str | None:
    """Snap a proposed area onto the one already in use, when it is the same one.

    The prompt asks the model to reuse an existing area, but asking is not
    enforcing: it still returns "auth" for "Authentication" often enough to
    fragment the grouping, and a grouping that fragments is worse than none
    because it looks organised. The backend normalises on write for exactly the
    same reason -- doing it here as well means the *proposal* a caller inspects
    already reads the way it will be stored, rather than being quietly corrected
    one layer down.

    Blank becomes ``None``: "could not place it" is a real answer, and an empty
    string would be a group whose name is nothing.
    """
    trimmed = (raw or "").strip()
    if not trimmed:
        return None
    for area in existing:
        if area.casefold() == trimmed.casefold():
            return area
    return trimmed


def _build_prompt(
    chunks: list[ScoredChunk],
    active: list[ActiveCompetency],
    existing_areas: list[str] | None = None,
    tombstoned: list[TombstonedCompetency] | None = None,
) -> list[Message]:
    evidence = "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)
    exclusion = ""
    if active:
        existing = "\n".join(f"- {c.key}: {c.label}" for c in active)
        exclusion = (
            "\nThese competencies already exist -- do NOT propose them again "
            "under a new key, even if phrased differently.\n\n"
            f"Existing competencies:\n{existing}\n"
        )
    graves = tombstoned or []
    if graves:
        removed = "\n".join(f"- {c.key}: {c.label}" for c in graves)
        exclusion += (
            "\nThese were deliberately REMOVED by a person. Do not propose them "
            "again, under any key and however differently phrased -- they were "
            "not missed, they were rejected.\n\n"
            f"Removed competencies:\n{removed}\n"
        )
    areas = existing_areas or []
    # Show the model the grouping that exists so it joins it rather than coining a
    # synonym. The backend normalises on write as well, but a model that picks the
    # area itself produces a vocabulary somebody can read, not one that had to be
    # corrected. Same mechanic as the existing-competency list above.
    area_rule = (
        "6. `area` groups the vocabulary by subject (e.g. 'Authentication', "
        "'Ingestion'). Reuse one of the areas already in use below when the "
        "competency belongs to it -- only name a new area when none of them "
        "fits. Use null if the evidence does not place it in any area; a wrong "
        "grouping is worse than none.\n"
    )
    if areas:
        listed = "\n".join(f"- {a}" for a in areas)
        area_rule += f"\nAreas already in use:\n{listed}\n"
    system = (
        "You propose entries for a team's competency vocabulary from its "
        "knowledge base. You are given evidence snippets, each prefixed with "
        "its chunk id in square brackets.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences). Rules:\n"
        "1. Propose competencies of kind SKILL (a tool/language/technology) "
        "or CONCEPT (a domain/architecture idea specific to this codebase).\n"
        "2. Every competency MUST cite at least one chunk id from the evidence; "
        "do not invent sources.\n"
        "3. Each competency needs a short, stable, kebab-case `key` (e.g. "
        "'kotlin', 'our-domain-model') distinct from any existing key.\n"
        "4. `repo_ref` is an optional pointer to the file/path the competency "
        "is grounded in, when the evidence makes one obvious.\n"
        "5. This is a flat vocabulary, not an ordering: do not state or imply "
        "what must be learned before what. An area groups competencies by "
        "subject; it never says which comes first.\n"
        f"{area_rule}"
        f"{exclusion}"
        'JSON schema: {"competencies": [{"key": str, "label": str, '
        '"description": str, "kind": "SKILL"|"CONCEPT", "area": str|null, '
        '"repo_ref": str|null, "chunk_ids": [str]}]}'
    )
    user = f"Evidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _parse_payload(raw: str) -> _GenPayload:
    try:
        return _GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise GenerationError(f"invalid generation output: {exc}") from exc


def _filter_duplicate_competencies(
    candidates: list[tuple[_GenCompetency, list[CitationRef]]],
    active: list[ActiveCompetency],
    tombstoned: list[TombstonedCompetency],
    llm: LLMClient,
) -> list[tuple[_GenCompetency, list[CitationRef]]]:
    """Drop exact-key and near-duplicate (embedding similarity) proposals.

    Mirrors ``generation.filter_semantic_duplicates``, adapted to competencies:
    a proposal is dropped if its key exactly matches an active competency, or
    if its embedding is too close to an active competency's or an
    already-kept proposal's (first occurrence wins).

    Tombstoned competencies are blocked the same two ways, and the second is the
    one that matters: the prompt already asks the model not to re-propose them,
    but asking is not enforcing, and a deletion that only holds against the exact
    key leaks -- the competency returns next crawl under a rephrasing and is
    deleted again, forever.
    """
    blocked_keys = {c.key for c in active} | {c.key for c in tombstoned}
    seen_embeddings: list[list[float]] = [
        llm.embed(step_text(c.label, c.description)) for c in active
    ] + [llm.embed(step_text(c.label, "")) for c in tombstoned]

    kept: list[tuple[_GenCompetency, list[CitationRef]]] = []
    seen_keys: set[str] = set()
    for competency, citations in candidates:
        key = _normalize_key(competency.key)
        if not key or key in blocked_keys or key in seen_keys:
            continue
        embedding = llm.embed(step_text(competency.label, competency.description))
        max_sim = max(
            (cosine_similarity(embedding, prior) for prior in seen_embeddings),
            default=0.0,
        )
        if max_sim >= SIMILARITY_THRESHOLD:
            logger.info(
                "Dropped duplicate competency proposal %r (sim=%.2f)",
                competency.label,
                max_sim,
            )
            continue
        seen_keys.add(key)
        seen_embeddings.append(embedding)
        kept.append((competency, citations))
    return kept


# --- the run -------------------------------------------------------------------


def stream_competency_graph(
    llm: LLMClient,
    store: VectorStore,
    *,
    active_competencies: list[ActiveCompetency] | None = None,
    existing_areas: list[str] | None = None,
    tombstoned_competencies: list[TombstonedCompetency] | None = None,
    last_fingerprint: str | None = None,
) -> Generator[ProgressEvent, None, GraphProposalOutcome]:
    """Propose the vocabulary, yielding live progress and returning the outcome.

    This is the single implementation: :func:`generate_competency_graph` drives it
    to completion for the non-streaming path, and the streaming route relays its
    events. So the proposal a PM watches assemble is byte-for-byte the proposal the
    batch call would have produced — the stream is a view, never a second answer.

    The vocabulary literally builds: each grounded, deduped competency is emitted
    as an ``item`` the instant it clears its gate. An ``item`` is a promise of
    validation — if it was streamed, it is in the persisted proposal.
    """
    progress = ProgressStream("competency_graph")
    active = active_competencies or []
    tombstoned = tombstoned_competencies or []
    fingerprint = corpus_fingerprint(store)

    if store.count() == 0:
        outcome = GraphProposalOutcome(status="skipped", notes=["corpus is empty"])
        yield progress.warning("The project has no indexed material yet")
        yield progress.done("No competencies could be proposed", _dump(outcome))
        return outcome

    # Competencies are all this derives, and they are a function of the corpus, so
    # an unchanged corpus can hold nothing new and the whole run short-circuits.
    if last_fingerprint is not None and last_fingerprint == fingerprint:
        outcome = GraphProposalOutcome(
            status="unchanged",
            notes=["corpus unchanged since last proposal run"],
        )
        yield progress.done("Nothing new to propose", _dump(outcome))
        return outcome

    yield progress.stage("retrieving", "Searching the corpus for competencies")
    chunks = hybrid_retrieve(
        question=_QUERY,
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=BM25IndexCache(),
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
    )
    if not chunks:
        outcome = GraphProposalOutcome(
            status="skipped", notes=["no grounding evidence retrieved"]
        )
        yield progress.warning("Nothing in the corpus grounded a competency")
        yield progress.done("No competencies could be proposed", _dump(outcome))
        return outcome

    yield progress.stage(
        "grounding", f"Proposing competencies from {len(chunks)} source(s)"
    )
    try:
        proposed_competencies = _propose_competencies(
            llm, chunks, active, existing_areas, tombstoned
        )
    except GenerationError as exc:
        logger.warning("Competency proposal generation failed: %s", exc)
        outcome = GraphProposalOutcome(status="skipped", notes=[str(exc)])
        yield progress.warning("The proposed competencies could not be read")
        yield progress.done("No competencies could be proposed", _dump(outcome))
        return outcome

    for competency in proposed_competencies:
        yield progress.item(
            competency.model_dump(mode="json"),
            f"Competency: {competency.label}",
        )

    if not proposed_competencies:
        outcome = GraphProposalOutcome(
            status="skipped",
            chunks_retrieved=len(chunks),
            notes=["no grounded, non-duplicate competencies proposed"],
        )
        yield progress.done("Nothing new to propose", _dump(outcome))
        return outcome

    outcome = GraphProposalOutcome(
        status="proposed",
        competencies=proposed_competencies,
        provenance=GraphProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
        ),
        chunks_retrieved=len(chunks),
    )
    yield progress.done(
        f"Proposed {len(proposed_competencies)} competency/-ies",
        _dump(outcome),
    )
    return outcome


def _dump(outcome: GraphProposalOutcome) -> dict[str, object]:
    """The outcome as a JSON-safe dict for a ``done`` event's ``result``."""
    return outcome.model_dump(mode="json")


def generate_competency_graph(
    llm: LLMClient,
    store: VectorStore,
    *,
    active_competencies: list[ActiveCompetency] | None = None,
    existing_areas: list[str] | None = None,
    tombstoned_competencies: list[TombstonedCompetency] | None = None,
    last_fingerprint: str | None = None,
) -> GraphProposalOutcome:
    """Propose new competencies for the backend to persist.

    ``active_competencies`` is the backend's current live vocabulary -- it drives
    dedup (never re-propose an existing key). ``last_fingerprint`` is whatever
    fingerprint the caller recorded from the previous run (idempotency); this
    service holds no state of its own.

    Backed by :func:`stream_competency_graph` so the non-streaming and streaming
    paths are the same computation.
    """
    return drain(
        stream_competency_graph(
            llm,
            store,
            active_competencies=active_competencies,
            existing_areas=existing_areas,
            tombstoned_competencies=tombstoned_competencies,
            last_fingerprint=last_fingerprint,
        )
    )


def _propose_competencies(
    llm: LLMClient,
    chunks: list[ScoredChunk],
    active: list[ActiveCompetency],
    existing_areas: list[str] | None = None,
    tombstoned: list[TombstonedCompetency] | None = None,
) -> list[ProposedCompetency]:
    """Grounded, deduped competency proposals."""
    payload = _parse_payload(
        llm.generate(_build_prompt(chunks, active, existing_areas, tombstoned))
    )

    chunks_by_id = {c.id: c for c in chunks}
    grounded: list[tuple[_GenCompetency, list[CitationRef]]] = []
    for item in payload.competencies:
        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if not citations:
            continue  # grounding gate: drop ungrounded proposals
        if item.kind not in ("SKILL", "CONCEPT"):
            continue
        grounded.append((item, citations))

    return [
        ProposedCompetency(
            key=_normalize_key(item.key),
            label=item.label,
            description=item.description,
            kind=item.kind,  # type: ignore[arg-type]
            area=_resolve_area(item.area, existing_areas or []),
            repo_ref=item.repo_ref,
            citations=citations,
        )
        for item, citations in _filter_duplicate_competencies(
            grounded, active, tombstoned or [], llm
        )
    ]
