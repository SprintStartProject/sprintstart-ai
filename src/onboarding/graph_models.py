"""Domain models for AI-proposed competencies.

A competency is a durable name for something somebody can be proficient in
(see the backend's ``Competency`` entity): the ledger keys off it, a module
teaches it, the starter-work matcher counts it. This service never persists any
of it -- it proposes candidates for the backend to store, the same
proposal-only relationship every other proposal-only job has with the backend.

It is a **flat vocabulary, not a graph**. Prerequisite and related edges were
retired along with the DAG they described (see
``forks/SKILL_MAP_RETIREMENT_DESIGN.md``); nothing here states an order.

The backend's vocabulary can only grow through this job (no
replace-whole-vocabulary, no removal/modification) -- so this module has no
analogue of ``generation._enforce_invariants``. There is nothing to protect a
proposal run from silently dropping, because a proposal run never removes
anything; it only ever adds new candidates alongside whatever already exists.
"""

from typing import Literal

from pydantic import BaseModel, Field

from onboarding.models import CitationRef

CompetencyKind = Literal["SKILL", "CONCEPT"]

ProposalStatus = Literal["proposed", "unchanged", "skipped"]


class ProposedCompetency(BaseModel):
    """A candidate competency node grounded in retrieved evidence."""

    key: str
    label: str
    description: str = ""
    kind: CompetencyKind
    area: str | None = None
    """What this competency is about, for grouping -- "Authentication", "Ingestion".

    Free text, because a fixed taxonomy cannot fit a codebase nobody has seen.
    ``None`` when the evidence does not place it in one; the backend stores that
    as "not grouped yet" rather than inventing a bucket for it.
    """
    repo_ref: str | None = None
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class GraphProvenance(BaseModel):
    """Why a proposal run looks the way it does; the usual provenance shape."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class ActiveCompetency(BaseModel):
    """A competency already live in the backend's vocabulary.

    Drives dedup (never re-proposed as new). Carries no proposal-time metadata
    because the backend's vocabulary has none -- competencies are just live
    rows, not versioned drafts.
    """

    key: str
    label: str
    description: str = ""
    kind: CompetencyKind
    area: str | None = None
    repo_ref: str | None = None


class TombstonedCompetency(BaseModel):
    """A competency somebody deliberately removed, which must not come back.

    Carries the label as well as the key because the thing a tombstone has to
    stop is a *rephrasing*: dedup matches on the exact key and on embedding
    similarity, and only the label feeds the second. A tombstone the generator
    never sees is not a tombstone.
    """

    key: str
    label: str


class GraphProposalOutcome(BaseModel):
    """Result of one competency proposal run."""

    status: ProposalStatus
    competencies: list[ProposedCompetency] = Field(
        default_factory=list[ProposedCompetency]
    )
    provenance: GraphProvenance | None = None
    chunks_retrieved: int = 0
    notes: list[str] = Field(default_factory=list[str])
