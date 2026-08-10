"""Domain models for AI-proposed competency modules.

A module is the shared, reviewable teaching artifact for one competency: an
ordered list of typed pages plus the gating check a hire must pass. It replaces
the per-hire lesson (:mod:`onboarding.lessons`), where every hire got a
separately generated wall of prose for the same competency — content nobody
could review, edit, or improve, because there was no single thing to edit.

Losing per-hire tailoring of the prose is deliberate. Personalization lives in
*which* competencies are on a hire's path and what they can test out of, not in
each person getting their own paraphrase.

This service never persists modules: it proposes one per request and the backend
stores it as a proposal until a PM approves it.
"""

from typing import Literal

from pydantic import BaseModel, Field

from onboarding.models import CitationRef

# Mirrors ``onboarding.models.SKILL_LEVELS`` and the backend's target levels.
ModuleLevel = Literal["beginner", "intermediate", "advanced", "expert"]

# Mirrors the backend's ``ModulePageKind``. ``VERIFY`` is deliberately absent:
# the gating check travels as ``ProposedModule.verification`` and the backend
# renders its page itself, so asking the model for one too would produce two
# sources of truth for the same gate.
ModulePageKind = Literal[
    "CONTEXT",
    "LESSON",
    "WALKTHROUGH",
    "TASK",
    "RESOURCE",
    "CHECK",
]

# Kinds that make factual claims about the codebase, and therefore must cite the
# evidence they came from. ``TASK`` and ``CHECK`` are exercises derived from the
# pages above them, not claims of their own, so they carry no citation
# requirement — demanding one would only teach the model to attach a chunk id it
# didn't use.
GROUNDED_PAGE_KINDS: frozenset[str] = frozenset(
    {"CONTEXT", "LESSON", "WALKTHROUGH", "RESOURCE"}
)

ModuleStatus = Literal["proposed", "unchanged", "skipped"]

VerificationType = Literal["KNOWLEDGE", "EXACT", "ATTEST", "ARTIFACT"]


class ProposedModulePage(BaseModel):
    """One page of a proposed module, in render order."""

    kind: ModulePageKind
    title: str
    body: str = Field(description="Page body, markdown.")
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class ProposedModuleVerification(BaseModel):
    """The module's gating check.

    Part of the proposal because the check belongs to the module, not to a
    per-user step. The backend still holds it unapproved until a PM says so.
    """

    type: VerificationType = "KNOWLEDGE"
    prompt: str
    rubric: str | None = None


class ProposedModule(BaseModel):
    """A shared module for one competency at one target level."""

    competency_key: str
    level: ModuleLevel
    title: str
    summary: str = ""
    pages: list[ProposedModulePage] = Field(default_factory=list[ProposedModulePage])
    verification: ProposedModuleVerification | None = None


class ModuleProvenance(BaseModel):
    """Why a module looks the way it does; mirrors ``LessonProvenance``."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class ModuleOutcome(BaseModel):
    """Result of one module proposal run."""

    status: ModuleStatus
    module: ProposedModule | None = None
    provenance: ModuleProvenance | None = None
    chunks_retrieved: int = 0
    pages_dropped: int = 0
    notes: list[str] = Field(default_factory=list[str])
