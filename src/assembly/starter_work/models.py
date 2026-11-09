"""The result types of a starter-work mining run."""

from typing import Literal

from pydantic import BaseModel, Field

from assembly.models import CitationRef

ProposalStatus = Literal["proposed", "unchanged", "skipped"]


class GenerationError(Exception):
    """Raised when the LLM output for a starter-work proposal can't be parsed."""


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
