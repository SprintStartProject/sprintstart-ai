"""Domain models for AI-assembled onboarding-phase content.

A blueprint phase carries an authored prompt (``aiPrompt``) instead of a fixed
step list. Assembly fills it: actionable steps (with tasks and resources) and a
small grounded knowledge check, all derived from the project's own corpus. The
consuming backend persists the result onto the user's onboarding path.
"""

from typing import Literal

from pydantic import BaseModel, Field

from onboarding.models import CheckQuestion

PhaseAssembleStatus = Literal["assembled", "unchanged", "skipped"]


class PhaseTask(BaseModel):
    """One actionable sub-task within a phase step."""

    title: str
    description: str = ""


class PhaseResource(BaseModel):
    """A concrete pointer the step can link out to; url comes from the evidence."""

    title: str
    url: str


class PhaseStep(BaseModel):
    """One step of the assembled phase, grounded in the project's corpus.

    ``expected_outcome`` states what a newcomer should be able to do afterwards;
    ``estimated_minutes`` is a rough time budget. Neither is invented: both are
    written to reflect the evidence the step cites.

    ``key`` and ``blocked_by`` express optional dependency edges between the
    assembled items: ``key`` is a short unique handle assigned by the model and
    ``blocked_by`` lists the keys of steps/questions that must be completed
    first. Both stay empty when there is no genuine dependency.
    """

    title: str
    description: str = ""
    tasks: list[PhaseTask] = Field(default_factory=list[PhaseTask])
    resources: list[PhaseResource] = Field(default_factory=list[PhaseResource])
    estimated_minutes: int | None = None
    expected_outcome: str = ""
    key: str = ""
    blocked_by: list[str] = Field(default_factory=list[str])


class PhaseProvenance(BaseModel):
    """Why a phase's assembled content looks the way it does.

    ``corpus_fingerprint`` ties the result to the exact corpus state it was
    drafted from, which is what makes re-assembly idempotent: an unchanged
    corpus is answered without retrieval or generation.
    """

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class PhaseOutcome(BaseModel):
    """Result of one phase-content assembly run.

    ``skipped`` is a real answer and the backend must persist it as an honest
    empty phase: an empty corpus, no retrieved evidence, or a generation whose
    every step was ungrounded all yield ``skipped`` with no content — never
    fabricated steps dressed up as guidance. Check questions reuse the shared
    :class:`CheckQuestion` model (``position`` assigned over survivors in
    assembly order) so the backend's phase has no field translation to do.
    """

    status: PhaseAssembleStatus
    steps: list[PhaseStep] = Field(default_factory=list[PhaseStep])
    check_questions: list[CheckQuestion] = Field(default_factory=list[CheckQuestion])
    provenance: PhaseProvenance | None = None
    chunks_retrieved: int = 0
    chunks_collapsed: int = 0
    steps_dropped: int = 0
    questions_dropped: int = 0
    notes: list[str] = Field(default_factory=list[str])
