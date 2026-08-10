"""Domain models for task-scoped orientation packets.

An orientation packet is what a hire reads *while doing one task*: where the
code lives, which conventions apply, what "done" looks like here, what to run
before pushing. It is assembled from material that already exists in the
project's corpus — not authored.

Three properties separate a packet from a :mod:`onboarding.module_models`
module, and every one of them is deliberate:

* **Scoped to a task, not a competency.** A packet answers "how do I do *this*",
  so it dies with the task it was made for.
* **Disposable.** Nothing here is versioned, reviewed or approved. Regenerating
  is cheaper than maintaining, so no PM stands between a hire and their
  orientation.
* **Nothing ships without a source.** A module page may carry uncited exercises
  (``TASK``/``CHECK``) because those are prompts, not claims. A packet has no
  such exemption: every section states something about *this* codebase, so a
  section that cites nothing is dropped. That hard rule is what makes this
  assembly rather than generation.
"""

from typing import Literal

from pydantic import BaseModel, Field

from onboarding.models import CitationRef

# The path to a first pull request, segmented by *step* rather than by topic.
# Segmentation by the process somebody is actually walking is the mechanism the
# research credits for lower cognitive load — a hire on day three opens "check
# locally" and does not re-read setup.
OrientationStep = Literal[
    "SET_UP",
    "FIND_THE_CODE",
    "MAKE_THE_CHANGE",
    "CHECK_LOCALLY",
    "OPEN_THE_PR",
]

# Render order. Sections are sorted into it, so a model returning them
# out of order cannot hand a hire "open the PR" before "find the code".
STEP_ORDER: tuple[OrientationStep, ...] = (
    "SET_UP",
    "FIND_THE_CODE",
    "MAKE_THE_CHANGE",
    "CHECK_LOCALLY",
    "OPEN_THE_PR",
)

OrientationStatus = Literal["assembled", "unchanged", "skipped"]


class OrientationSource(BaseModel):
    """One piece of existing material the packet drew on.

    Listed on the packet itself so a hire can see the ground it stands on even
    when a section they wanted was dropped, and so "this is out of date" has
    somewhere to point.
    """

    filename: str
    source_url: str | None = None
    artifact_type: str | None = None


class OrientationSection(BaseModel):
    """One segment of the packet, belonging to exactly one step."""

    step: OrientationStep
    title: str
    body: str = Field(description="Section body, markdown.")
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class OrientationPacket(BaseModel):
    """Assembled orientation for one task."""

    task_title: str
    summary: str = ""
    sections: list[OrientationSection] = Field(default_factory=list[OrientationSection])
    sources: list[OrientationSource] = Field(default_factory=list[OrientationSource])


class OrientationProvenance(BaseModel):
    """Why a packet looks the way it does; mirrors ``ModuleProvenance``."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class OrientationOutcome(BaseModel):
    """Result of one assembly run.

    ``skipped`` is a real answer and the client must be able to show it as one:
    an empty corpus, no retrieved evidence or a packet whose every section was
    ungrounded all yield ``skipped`` with ``packet=None``, never an empty packet
    dressed up as guidance.
    """

    status: OrientationStatus
    packet: OrientationPacket | None = None
    provenance: OrientationProvenance | None = None
    chunks_retrieved: int = 0
    chunks_collapsed: int = 0
    sections_dropped: int = 0
    notes: list[str] = Field(default_factory=list[str])
