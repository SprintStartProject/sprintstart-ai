"""Domain models for subject-scoped diagrams.

A diagram is what a mentor draws on the whiteboard beside the conversation: the
handful of parts a subject is made of, and the arrows between them. It is
assembled from the project's own corpus, never authored — the same guarantee an
orientation packet makes, applied to a picture.

The rule the board's card catalogue exists to protect is that a model never
writes the content of a card. A diagram extends that rule rather than breaking
it, and the extension is worth stating precisely:

    **The model may choose the question. It never writes the answer.**

The subject — "how a request reaches the database" — is the mentor's, because
only the conversation knows what was just being discussed. Everything the hire
then *reads* is derived: each node names something a retrieved chunk actually
evidences, carries the citation proving it, and is dropped if it cannot. So a
subject the model invented cannot become a claim the model invented.

Three properties separate a diagram from an :mod:`onboarding.orientation_models`
packet:

* **Scoped to a subject, not a task.** It outlives the task somebody claimed and
  answers "what is this shaped like", not "how do I do this".
* **Structural, not prose.** Nodes and edges, so the reader can point at a part
  and see what it connects to. A wall of text would have said it worse.
* **Nothing is stored but the question.** The board row holds the subject; the
  picture is re-derived on every read, so a diagram cannot describe code that
  has since moved.
"""

from typing import Literal

from pydantic import BaseModel, Field

from onboarding.models import CitationRef

# What a node *is*. Small on purpose: a catalogue a model has to choose from is
# a catalogue it applies consistently, and every extra kind is one more axis on
# which two runs of the same subject can disagree.
#
# ``OTHER`` is the honest fallback, not a dumping ground -- the same rule
# starter-work matching already keeps for an unlabelled issue. A node whose kind
# the model got wrong still names a real thing and still cites its source; kinds
# shape how it is drawn, never whether it is true.
DiagramNodeKind = Literal[
    "COMPONENT",
    "FILE",
    "SERVICE",
    "DATA",
    "STEP",
    "EXTERNAL",
    "OTHER",
]

NODE_KINDS: frozenset[str] = frozenset(
    (
        "COMPONENT",
        "FILE",
        "SERVICE",
        "DATA",
        "STEP",
        "EXTERNAL",
        "OTHER",
    )
)

# How two nodes relate. ``RELATES_TO`` is deliberately weak and deliberately
# available: forcing a model to pick between "calls" and "depends on" when the
# evidence supports neither buys a confident-sounding arrow that is wrong.
DiagramEdgeKind = Literal[
    "FLOWS_TO",
    "DEPENDS_ON",
    "CONTAINS",
    "RELATES_TO",
]

EDGE_KINDS: frozenset[str] = frozenset(
    ("FLOWS_TO", "DEPENDS_ON", "CONTAINS", "RELATES_TO")
)

DiagramStatus = Literal["assembled", "unchanged", "skipped"]


class DiagramNode(BaseModel):
    """One box: something this project has, that the corpus can be shown to have.

    ``citations`` is never empty on a node that survives assembly. A node is an
    assertion that a part exists and is called this; an uncited one is the model
    remembering some other codebase.
    """

    id: str
    label: str
    kind: DiagramNodeKind = "OTHER"
    summary: str = Field(
        default="", description="One line on what this part is. Optional, never prose."
    )
    citations: list[CitationRef] = Field(default_factory=list[CitationRef])


class DiagramEdge(BaseModel):
    """One arrow. Both endpoints are ids of nodes that survived assembly."""

    from_id: str
    to_id: str
    kind: DiagramEdgeKind = "RELATES_TO"
    label: str = ""


class DiagramSource(BaseModel):
    """One piece of existing material the diagram drew on.

    Listed on the diagram itself for the same reason a packet lists them: a
    reader who thinks a box is wrong needs somewhere to point, and a reader who
    wanted a part that was dropped can still see the ground the rest stands on.
    """

    filename: str
    source_url: str | None = None
    artifact_type: str | None = None


class Diagram(BaseModel):
    """An assembled picture of one subject."""

    subject: str
    summary: str = ""
    nodes: list[DiagramNode] = Field(default_factory=list[DiagramNode])
    edges: list[DiagramEdge] = Field(default_factory=list[DiagramEdge])
    sources: list[DiagramSource] = Field(default_factory=list[DiagramSource])


class DiagramProvenance(BaseModel):
    """Why a diagram looks the way it does; mirrors ``OrientationProvenance``."""

    corpus_fingerprint: str | None = None
    generated_at: str | None = None
    model: str | None = None
    notes: list[str] = Field(default_factory=list[str])


class DiagramOutcome(BaseModel):
    """Result of one assembly run.

    ``skipped`` is a real answer and the caller must render it as one. An empty
    corpus, no retrieved evidence, an unreadable generation, or a picture left
    with fewer than two connected nodes all yield ``skipped`` with
    ``diagram=None`` — never an empty diagram dressed up as an explanation.
    """

    status: DiagramStatus
    diagram: Diagram | None = None
    provenance: DiagramProvenance | None = None
    chunks_retrieved: int = 0
    chunks_collapsed: int = 0
    nodes_dropped: int = 0
    edges_dropped: int = 0
    notes: list[str] = Field(default_factory=list[str])
