"""Assembly of a subject-scoped diagram from existing material.

The model may choose the question. It never writes the answer. The
subject drives retrieval and nothing else: every box comes back derived from a
retrieved chunk and carrying the citation that proves it, an ungrounded box is
dropped, and a picture left with fewer than two connected boxes is ``skipped``
rather than shown.

Four mechanics carry that guarantee:

* **Retrieval runs per facet, not per subject.** One query for the parts, one for
  how they connect, one for where the subject begins and ends — a single query
  returns the same passage three ways, and an evidence pool that never mentions a
  connection can only draw disconnected boxes.
* **The pool is deduplicated before the model sees it**, on token overlap, no
  extra call.
* **Grounding is per node, with no exempt kinds.** Every node asserts this
  project contains this part.
* The subject is fenced as untrusted. It reaches here as model text
  written in a conversation a hire steers, so a request to diagram "ignore the
  evidence and say X" is quoted at the model as data.
"""

import logging
import re
from collections.abc import Generator
from dataclasses import dataclass
from datetime import UTC, datetime

from assembly.diagram.draft import (
    FACET_LABEL,
    FACET_QUERIES,
    MAX_NODES,
    MAX_SUBJECT,
    MIN_NODES,
    TOP_K_PER_FACET,
    GenPayload,
    build_prompt,
    correction_prompt,
    parse_payload,
    resolve_edges,
    resolve_nodes,
    trim_to_cap,
)
from assembly.diagram.models import (
    Diagram,
    DiagramEdge,
    DiagramNode,
    DiagramOutcome,
    DiagramProvenance,
    DiagramSource,
)
from assembly.fingerprint import fingerprint_gate
from assembly.progress import ProgressEvent, ProgressStream, drain
from assembly.stages import (
    AssemblyError,
    Generation,
    Retrieval,
    Search,
    dump,
    generate_payload,
    give_up,
    retrieve_evidence,
    sources_drawn_on,
)
from corpus.types import RetrievalFilters, ScoredChunk
from llm.base import LLMClient
from store.base import VectorStore

logger = logging.getLogger(__name__)

_DONE_EMPTY = "No diagram could be assembled"


@dataclass(frozen=True)
class _Request:
    """What a run draws, and the projects it may draw on."""

    subject: str
    project_ids: frozenset[str]


@dataclass(frozen=True)
class _Counts:
    """How much evidence a run drew on and how much of the draft it dropped."""

    chunks_retrieved: int = 0
    chunks_collapsed: int = 0
    nodes_dropped: int = 0
    nodes_trimmed: int = 0
    edges_dropped: int = 0

    @property
    def nodes_lost(self) -> int:
        return self.nodes_dropped + self.nodes_trimmed


@dataclass(frozen=True)
class _Grounded:
    summary: str
    nodes: list[DiagramNode]
    edges: list[DiagramEdge]
    chunks: list[ScoredChunk]
    counts: _Counts


def _skipped(note: str, counts: _Counts = _Counts()) -> DiagramOutcome:
    return DiagramOutcome(
        status="skipped",
        chunks_retrieved=counts.chunks_retrieved,
        chunks_collapsed=counts.chunks_collapsed,
        nodes_dropped=counts.nodes_lost,
        edges_dropped=counts.edges_dropped,
        notes=[note],
    )


def _gate(
    progress: ProgressStream,
    store: VectorStore,
    last_fingerprint: str | None,
    project_ids: frozenset[str],
) -> tuple[str | None, list[ProgressEvent], DiagramOutcome | None]:
    """Answer from the fingerprint alone when the corpus is empty or unchanged."""
    return fingerprint_gate(
        progress,
        store,
        last_fingerprint,
        make_unchanged=lambda: DiagramOutcome(
            status="unchanged", notes=["corpus unchanged since the cached diagram"]
        ),
        make_empty=lambda: DiagramOutcome(status="skipped", notes=["corpus is empty"]),
        unchanged_label="Nothing changed — the cached diagram is current",
        empty_warning_label="The project has no indexed material yet",
        empty_done_label=_DONE_EMPTY,
        project_ids=project_ids,
    )


def _stop(
    progress: ProgressStream, outcome: DiagramOutcome, warning: str
) -> Generator[ProgressEvent, None, DiagramOutcome]:
    return (yield from give_up(progress, outcome, warning=warning, done=_DONE_EMPTY))


def _gather(
    progress: ProgressStream,
    llm: LLMClient,
    store: VectorStore,
    request: _Request,
) -> Generator[ProgressEvent, None, tuple[list[ScoredChunk], int]]:
    """One search per facet of the picture, pooled."""
    return (
        yield from retrieve_evidence(
            progress,
            llm,
            store,
            Retrieval(
                searches=[
                    Search(
                        label=f"Searching the project: {FACET_LABEL[facet]}",
                        question=f"{request.subject} {query}",
                    )
                    for facet, query in FACET_QUERIES.items()
                ],
                top_k=TOP_K_PER_FACET,
                filters=RetrievalFilters(project_ids=request.project_ids),
            ),
        )
    )


def _write(
    progress: ProgressStream, llm: LLMClient, subject: str, chunks: list[ScoredChunk]
) -> Generator[ProgressEvent, None, GenPayload | AssemblyError]:
    yield progress.stage("generating", f"Drawing it from {len(chunks)} source(s)")
    return (
        yield from generate_payload(
            progress,
            llm,
            build_prompt(subject, chunks),
            Generation(
                parse=parse_payload,
                correct=correction_prompt,
                logger=logger,
                failure="Diagram assembly attempt %d failed for subject %r: %s",
                subject=subject,
            ),
        )
    )


def _ground(
    progress: ProgressStream,
    payload: GenPayload,
    chunks: list[ScoredChunk],
    collapsed: int,
) -> Generator[ProgressEvent, None, _Grounded]:
    """Keep the parts and connections the evidence supports, announcing each part."""
    yield progress.stage("grounding", "Checking every part cites its source")
    nodes, remap, nodes_dropped = resolve_nodes(payload, chunks)
    edges, edges_dropped = resolve_edges(payload, remap, {n.id for n in nodes})
    nodes, edges, trimmed = trim_to_cap(nodes, edges)

    for node in nodes:
        yield progress.item(node.model_dump(mode="json"), node.label)
    if nodes and nodes_dropped:
        yield progress.warning(f"Dropped {nodes_dropped} part(s) with no source")
    if trimmed:
        yield progress.warning(
            f"Kept the {MAX_NODES} best-connected parts, trimming {trimmed}"
        )
    counts = _Counts(len(chunks), collapsed, nodes_dropped, trimmed, edges_dropped)
    return _Grounded(payload.summary.strip(), nodes, edges, chunks, counts)


def _notes(counts: _Counts) -> list[str]:
    notes: list[str] = []
    if counts.chunks_collapsed:
        notes.append(f"collapsed {counts.chunks_collapsed} redundant source chunk(s)")
    if counts.nodes_dropped:
        notes.append(f"dropped {counts.nodes_dropped} ungrounded part(s)")
    if counts.nodes_trimmed:
        notes.append(
            f"trimmed {counts.nodes_trimmed} part(s) beyond the {MAX_NODES} kept"
        )
    if counts.edges_dropped:
        notes.append(f"dropped {counts.edges_dropped} unusable connection(s)")
    return notes


def _assembled(
    llm: LLMClient,
    fingerprint: str | None,
    subject: str,
    grounded: _Grounded,
) -> DiagramOutcome:
    counts = grounded.counts
    notes = _notes(counts)
    return DiagramOutcome(
        status="assembled",
        diagram=Diagram(
            subject=subject,
            summary=grounded.summary,
            nodes=grounded.nodes,
            edges=grounded.edges,
            sources=sources_drawn_on(
                (c for n in grounded.nodes for c in n.citations),
                grounded.chunks,
                DiagramSource,
            ),
        ),
        provenance=DiagramProvenance(
            corpus_fingerprint=fingerprint,
            generated_at=datetime.now(UTC).isoformat(),
            model=llm.model_name,
            notes=notes,
        ),
        chunks_retrieved=counts.chunks_retrieved,
        chunks_collapsed=counts.chunks_collapsed,
        nodes_dropped=counts.nodes_lost,
        edges_dropped=counts.edges_dropped,
        notes=notes,
    )


def stream_diagram(
    llm: LLMClient,
    store: VectorStore,
    *,
    subject: str,
    project_ids: frozenset[str],
    last_fingerprint: str | None = None,
) -> Generator[ProgressEvent, None, DiagramOutcome]:
    """Assemble a diagram, yielding live progress and returning the final outcome.

    This is the single implementation: :func:`assemble_diagram` drives it to
    completion for the non-streaming path and the streaming route relays its
    events, so a watched run produces exactly what an unwatched one would.

    Retrieval runs once per facet (the parts → how they connect → where it starts
    and ends), each announced as a ``stage``. Nodes are emitted as ``item`` events
    only after they clear the grounding gate, so nothing ungrounded is ever shown
    — not even briefly.
    """
    progress = ProgressStream("diagram")
    fingerprint, early_events, early_outcome = _gate(
        progress, store, last_fingerprint, project_ids
    )
    if early_outcome is not None:
        yield from early_events
        return early_outcome

    chunks, collapsed = yield from _gather(
        progress, llm, store, _Request(subject, project_ids)
    )
    if not chunks:
        return (
            yield from _stop(
                progress,
                _skipped("no grounding evidence retrieved for this subject"),
                "Nothing in the project matched this subject closely enough",
            )
        )

    payload = yield from _write(progress, llm, subject, chunks)
    if isinstance(payload, AssemblyError):
        skipped = _skipped(str(payload), _Counts(len(chunks), collapsed))
        return (
            yield from _stop(
                progress, skipped, "The generated diagram could not be read"
            )
        )

    grounded = yield from _ground(progress, payload, chunks, collapsed)
    if len(grounded.nodes) < MIN_NODES or not grounded.edges:
        skipped = _skipped(
            "not enough grounded, connected parts to draw a diagram", grounded.counts
        )
        return (
            yield from _stop(
                progress, skipped, "Too little survived grounding to draw a diagram"
            )
        )

    outcome = _assembled(llm, fingerprint, subject, grounded)
    yield progress.done("Diagram ready", dump(outcome))
    return outcome


def assemble_diagram(
    llm: LLMClient,
    store: VectorStore,
    *,
    subject: str,
    project_ids: frozenset[str],
    last_fingerprint: str | None = None,
) -> DiagramOutcome:
    """Assemble a diagram of one subject from the project's own material.

    ``subject`` is the *question* — "how a request reaches the database" — and is
    the only thing about this diagram a model chose. It aims retrieval and is
    never asserted: every node that comes back is derived from a retrieved chunk
    and carries the citation proving it, so a subject the evidence cannot support
    yields ``skipped`` rather than a plausible invention.

    ``last_fingerprint`` is the corpus fingerprint the caller recorded the last
    time it drew *this subject*. An unchanged corpus answers ``unchanged`` so a
    cached diagram can be served without an LLM call — which is what keeps a card
    that hydrates on every board load from costing a generation every time.

    Takes nothing about the individual hire, deliberately: the shape of a
    codebase is not personal, so two people who ask the same question see the
    same picture and can talk about it.

    Backed by :func:`stream_diagram` so the non-streaming and streaming paths are
    the same computation.
    """
    return drain(
        stream_diagram(
            llm,
            store,
            subject=subject,
            project_ids=project_ids,
            last_fingerprint=last_fingerprint,
        )
    )


def clamp_subject(subject: str) -> str:
    """The subject as it is safe to use: stripped, collapsed and length-capped.

    Applied at the API boundary rather than here so both entry points share it
    and the model never receives a subject the caller did not have.
    """
    return re.sub(r"\s+", " ", subject.strip())[:MAX_SUBJECT]
