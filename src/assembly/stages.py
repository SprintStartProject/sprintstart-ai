"""The stages the grounded assembly pipelines share.

Orientation, phase and diagram assembly run one pipeline — gate on the corpus
fingerprint, retrieve evidence, generate with one correction round, resolve
against the evidence, emit — and differ only in what they search for, how they
prompt and what they keep. The stages that do not differ live here, so a fix to
one of them reaches every pipeline.
"""

import logging
from collections.abc import Callable, Generator, Iterable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from assembly.models import CitationRef
from assembly.progress import ProgressEvent, ProgressStream
from assembly.similarity import OVERLAP_THRESHOLD, text_overlap
from corpus.source_role import GROUNDING_EXCLUDED_ROLES
from corpus.types import RetrievalFilters, ScoredChunk
from llm.base import LLMClient, Message
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from store.base import VectorStore

_MIN_SCORE = 0.3

# Re-running against an unchanged corpus must give materially the same result: a
# hire who reloads the page and reads different instructions, or sees a different
# picture, has no reason to trust either version.
_TEMPERATURE = 0.0

# A small local model breaks the JSON now and then; asked back for the same
# content, it usually fixes its own syntax.
_MAX_GENERATION_ATTEMPTS = 2


class AssemblyError(ValueError):
    """Raised when generated output cannot be parsed or validated."""


@dataclass(frozen=True)
class Search:
    """One retrieval of a run: what the watcher is told, and what is searched for."""

    label: str
    question: str


@dataclass(frozen=True)
class Retrieval:
    """What one run searches for, how much each search keeps, and where it looks."""

    searches: Sequence[Search]
    top_k: int
    filters: RetrievalFilters


@dataclass(frozen=True)
class Generation[P]:
    """How one run reads the model's reply, asks for a fix, and logs a failure.

    ``failure`` is the log format for a failed attempt; it receives the attempt
    number, ``subject`` and the error.
    """

    parse: Callable[[str], P]
    correct: Callable[[list[Message], str, AssemblyError], list[Message]]
    logger: logging.Logger
    failure: str
    subject: str


def collapse_duplicates(chunks: list[ScoredChunk]) -> tuple[list[ScoredChunk], int]:
    """Drop chunks that restate one already kept, best-scoring first.

    The README and the wiki repeating each other is load the model pays for
    twice. Comparison is token overlap rather than embeddings: it needs no extra
    call, and an assembly must be reproducible.
    """
    kept: list[ScoredChunk] = []
    collapsed = 0
    for chunk in sorted(chunks, key=lambda c: (-c.score, c.id)):
        if any(text_overlap(chunk.text, k.text) > OVERLAP_THRESHOLD for k in kept):
            collapsed += 1
            continue
        kept.append(chunk)
    return kept, collapsed


def evidence_line(chunk: ScoredChunk) -> str:
    """One chunk as the model sees it: id to cite, file, kind, text."""
    meta = chunk.artifact_type or "FILE"
    if chunk.language:
        meta += f"/{chunk.language}"
    return f"  [{chunk.id}] ({chunk.filename} | {meta}) {chunk.text}"


def retrieve_evidence(
    progress: ProgressStream,
    llm: LLMClient,
    store: VectorStore,
    retrieval: Retrieval,
) -> Generator[ProgressEvent, None, tuple[list[ScoredChunk], int]]:
    """Run each search, announced as a stage, and pool the results.

    A chunk found by two searches keeps its better score and appears once. The
    pool is deduplicated before it is returned, with the number collapsed.
    """
    bm25_cache = BM25IndexCache()
    by_id: dict[str, ScoredChunk] = {}
    for search in retrieval.searches:
        yield progress.stage("retrieving", search.label)
        for chunk in hybrid_retrieve(
            question=search.question,
            llm=llm,
            store=store,
            top_k=retrieval.top_k,
            min_score=_MIN_SCORE,
            bm25_cache=bm25_cache,
            exclude_roles=GROUNDING_EXCLUDED_ROLES,
            filters=retrieval.filters,
        ):
            existing = by_id.get(chunk.id)
            if existing is None or chunk.score > existing.score:
                by_id[chunk.id] = chunk
    return collapse_duplicates(list(by_id.values()))


def generate_payload[P](
    progress: ProgressStream,
    llm: LLMClient,
    messages: list[Message],
    generation: Generation[P],
) -> Generator[ProgressEvent, None, P | AssemblyError]:
    """Generate and parse, asking once for a corrected reply when parsing fails.

    Returns the parsed payload, or the last error when no attempt could be read.
    """
    error: AssemblyError | None = None
    for attempt in range(_MAX_GENERATION_ATTEMPTS):
        raw = llm.generate(messages, temperature=_TEMPERATURE)
        try:
            return generation.parse(raw)
        except AssemblyError as exc:
            error = exc
            generation.logger.warning(
                generation.failure, attempt + 1, generation.subject, exc
            )
            if attempt + 1 < _MAX_GENERATION_ATTEMPTS:
                yield progress.stage("generating", "Correcting invalid generated JSON")
                messages = generation.correct(messages, raw, exc)
    assert error is not None
    return error


def give_up[T: BaseModel](
    progress: ProgressStream, outcome: T, *, warning: str, done: str
) -> Generator[ProgressEvent, None, T]:
    """End a run without a result: say why, then finish with the skipped outcome."""
    yield progress.warning(warning)
    yield progress.done(done, dump(outcome))
    return outcome


def dump(outcome: BaseModel) -> dict[str, object]:
    """The outcome as a JSON-safe dict for a ``done`` event's ``result``."""
    return outcome.model_dump(mode="json")


def sources_drawn_on[S: BaseModel](
    citations: Iterable[CitationRef], chunks: list[ScoredChunk], make: Callable[..., S]
) -> list[S]:
    """The distinct material ``citations`` point at, in first-use order."""
    chunks_by_id = {c.id: c for c in chunks}
    sources: list[S] = []
    seen: set[tuple[str, str | None]] = set()
    for citation in citations:
        key = (citation.filename, citation.source_url)
        if key in seen:
            continue
        seen.add(key)
        chunk = chunks_by_id.get(citation.chunk_id)
        sources.append(
            make(
                filename=citation.filename,
                source_url=citation.source_url,
                artifact_type=chunk.artifact_type if chunk else None,
            )
        )
    return sources
