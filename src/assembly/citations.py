"""Resolve retrieved chunk ids into onboarding citation references."""

from collections.abc import Mapping

from assembly.models import CitationRef
from corpus.types import ScoredChunk


def resolve_citations(
    chunk_ids: list[str], chunks_by_id: Mapping[str, ScoredChunk]
) -> list[CitationRef]:
    """Map cited chunk ids to :class:`CitationRef`s, deduped and order-preserving.

    Unknown ids (the LLM citing a chunk that wasn't in the evidence) are skipped
    so an invented source never produces a dangling citation.
    """
    refs: list[CitationRef] = []
    seen: set[str] = set()
    for cid in chunk_ids:
        chunk = chunks_by_id.get(cid)
        if chunk is not None and chunk.id not in seen:
            refs.append(
                CitationRef(
                    filename=chunk.filename,
                    chunk_id=chunk.id,
                    source_url=chunk.source_url,
                )
            )
            seen.add(chunk.id)
    return refs
