from dataclasses import dataclass

from corpus.types import Chunk, ScoredChunk


@dataclass(frozen=True)
class SourceExclusions:
    """Connectors/sources to drop from retrieval results.

    ``sources`` entries are ``(connector_id, connector_source_id)`` pairs, e.g.
    ``("github", "owner/repo")``. A connector listed in ``connectors`` excludes
    all of its sources regardless of ``sources``.
    """

    connectors: frozenset[str] = frozenset()
    sources: frozenset[tuple[str, str]] = frozenset()

    def __bool__(self) -> bool:
        return bool(self.connectors or self.sources)


def is_excluded(chunk: Chunk | ScoredChunk, exclusions: SourceExclusions) -> bool:
    """Whether ``chunk`` belongs to a disabled connector/source.

    Chunks with no ``connector_id`` (legacy chunks ingested before this field
    existed, or manually-uploaded docs) are never excluded.
    """
    if not exclusions or chunk.connector_id is None:
        return False

    if chunk.connector_id in exclusions.connectors:
        return True

    return (
        chunk.connector_source_id is not None
        and (chunk.connector_id, chunk.connector_source_id) in exclusions.sources
    )
