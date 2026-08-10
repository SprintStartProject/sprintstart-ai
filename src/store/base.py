from typing import Protocol

from ingestion.source_role import SourceRole
from rag.source_filter import SourceExclusions
from rag.types import Chunk, RetrievalFilters, ScoredChunk


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk]) -> None: ...

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
        filters: RetrievalFilters | None = None,
        exclude_roles: frozenset[SourceRole] = frozenset(),
        exclusions: SourceExclusions = SourceExclusions(),
    ) -> list[ScoredChunk]:
        """The ``top_k`` best-scoring chunks that satisfy every constraint.

        All of ``filters``, ``exclude_roles`` and ``exclusions`` must be applied
        *before* the result set is limited to ``top_k``. An implementation that
        cannot push a constraint down to its backend must still not return an
        ineligible chunk; callers treat a short result as "the corpus is
        exhausted at this score floor" and stop widening their search.
        """
        ...

    def delete(
        self,
        artifact_id: str,
        exclude_ids: list[str] | None = None,
    ) -> int: ...

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]: ...

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]: ...

    def count_by_artifact(self, artifact_id: str) -> int: ...

    def all_chunks(self) -> list[Chunk]: ...

    def all_chunks_without_embeddings(self) -> list[Chunk]: ...

    def list_chunks_without_embeddings(
        self, limit: int, offset: int = 0
    ) -> list[Chunk]: ...

    def all_ids(self) -> frozenset[str]: ...

    def retrieval_fingerprints(self) -> frozenset[str]: ...

    def project_ids_for_artifact(self, artifact_id: str) -> frozenset[str]: ...

    def count(self) -> int: ...
