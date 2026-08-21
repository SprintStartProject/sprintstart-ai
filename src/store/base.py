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

    def set_project_ids_for_artifact(
        self,
        artifact_id: str,
        project_ids: tuple[str, ...],
    ) -> int:
        """Replace the project membership of every chunk of one artifact.

        The backend owns ``artifact_projects`` and sends the resulting set, so
        this is a set-operation rather than an add/remove delta: re-sending the
        same membership is a no-op, and no sequence of calls can leave the store
        holding a project the backend has already dropped.

        Returns the number of chunks rewritten (0 when the artifact is unknown).
        """
        ...

    def remove_project(self, project_id: str) -> int:
        """Drop one project from every chunk that carries it.

        Used when a project is deleted: its artifacts survive for the other
        projects they belong to, they just stop being reachable from this one.

        Returns the number of chunks rewritten.
        """
        ...

    def count(self) -> int: ...
