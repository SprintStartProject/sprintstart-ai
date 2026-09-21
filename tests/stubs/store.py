from collections.abc import Iterator
from dataclasses import replace

from ingestion.source_role import SourceRole
from rag.filters import encode_project_ids, matches_retrieval_filters
from rag.source_filter import SourceExclusions, is_excluded
from rag.types import Chunk, RetrievalFilters, ScoredChunk


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot / (norm_a * norm_b)


class StubVectorStore:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []
        self._corpus_revision = 0
        self._revoked_artifact_ids: set[str] = set()

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        new_ids = {c.id for c in chunks}
        self.chunks = [c for c in self.chunks if c.id not in new_ids] + chunks
        self._corpus_revision += 1

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
        filters: RetrievalFilters | None = None,
        exclude_roles: frozenset[SourceRole] = frozenset(),
        exclusions: SourceExclusions = SourceExclusions(),
    ) -> list[ScoredChunk]:
        """Like Chroma: every constraint is applied before the ``top_k`` cutoff."""
        scored: list[ScoredChunk] = []

        for chunk in self.chunks:
            if chunk.artifact_id in self._revoked_artifact_ids:
                continue

            if not matches_retrieval_filters(chunk, filters):
                continue

            if exclude_roles and chunk.source_role in exclude_roles:
                continue

            if is_excluded(chunk, exclusions):
                continue

            score = cosine_similarity(embedding, chunk.embedding)

            if score < min_score:
                continue

            scored.append(
                ScoredChunk(
                    id=chunk.id,
                    artifact_id=chunk.artifact_id,
                    filename=chunk.filename,
                    position=chunk.position,
                    kind=chunk.kind,
                    text=chunk.text,
                    score=score,
                    source_role=chunk.source_role,
                    source_url=chunk.source_url,
                    artifact_type=chunk.artifact_type,
                    language=chunk.language,
                    connector_id=chunk.connector_id,
                    connector_source_id=chunk.connector_source_id,
                    source_system=chunk.source_system,
                    created_at=chunk.created_at,
                    start_line=chunk.start_line,
                    start_page=chunk.start_page,
                    project_ids=chunk.project_ids,
                )
            )

        return sorted(scored, key=lambda item: item.score, reverse=True)[:top_k]

    def delete(self, artifact_id: str, exclude_ids: list[str] | None = None) -> int:
        if artifact_id not in self._revoked_artifact_ids:
            self._revoked_artifact_ids.add(artifact_id)
            self._corpus_revision += 1

        before = len(self.chunks)
        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.artifact_id != artifact_id or chunk.id in (exclude_ids or [])
        ]
        deleted = before - len(self.chunks)
        if deleted:
            self._corpus_revision += 1

        self._revoked_artifact_ids.discard(artifact_id)
        self._corpus_revision += 1
        return deleted

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]:
        page = self.chunks[offset : offset + limit]
        return [
            chunk
            for chunk in page
            if chunk.artifact_id not in self._revoked_artifact_ids
        ]

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]:
        if artifact_id in self._revoked_artifact_ids:
            return []
        matching = [chunk for chunk in self.chunks if chunk.artifact_id == artifact_id]
        return matching[offset : offset + limit]

    def list_chunks_by_positions(
        self,
        artifact_id: str,
        positions: frozenset[int],
        start_page: int | None = None,
    ) -> list[Chunk]:
        return sorted(
            (
                replace(chunk, embedding=[])
                for chunk in self.chunks
                if chunk.artifact_id == artifact_id
                and chunk.position in positions
                and (start_page is None or chunk.start_page == start_page)
            ),
            key=lambda chunk: (
                chunk.start_page if chunk.start_page is not None else 0,
                chunk.position if chunk.position is not None else -1,
            ),
        )

    def count_by_artifact(self, artifact_id: str) -> int:
        if artifact_id in self._revoked_artifact_ids:
            return 0
        return sum(1 for chunk in self.chunks if chunk.artifact_id == artifact_id)

    def all_chunks_without_embeddings(self) -> list[Chunk]:
        return list(self.iter_chunks_without_embeddings())

    def iter_chunks_without_embeddings(self) -> Iterator[Chunk]:
        yield from (
            chunk
            for chunk in self.chunks
            if chunk.artifact_id not in self._revoked_artifact_ids
        )

    def list_chunks_without_embeddings(
        self, limit: int, offset: int = 0
    ) -> list[Chunk]:
        page = self.chunks[offset : offset + limit]
        return [
            chunk
            for chunk in page
            if chunk.artifact_id not in self._revoked_artifact_ids
        ]

    def all_ids(self) -> frozenset[str]:
        return frozenset(
            chunk.id
            for chunk in self.chunks
            if chunk.artifact_id not in self._revoked_artifact_ids
        )

    def corpus_revision(self) -> int:
        return self._corpus_revision

    def retrieval_fingerprints(self) -> frozenset[str]:
        return frozenset(
            "\x00".join(
                [
                    chunk.id,
                    encode_project_ids(chunk.project_ids),
                    chunk.source_role,
                    chunk.connector_id or "",
                    chunk.connector_source_id or "",
                    chunk.source_system or "",
                    chunk.created_at or "",
                ]
            )
            for chunk in self.chunks
            if chunk.artifact_id not in self._revoked_artifact_ids
        )

    def project_ids_for_artifact(self, artifact_id: str) -> frozenset[str]:
        return frozenset(
            project_id
            for chunk in self.chunks
            if chunk.artifact_id == artifact_id
            for project_id in chunk.project_ids
        )

    def set_project_ids_for_artifact(
        self,
        artifact_id: str,
        project_ids: tuple[str, ...],
    ) -> int:
        normalized = tuple(dict.fromkeys(pid for pid in project_ids if pid))
        current = {
            project_id
            for chunk in self.chunks
            if chunk.artifact_id == artifact_id
            for project_id in chunk.project_ids
        }
        tombstoned = bool(current - set(normalized))
        if tombstoned:
            self._revoked_artifact_ids.add(artifact_id)
            self._corpus_revision += 1

        updated = 0
        rewritten: list[Chunk] = []
        for chunk in self.chunks:
            if chunk.artifact_id != artifact_id:
                rewritten.append(chunk)
                continue
            rewritten.append(replace(chunk, project_ids=normalized))
            updated += 1

        self.chunks = rewritten
        if updated:
            self._corpus_revision += 1
        if tombstoned:
            self._revoked_artifact_ids.discard(artifact_id)
            self._corpus_revision += 1
        return updated

    def remove_project(self, project_id: str) -> int:
        tombstones = {
            chunk.artifact_id
            for chunk in self.chunks
            if project_id in chunk.project_ids
        }
        if tombstones:
            self._revoked_artifact_ids.update(tombstones)
            self._corpus_revision += 1

        updated = 0
        rewritten: list[Chunk] = []
        for chunk in self.chunks:
            if project_id not in chunk.project_ids:
                rewritten.append(chunk)
                continue
            rewritten.append(
                replace(
                    chunk,
                    project_ids=tuple(
                        pid for pid in chunk.project_ids if pid != project_id
                    ),
                )
            )
            updated += 1

        self.chunks = rewritten
        if updated:
            self._corpus_revision += 1
        if tombstones:
            self._revoked_artifact_ids.difference_update(tombstones)
            self._corpus_revision += 1
        return updated

    def count(self) -> int:
        return len(self.chunks)
