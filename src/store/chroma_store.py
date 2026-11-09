from collections.abc import Iterator, Mapping
from typing import cast

import chromadb
import chromadb.api
from chromadb.api.types import Metadata, PyEmbeddings, QueryResult, Where

from corpus.ids import ArtifactId, ChunkId
from corpus.source_filter import SourceExclusions
from corpus.source_role import SourceRole
from corpus.types import Chunk, RetrievalFilters, ScoredChunk
from store import chroma_records
from store.chroma_client import get_persistent_client
from store.chroma_membership import ChromaMembership, iter_metadata_records
from store.chroma_records import (
    chunk_metadata,
    chunks_from_get_result,
    chunks_without_embeddings_from_get_result,
    retrieval_fingerprint,
    scored_chunk,
    visible_chunks,
)
from store.chroma_where import where_filter_for_chroma
from store.metadata_store import IngestionMetadataStore
from store.revocation_ledger import RevocationLedger

# Used only when the client cannot report its own write ceiling. Chroma's local
# ceiling is 5,461 (the 32,766 bind-variable budget over the six columns a write
# fills), so staying under that is safe for any backend we run against.
_MAX_WRITE_PAGE_FALLBACK: int = 5_000


class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str = "chunks",
        client: chromadb.api.ClientAPI | None = None,
        path: str | None = None,
        revision_store: IngestionMetadataStore | None = None,
    ) -> None:
        if client is not None:
            self._client: chromadb.api.ClientAPI = client
        elif path is not None:
            self._client = get_persistent_client(path)
        else:
            self._client = chromadb.EphemeralClient()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self._revision_store = revision_store
        self._ledger = RevocationLedger(revision_store)
        self._membership = ChromaMembership(
            self._collection,
            self._ledger,
            revision_store,
            lambda chunks: self.add(chunks),
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return

        embeddings: list[list[float]] = [chunk.embedding for chunk in chunks]

        metadatas = [chunk_metadata(chunk) for chunk in chunks]

        ids = [chunk.id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]

        # A single write may not exceed the backend's batch ceiling, which is
        # far below what a ``get`` can page (see ``MAX_GET_PAGE``). Exceeding
        # it raises *after* the delete below has already committed, which would
        # leave the batch deleted and never rewritten, so the batching has to
        # happen here rather than in any one caller.
        page = self._max_write_page()

        mutated = False
        try:
            for start in range(0, len(ids), page):
                stop = start + page
                page_ids = ids[start:stop]

                # Chroma's upsert *merges* metadata instead of replacing it, so a
                # key that is no longer written would survive a re-ingest —
                # including the ``project:<id>`` marker of a project the artifact
                # was removed from, which would keep it retrievable from that
                # project forever. Deleting first makes each chunk's metadata
                # exactly what we write here.
                mutated = True
                self._collection.delete(ids=page_ids)

                self._collection.upsert(
                    ids=page_ids,
                    documents=documents[start:stop],
                    embeddings=cast(PyEmbeddings, embeddings[start:stop]),
                    metadatas=cast(list[Metadata], metadatas[start:stop]),
                )
        finally:
            # Bump even when a later page fails: an earlier delete/upsert may
            # already have changed retrieval-visible state.
            if mutated:
                self._ledger.bump_corpus_revision()

    def _max_write_page(self) -> int:
        """Largest id count one ``delete``/``upsert`` pair may carry.

        The client reports its own ceiling, so a remote backend with a different
        budget is honoured too. It is still capped at ``MAX_GET_PAGE``, because
        the ``delete`` half of the pair spends the same bind variables a ``get``
        does, whatever the upsert would allow.
        """
        try:
            limit = int(self._client.get_max_batch_size())
        except Exception:  # pragma: no cover - older clients / custom doubles
            return _MAX_WRITE_PAGE_FALLBACK
        return (
            min(limit, chroma_records.MAX_GET_PAGE)
            if limit > 0
            else _MAX_WRITE_PAGE_FALLBACK
        )

    def query(
        self,
        embedding: list[float],
        top_k: int,
        min_score: float,
        filters: RetrievalFilters | None = None,
        exclude_roles: frozenset[SourceRole] = frozenset(),
        exclusions: SourceExclusions = SourceExclusions(),
    ) -> list[ScoredChunk]:
        if self._collection.count() == 0 or top_k <= 0:
            return []

        searched = self._search_stable(
            embedding, top_k, filters, exclude_roles, exclusions
        )
        if searched is None:
            return []
        raw_result, revoked_artifact_ids = searched

        documents = (raw_result["documents"] or [[]])[0]
        metadatas = cast(
            list[Mapping[str, object]],
            (raw_result["metadatas"] or [[]])[0],
        )
        distances = (raw_result["distances"] or [[]])[0]

        results: list[ScoredChunk] = []
        for chunk_id, text, metadata, distance in zip(
            raw_result["ids"][0], documents, metadatas, distances, strict=True
        ):
            if str(metadata["artifact_id"]) in revoked_artifact_ids:
                # Defensive: custom/older Chroma clients may not enforce the predicate.
                continue
            score = 1.0 - distance
            if score >= min_score:
                results.append(
                    scored_chunk(ChunkId(str(chunk_id)), str(text), metadata, score)
                )

        results.sort(key=lambda c: c.score, reverse=True)
        return results

    def _search_stable(
        self,
        embedding: list[float],
        top_k: int,
        filters: RetrievalFilters | None,
        exclude_roles: frozenset[SourceRole],
        exclusions: SourceExclusions,
    ) -> tuple[QueryResult, frozenset[str]] | None:
        """Search against a revocation snapshot that did not change meanwhile.

        Every constraint goes into the ``where`` clause, which Chroma applies
        before limiting to ``n_results``; filtering the returned window instead
        would let higher-ranked ineligible chunks push every eligible one out.

        Returns ``None`` when revocations changed throughout both attempts:
        no candidates is bounded and fail-closed, and the next request retries.
        """
        for _attempt in range(2):
            revocation_revision, revoked_artifact_ids = self._ledger.snapshot()
            where_filter = where_filter_for_chroma(
                filters,
                exclude_roles,
                exclusions,
                revoked_artifact_ids,
            )
            raw_result = self._collection.query(
                query_embeddings=[embedding],
                n_results=top_k,
                where=where_filter,
                include=["documents", "metadatas", "distances"],
            )
            if self._ledger.revocation_revision() == revocation_revision:
                return raw_result, revoked_artifact_ids
        return None

    def delete(self, artifact_id: str, exclude_ids: list[str] | None = None) -> int:
        tombstone = frozenset({artifact_id})
        membership_recoveries = self._ledger.failed_operations_for_artifact(
            ArtifactId(artifact_id),
            reason="membership",
        )
        # Commit the revocation before the first Chroma read. If that read, the
        # delete, or the process itself fails, retrieval remains fail-closed.
        delete_operation = self._ledger.begin(
            tombstone,
            reason="delete",
            owner="delete",
        )

        # Any exception before cleanup leaves the durable tombstone live for
        # the retry.
        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )

        ids = list(raw_result["ids"])

        if exclude_ids:
            ids = [i for i in ids if i not in exclude_ids]

        if ids:
            try:
                self._collection.delete(ids=ids)
            finally:
                # Invalidate caches even if the backend reports an error after
                # partially applying the deletion.
                self._ledger.bump_corpus_revision()

        # Deleting zero chunks also confirms that nothing remains to expose.
        # If cleanup fails, report failure and retain the safe over-redaction.
        # A confirmed replacement/deletion also repairs membership rewrites
        # that have explicitly failed. Active rewrites are excluded from the
        # recovery snapshot so a concurrent ingest cannot expose an artifact
        # whose membership rewrite later fails. Each owner/generation is
        # completed conditionally, so a newer retry stays protected.
        for owner, generation in membership_recoveries.items():
            self._ledger.complete(
                {artifact_id: generation},
                reason="membership",
                owner=owner,
            )
        self._ledger.complete(
            delete_operation,
            reason="delete",
            owner="delete",
        )
        return len(ids)

    def list_chunks(self, limit: int, offset: int = 0) -> list[Chunk]:
        raw_result = self._collection.get(
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        _, revoked = self._ledger.snapshot()
        return visible_chunks(chunks_from_get_result(raw_result), revoked)

    def list_chunks_by_artifact(
        self,
        artifact_id: str,
        limit: int,
        offset: int = 0,
    ) -> list[Chunk]:
        _, revoked = self._ledger.snapshot()
        if artifact_id in revoked:
            return []

        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            limit=limit,
            offset=offset,
            include=["documents", "metadatas", "embeddings"],
        )
        _, current_revoked = self._ledger.snapshot()
        if artifact_id in current_revoked:
            return []
        return chunks_from_get_result(raw_result)

    def list_chunks_by_positions(
        self,
        artifact_id: str,
        positions: frozenset[int],
        start_page: int | None = None,
    ) -> list[Chunk]:
        if not positions:
            return []

        _, revoked = self._ledger.snapshot()
        if artifact_id in revoked:
            return []

        conditions: list[Where] = [
            {"artifact_id": {"$eq": artifact_id}},
            {"position": {"$in": sorted(positions)}},
        ]
        if start_page is not None:
            conditions.append({"start_page": {"$eq": start_page}})

        raw_result = self._collection.get(
            where={"$and": conditions},
            # Re-ingestion can temporarily leave more than one row at a logical
            # position. Do not assume one stored row per requested position.
            limit=chroma_records.MAX_GET_PAGE,
            include=["documents", "metadatas"],
        )

        # A deletion can become visible while Chroma is serving the read. Check
        # again before publishing any content so revocation remains fail-closed.
        _, current_revoked = self._ledger.snapshot()
        if artifact_id in current_revoked:
            return []

        return sorted(
            chunks_without_embeddings_from_get_result(raw_result),
            key=lambda chunk: (
                chunk.start_page if chunk.start_page is not None else 0,
                chunk.position if chunk.position is not None else -1,
            ),
        )

    def count_by_artifact(self, artifact_id: str) -> int:
        _, revoked = self._ledger.snapshot()
        if artifact_id in revoked:
            return 0

        raw_result = self._collection.get(
            where={"artifact_id": artifact_id},
            include=[],
        )
        _, current_revoked = self._ledger.snapshot()
        return 0 if artifact_id in current_revoked else len(raw_result["ids"])

    def all_chunks_without_embeddings(self) -> list[Chunk]:
        return list(self.iter_chunks_without_embeddings())

    def iter_chunks_without_embeddings(self) -> Iterator[Chunk]:
        """Yield the visible corpus through bounded, embedding-free reads."""
        offset = 0
        total = self._collection.count()

        while offset < total:
            page = self.list_chunks_without_embeddings(
                limit=chroma_records.MAX_GET_PAGE,
                offset=offset,
            )
            yield from page
            offset += min(chroma_records.MAX_GET_PAGE, total - offset)

    def list_chunks_without_embeddings(
        self, limit: int, offset: int = 0
    ) -> list[Chunk]:
        raw_result = self._collection.get(
            include=["documents", "metadatas"],
            limit=limit,
            offset=offset,
        )
        chunks = chunks_without_embeddings_from_get_result(raw_result)
        _, revoked = self._ledger.snapshot()
        return visible_chunks(chunks, revoked)

    def all_ids(self) -> frozenset[str]:
        raw_result = self._collection.get(include=[])
        return frozenset(str(chunk_id) for chunk_id in raw_result["ids"])

    def corpus_revision(self) -> int:
        return self._ledger.corpus_revision()

    def retrieval_fingerprints(self) -> frozenset[str]:
        _, revoked = self._ledger.snapshot()
        return frozenset(
            retrieval_fingerprint(ChunkId(chunk_id), metadata)
            for chunk_id, metadata in iter_metadata_records(self._collection)
            if str(metadata["artifact_id"]) not in revoked
        )

    def project_ids_for_artifact(self, artifact_id: str) -> frozenset[str]:
        return self._membership.project_ids_for_artifact(ArtifactId(artifact_id))

    def set_project_ids_for_artifact(
        self,
        artifact_id: str,
        project_ids: tuple[str, ...],
    ) -> int:
        return self._membership.set_project_ids_for_artifact(
            ArtifactId(artifact_id), project_ids
        )

    def remove_project(self, project_id: str) -> int:
        return self._membership.remove_project(project_id)

    def count(self) -> int:
        return self._collection.count()
