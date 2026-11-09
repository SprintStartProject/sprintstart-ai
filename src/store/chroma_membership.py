"""Which projects each stored chunk belongs to, and rewriting it.

Chroma metadata cannot hold lists, so membership lives twice in a chunk's metadata
(see ``corpus.filters``). Rewriting it is the dangerous part: a project removed from
a chunk must stop seeing it before the first page is rewritten, so every artifact
that loses a project is revoked first and un-revoked only once all of its chunks
are repaired.
"""

from collections.abc import Callable, Iterator, Mapping
from dataclasses import replace
from typing import Any, cast

from chromadb import Collection
from chromadb.api.types import Where

from corpus.filters import (
    PROJECT_IDS_METADATA_KEY,
    decode_project_ids,
    project_metadata_key,
)
from corpus.ids import ArtifactId
from corpus.types import Chunk
from store import chroma_records
from store.chroma_records import chunks_from_get_result
from store.metadata_store import IngestionMetadataStore
from store.revocation_ledger import RevocationLedger


def iter_metadata_records(
    collection: Collection,
    where: Where | None = None,
) -> Iterator[tuple[str, Mapping[str, object]]]:
    """Yield ids and metadata through bounded reads without hydrating text."""
    offset = 0

    while True:
        raw_result = collection.get(
            where=where,
            include=["metadatas"],
            limit=chroma_records.MAX_GET_PAGE,
            offset=offset,
        )
        ids = raw_result["ids"]
        if not ids:
            return

        metadatas = cast(
            list[Mapping[str, object]],
            raw_result.get("metadatas") or [],
        )
        yield from zip(ids, metadatas, strict=True)
        offset += len(ids)


class ChromaMembership:
    def __init__(
        self,
        collection: Collection,
        ledger: RevocationLedger,
        revision_store: IngestionMetadataStore | None,
        write: Callable[[list[Chunk]], None],
    ) -> None:
        self._collection = collection
        self._ledger = ledger
        self._revision_store = revision_store
        self._write = write

    def project_ids_for_artifact(self, artifact: ArtifactId) -> frozenset[str]:
        return frozenset(
            project_id
            for _, metadata in iter_metadata_records(
                self._collection, where={"artifact_id": artifact}
            )
            for project_id in decode_project_ids(metadata.get(PROJECT_IDS_METADATA_KEY))
        )

    def set_project_ids_for_artifact(
        self,
        artifact: ArtifactId,
        project_ids: tuple[str, ...],
    ) -> int:
        normalized = tuple(dict.fromkeys(pid for pid in project_ids if pid))
        revocation_owner = f"artifact:{artifact}"
        recovery_operations = self._ledger.failed_operations_for_artifact(
            artifact,
            reason="membership",
        )
        own_recovery = artifact in self._ledger.artifacts(
            reason="membership",
            owner=revocation_owner,
        )
        return self._rewrite(
            {"artifact_id": artifact},
            lambda _chunk: normalized,
            recovery_artifact_ids=(
                frozenset({artifact})
                if recovery_operations or own_recovery
                else frozenset()
            ),
            revocation_owner=revocation_owner,
            recovery_operations={artifact: recovery_operations},
        )

    def remove_project(self, project: str) -> int:
        # Selected by the boolean marker rather than the delimited string,
        # because the marker is what retrieval filters on -- a chunk that still
        # carries it is still reachable, whatever the string says.
        revocation_owner = f"project:{project}"
        recovery_artifact_ids = self._ledger.artifacts(
            reason="membership",
            owner=revocation_owner,
        )
        if self._revision_store is not None:
            recovery_artifact_ids |= frozenset(
                artifact.id
                for artifact in self._revision_store.list_artifacts(
                    status=None,
                    project_id=project,
                )
            )
        return self._rewrite(
            {project_metadata_key(project): {"$eq": True}},
            lambda projects: tuple(pid for pid in projects if pid != project),
            recovery_artifact_ids=recovery_artifact_ids,
            revocation_owner=revocation_owner,
        )

    def _rewrite(
        self,
        where: dict[str, Any],
        membership: Callable[[tuple[str, ...]], tuple[str, ...]],
        recovery_artifact_ids: frozenset[str] = frozenset(),
        revocation_owner: str = "membership",
        recovery_operations: Mapping[str, Mapping[str, int]] | None = None,
    ) -> int:
        """Rewrite membership while keeping removals fail-closed on failure.

        Matching metadata is collected before mutation. Any artifact losing a
        project is tombstoned before the first page is rewritten, so a partial
        failure cannot leave later pages searchable through stale markers.
        """
        records = list(iter_metadata_records(self._collection, where=cast(Any, where)))
        ids = [chunk_id for chunk_id, _ in records]
        tombstones = recovery_artifact_ids | frozenset(
            str(metadata["artifact_id"])
            for _, metadata in records
            if set(decode_project_ids(metadata.get(PROJECT_IDS_METADATA_KEY)))
            - set(
                membership(decode_project_ids(metadata.get(PROJECT_IDS_METADATA_KEY)))
            )
        )
        operations = self._ledger.begin(
            tombstones,
            reason="membership",
            owner=revocation_owner,
        )

        try:
            # Any failure leaves every affected artifact tombstoned until a
            # retry repairs all of its chunks.
            page = chroma_records.MAX_GET_PAGE
            for start in range(0, len(ids), page):
                raw_result = self._collection.get(
                    ids=ids[start : start + page],
                    include=["documents", "metadatas", "embeddings"],
                )
                self._write(
                    [
                        replace(
                            chunk,
                            project_ids=membership(chunk.project_ids),
                        )
                        for chunk in chunks_from_get_result(raw_result)
                    ]
                )

            self._ledger.complete(
                operations,
                reason="membership",
                owner=revocation_owner,
            )
            for artifact_id, owner_generations in (recovery_operations or {}).items():
                for owner, generation in owner_generations.items():
                    self._ledger.complete(
                        {artifact_id: generation},
                        reason="membership",
                        owner=owner,
                    )
        except Exception:
            self._ledger.fail(
                operations,
                reason="membership",
                owner=revocation_owner,
            )
            raise
        return len(ids)
