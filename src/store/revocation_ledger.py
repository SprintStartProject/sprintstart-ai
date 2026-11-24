"""Which artifacts a vector store must hide right now, and the corpus revision.

With a metadata store the ledger is a thin view of what that store persists, so every
process agrees. Without one (directly constructed ephemeral stores used by tests and
local tools) it keeps the same bookkeeping in memory, so caches stay correct in-process.
"""

import threading
from collections.abc import Mapping

from corpus.ids import ArtifactId
from store.metadata_store import IngestionMetadataStore
from store.revocation_tables import RevocationReason

_Key = tuple[str, RevocationReason, str]


class RevocationLedger:
    def __init__(self, revision_store: IngestionMetadataStore | None) -> None:
        self._revision_store = revision_store
        self._local_revision = 0
        self._local_revocation_revision = 0
        self._local_revocations: dict[_Key, int] = {}
        self._local_failed_revocations: set[_Key] = set()
        self._cached_revocation_revision = -1
        self._cached_revocations: frozenset[str] = frozenset()
        self._lock = threading.Lock()

    def bump_corpus_revision(self) -> None:
        if self._revision_store is not None:
            self._revision_store.bump_corpus_revision()
            return

        with self._lock:
            self._local_revision += 1

    def corpus_revision(self) -> int:
        if self._revision_store is not None:
            return self._revision_store.corpus_revision()

        with self._lock:
            return self._local_revision

    def revocation_revision(self) -> int:
        if self._revision_store is not None:
            return self._revision_store.revocation_revision()

        with self._lock:
            return self._local_revocation_revision

    def snapshot(self) -> tuple[int, frozenset[str]]:
        revision = self.revocation_revision()
        with self._lock:
            if revision == self._cached_revocation_revision:
                return revision, self._cached_revocations

        if self._revision_store is not None:
            revision, revoked = self._revision_store.revocation_snapshot()
        else:
            with self._lock:
                revision = self._local_revocation_revision
                revoked = frozenset(
                    artifact_id
                    for artifact_id, _reason, _owner in self._local_revocations
                )

        with self._lock:
            self._cached_revocation_revision = revision
            self._cached_revocations = revoked
        return revision, revoked

    def begin(
        self,
        artifact_ids: frozenset[str],
        reason: RevocationReason,
        owner: str,
    ) -> dict[str, int]:
        if not artifact_ids:
            return {}

        if self._revision_store is not None:
            return self._revision_store.begin_revocation_operations(
                sorted(artifact_ids),
                reason=reason,
                owner=owner,
            )

        with self._lock:
            visible_change = False
            generations: dict[str, int] = {}
            for artifact_id in artifact_ids:
                key = (artifact_id, reason, owner)
                visible_change = visible_change or key not in self._local_revocations
                generation = self._local_revocations.get(key, 0) + 1
                self._local_revocations[key] = generation
                self._local_failed_revocations.discard(key)
                generations[artifact_id] = generation
            if visible_change:
                self._local_revocation_revision += 1
                self._local_revision += 1
            return generations

    def fail(
        self,
        generations: Mapping[str, int],
        reason: RevocationReason,
        owner: str,
    ) -> None:
        if not generations:
            return

        if self._revision_store is not None:
            self._revision_store.fail_revocation_operations(
                generations,
                reason=reason,
                owner=owner,
            )
            return

        with self._lock:
            for artifact_id, generation in generations.items():
                key = (artifact_id, reason, owner)
                if self._local_revocations.get(key) == generation:
                    self._local_failed_revocations.add(key)

    def complete(
        self,
        generations: Mapping[str, int],
        reason: RevocationReason,
        owner: str,
    ) -> None:
        if not generations:
            return

        if self._revision_store is not None:
            self._revision_store.complete_revocation_operations(
                generations,
                reason=reason,
                owner=owner,
            )
            return

        with self._lock:
            changed = False
            for artifact_id, generation in generations.items():
                key = (artifact_id, reason, owner)
                if self._local_revocations.get(key) == generation:
                    del self._local_revocations[key]
                    self._local_failed_revocations.discard(key)
                    changed = True
            if changed:
                self._local_revocation_revision += 1
                self._local_revision += 1

    def artifacts(self, reason: RevocationReason, owner: str) -> frozenset[str]:
        if self._revision_store is not None:
            return self._revision_store.revocation_artifacts(reason, owner)

        with self._lock:
            return frozenset(
                artifact_id
                for artifact_id, stored_reason, stored_owner in self._local_revocations
                if stored_reason == reason and stored_owner == owner
            )

    def failed_operations_for_artifact(
        self,
        artifact: ArtifactId,
        reason: RevocationReason,
    ) -> dict[str, int]:
        if self._revision_store is not None:
            return self._revision_store.revocation_operations_for_artifact(
                artifact,
                reason,
            )

        with self._lock:
            return {
                owner: generation
                for (stored_id, stored_reason, owner), generation in (
                    self._local_revocations.items()
                )
                if stored_id == artifact
                and stored_reason == reason
                and (stored_id, stored_reason, owner) in self._local_failed_revocations
            }
