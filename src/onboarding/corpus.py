"""What the indexed corpus currently is, as one comparable value.

Every job in this package that can be asked to run again — orientation packets,
starter-work mining, board diagrams — is idempotent against the corpus rather
than against a clock. The caller records the fingerprint its last result was
derived from and sends it back; an unchanged corpus is answered without
retrieval or generation, and a corpus that has moved is redone.

That is the whole reason age is never treated as staleness here: a packet
describing code nobody has touched in a year is perfectly current, and one
describing code that changed this morning is not, however recently it was
written.
"""

import hashlib
from collections.abc import Callable, Iterable
from typing import cast

from pydantic import BaseModel

from onboarding.progress import ProgressEvent, ProgressStream
from store.base import VectorStore


def _compute_fingerprint(
    store: VectorStore,
    project_ids: frozenset[str] | None = None,
    extra_fingerprint_material: Iterable[str] | None = None,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    scoped_count = 0
    chunk_fields: list[tuple[str, str]] = []
    for chunk in store.iter_chunks_without_embeddings():
        if project_ids is not None:
            if not any(pid in project_ids for pid in chunk.project_ids):
                continue
        scoped_count += 1
        chunk_fields.append((chunk.id, chunk.text))

    for chunk_id, text in sorted(chunk_fields):
        digest.update(chunk_id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    if extra_fingerprint_material:
        for item in sorted(extra_fingerprint_material):
            digest.update(item.encode("utf-8"))
            digest.update(b"\0")
    return digest.hexdigest(), scoped_count


def corpus_fingerprint(
    store: VectorStore,
    extra_fingerprint_material: Iterable[str] | None = None,
    *,
    project_ids: frozenset[str] | None = None,
) -> str:
    """Stable hash of the corpus contents; changes iff the corpus changes.

    Ordered by chunk id and folded over both id and text, so it is the *content*
    that is fingerprinted — re-ingesting the same material unchanged produces the
    same value, and a caller's cache survives a crawl that found nothing new.

    Optional ``project_ids`` scopes the fingerprint to only chunks belonging to
    those projects.

    Optional ``extra_fingerprint_material`` allows caller jobs (e.g. starter-work
    mining) to include non-chunk metadata (such as issue states or pool ids) that
    influence eligibility. When omitted or empty, the base corpus hash is identical.
    """
    digest, _ = _compute_fingerprint(
        store,
        project_ids=project_ids,
        extra_fingerprint_material=extra_fingerprint_material,
    )
    return digest


def fingerprint_gate[T: BaseModel](
    progress: ProgressStream,
    store: VectorStore,
    last_fingerprint: str | None,
    make_unchanged: Callable[[], T],
    make_empty: Callable[[], T],
    *,
    unchanged_label: str,
    empty_warning_label: str,
    empty_done_label: str,
    extra_fingerprint_material: Iterable[str] | None = None,
    project_ids: frozenset[str] | None = None,
) -> tuple[str | None, list[ProgressEvent], T | None]:
    """Shared short-circuit gate for generator jobs.

    Returns ``(fingerprint, early_events, early_outcome)``. If the corpus is
    unchanged from ``last_fingerprint`` or the store is empty, ``fingerprint``
    is None, ``early_events`` contains the terminal event(s), and ``early_outcome``
    is the outcome to return. Otherwise, ``fingerprint`` is the computed hash,
    ``early_events`` is empty, and ``early_outcome`` is None.
    """
    fingerprint, scoped_count = _compute_fingerprint(
        store,
        project_ids=project_ids,
        extra_fingerprint_material=extra_fingerprint_material,
    )
    if last_fingerprint is not None and last_fingerprint == fingerprint:
        outcome = make_unchanged()
        return (
            None,
            [
                progress.done(
                    unchanged_label,
                    cast(dict[str, object], outcome.model_dump(mode="json")),
                )
            ],
            outcome,
        )

    is_empty = scoped_count == 0 if project_ids is not None else store.count() == 0
    if is_empty:
        outcome = make_empty()
        return (
            None,
            [
                progress.warning(empty_warning_label),
                progress.done(
                    empty_done_label,
                    cast(dict[str, object], outcome.model_dump(mode="json")),
                ),
            ],
            outcome,
        )

    return fingerprint, [], None


__all__ = ["corpus_fingerprint", "fingerprint_gate"]
