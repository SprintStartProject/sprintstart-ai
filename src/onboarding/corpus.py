"""What the indexed corpus currently is, as one comparable value.

Every job in this package that can be asked to run again — orientation packets,
competency modules, starter-work mining, graph proposals, board diagrams — is
idempotent against the corpus rather than against a clock. The caller records the
fingerprint its last result was derived from and sends it back; an unchanged
corpus is answered without retrieval or generation, and a corpus that has moved
is redone.

That is the whole reason age is never treated as staleness here: a packet
describing code nobody has touched in a year is perfectly current, and one
describing code that changed this morning is not, however recently it was
written.
"""

import hashlib

from store.base import VectorStore


def corpus_fingerprint(store: VectorStore) -> str:
    """Stable hash of the corpus contents; changes iff the corpus changes.

    Ordered by chunk id and folded over both id and text, so it is the *content*
    that is fingerprinted — re-ingesting the same material unchanged produces the
    same value, and a caller's cache survives a crawl that found nothing new.
    """
    digest = hashlib.sha256()
    for chunk in sorted(store.all_chunks(), key=lambda c: c.id):
        digest.update(chunk.id.encode("utf-8"))
        digest.update(b"\0")
        digest.update(chunk.text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["corpus_fingerprint"]
