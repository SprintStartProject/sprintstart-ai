"""The one value five idempotent jobs compare against.

Orientation packets, competency modules, starter-work mining, graph proposals and
board diagrams all decide "is my cached answer still true?" by comparing this, so
the properties below are load-bearing for every one of them. It had no test of its
own while it lived inside blueprint generation -- it was only ever exercised
through the job that happened to define it.
"""

from onboarding.corpus import corpus_fingerprint
from rag.types import Chunk
from tests.stubs.store import StubVectorStore


def _store(*texts: str) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id=f"a{i}",
                filename=f"doc{i}.md",
                text=text,
                embedding=[0.0] * 768,
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


def test_the_same_corpus_fingerprints_the_same() -> None:
    # What makes a cache servable: a crawl that found nothing new must not look
    # like a change, or every job re-runs on every crawl.
    assert corpus_fingerprint(_store("a", "b")) == corpus_fingerprint(_store("a", "b"))


def test_changed_content_changes_the_fingerprint() -> None:
    # And the other half: an edit must be visible, or a packet goes on describing
    # code that has moved.
    assert corpus_fingerprint(_store("a", "b")) != corpus_fingerprint(_store("a", "B"))


def test_added_material_changes_the_fingerprint() -> None:
    assert corpus_fingerprint(_store("a")) != corpus_fingerprint(_store("a", "b"))


def test_an_empty_corpus_has_a_stable_fingerprint() -> None:
    # Not an error and not empty: a project with nothing indexed compares fine.
    assert corpus_fingerprint(StubVectorStore()) == corpus_fingerprint(
        StubVectorStore()
    )
