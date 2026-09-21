"""The one value five idempotent jobs compare against.

Orientation packets, competency modules, starter-work mining, graph proposals and
board diagrams all decide "is my cached answer still true?" by comparing this, so
the properties below are load-bearing for every one of them. It had no test of its
own while it lived inside blueprint generation -- it was only ever exercised
through the job that happened to define it.
"""

from pydantic import BaseModel

from onboarding.corpus import corpus_fingerprint, fingerprint_gate
from onboarding.progress import ProgressStream
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


def test_extra_fingerprint_material_changes_hash() -> None:
    store = _store("a")
    base = corpus_fingerprint(store)
    with_extra = corpus_fingerprint(store, extra_fingerprint_material=["issue:1:OPEN"])
    assert base != with_extra
    # Without extra, output is identical
    assert corpus_fingerprint(store, extra_fingerprint_material=None) == base
    assert corpus_fingerprint(store, extra_fingerprint_material=[]) == base


class _DummyOutcome(BaseModel):
    status: str
    note: str = ""


def test_fingerprint_gate_unchanged_path() -> None:
    store = _store("a")
    fp = corpus_fingerprint(store)
    progress = ProgressStream("test")
    computed_fp, events, outcome = fingerprint_gate(
        progress,
        store,
        last_fingerprint=fp,
        make_unchanged=lambda: _DummyOutcome(status="unchanged"),
        make_empty=lambda: _DummyOutcome(status="skipped"),
        unchanged_label="unchanged",
        empty_warning_label="empty_warn",
        empty_done_label="empty_done",
    )
    assert computed_fp is None
    assert outcome is not None
    assert outcome.status == "unchanged"
    assert len(events) == 1
    assert events[0]["type"] == "done"


def test_fingerprint_gate_empty_store_path() -> None:
    store = StubVectorStore()
    progress = ProgressStream("test")
    computed_fp, events, outcome = fingerprint_gate(
        progress,
        store,
        last_fingerprint=None,
        make_unchanged=lambda: _DummyOutcome(status="unchanged"),
        make_empty=lambda: _DummyOutcome(status="skipped"),
        unchanged_label="unchanged",
        empty_warning_label="empty_warn",
        empty_done_label="empty_done",
    )
    assert computed_fp is None
    assert outcome is not None
    assert outcome.status == "skipped"
    assert len(events) == 2
    assert events[0]["type"] == "warning"
    assert events[1]["type"] == "done"


def test_corpus_fingerprint_scoped_to_projects() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="doc1.md",
                text="content for proj1",
                embedding=[0.0] * 768,
                project_ids=("p1",),
            ),
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="doc2.md",
                text="content for proj2",
                embedding=[0.0] * 768,
                project_ids=("p2",),
            ),
        ]
    )
    fp_p1 = corpus_fingerprint(store, project_ids=frozenset({"p1"}))
    fp_p2 = corpus_fingerprint(store, project_ids=frozenset({"p2"}))
    assert fp_p1 != fp_p2

    # Mutating p2 content does not change p1's fingerprint
    store.add(
        [
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="doc2.md",
                text="mutated content for proj2",
                embedding=[0.0] * 768,
                project_ids=("p2",),
            )
        ]
    )
    assert corpus_fingerprint(store, project_ids=frozenset({"p1"})) == fp_p1
    assert corpus_fingerprint(store, project_ids=frozenset({"p2"})) != fp_p2


def test_fingerprint_gate_scoped_empty_store_path() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="doc1.md",
                text="only in p2",
                embedding=[0.0] * 768,
                project_ids=("p2",),
            )
        ]
    )
    progress = ProgressStream("test")
    computed_fp, events, outcome = fingerprint_gate(
        progress,
        store,
        last_fingerprint=None,
        make_unchanged=lambda: _DummyOutcome(status="unchanged"),
        make_empty=lambda: _DummyOutcome(status="skipped"),
        unchanged_label="unchanged",
        empty_warning_label="empty_warn",
        empty_done_label="empty_done",
        project_ids=frozenset({"p1"}),
    )
    assert computed_fp is None
    assert outcome is not None
    assert outcome.status == "skipped"
    assert len(events) == 2
    assert events[0]["type"] == "warning"
    assert events[1]["type"] == "done"


def test_fingerprint_gate_proceeds_on_new_fingerprint() -> None:
    store = _store("a")
    progress = ProgressStream("test")
    computed_fp, events, outcome = fingerprint_gate(
        progress,
        store,
        last_fingerprint="old_hash",
        make_unchanged=lambda: _DummyOutcome(status="unchanged"),
        make_empty=lambda: _DummyOutcome(status="skipped"),
        unchanged_label="unchanged",
        empty_warning_label="empty_warn",
        empty_done_label="empty_done",
    )
    assert computed_fp is not None
    assert events == []
    assert outcome is None
