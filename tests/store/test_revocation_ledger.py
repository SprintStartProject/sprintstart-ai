from corpus.ids import ArtifactId
from store.revocation_ledger import RevocationLedger


def _ledger() -> RevocationLedger:
    return RevocationLedger(None)


def test_a_new_revocation_is_visible_and_bumps_both_revisions() -> None:
    ledger = _ledger()

    generations = ledger.begin(frozenset({"a1"}), reason="delete", owner="o")

    assert generations == {"a1": 1}
    assert ledger.snapshot() == (1, frozenset({"a1"}))
    assert ledger.corpus_revision() == 1


def test_beginning_the_same_revocation_again_advances_only_the_generation() -> None:
    ledger = _ledger()
    ledger.begin(frozenset({"a1"}), reason="delete", owner="o")

    assert ledger.begin(frozenset({"a1"}), reason="delete", owner="o") == {"a1": 2}
    assert ledger.revocation_revision() == 1
    assert ledger.corpus_revision() == 1


def test_beginning_nothing_changes_nothing() -> None:
    ledger = _ledger()

    assert ledger.begin(frozenset(), reason="delete", owner="o") == {}
    assert ledger.revocation_revision() == 0
    assert ledger.corpus_revision() == 0


def test_completing_the_current_generation_lifts_the_revocation() -> None:
    ledger = _ledger()
    generations = ledger.begin(frozenset({"a1"}), reason="delete", owner="o")

    ledger.complete(generations, reason="delete", owner="o")

    assert ledger.snapshot() == (2, frozenset())
    assert ledger.corpus_revision() == 2


def test_completing_a_stale_generation_keeps_the_newer_retry_revoked() -> None:
    ledger = _ledger()
    stale = ledger.begin(frozenset({"a1"}), reason="delete", owner="o")
    ledger.begin(frozenset({"a1"}), reason="delete", owner="o")

    ledger.complete(stale, reason="delete", owner="o")

    assert ledger.snapshot()[1] == frozenset({"a1"})


def test_completing_nothing_changes_nothing() -> None:
    ledger = _ledger()
    ledger.begin(frozenset({"a1"}), reason="delete", owner="o")

    ledger.complete({}, reason="delete", owner="o")

    assert ledger.snapshot() == (1, frozenset({"a1"}))


def test_a_failed_membership_revocation_is_offered_for_recovery() -> None:
    ledger = _ledger()
    generations = ledger.begin(frozenset({"a1"}), reason="membership", owner="o")

    assert ledger.failed_operations_for_artifact(ArtifactId("a1"), "membership") == {}
    ledger.fail(generations, reason="membership", owner="o")

    assert ledger.failed_operations_for_artifact(ArtifactId("a1"), "membership") == {
        "o": 1
    }
    assert ledger.failed_operations_for_artifact(ArtifactId("a1"), "delete") == {}
    assert (
        ledger.failed_operations_for_artifact(ArtifactId("other"), "membership") == {}
    )
    assert ledger.snapshot()[1] == frozenset({"a1"})


def test_failing_a_stale_generation_marks_nothing() -> None:
    ledger = _ledger()
    stale = ledger.begin(frozenset({"a1"}), reason="membership", owner="o")
    ledger.begin(frozenset({"a1"}), reason="membership", owner="o")

    ledger.fail(stale, reason="membership", owner="o")

    assert ledger.failed_operations_for_artifact(ArtifactId("a1"), "membership") == {}


def test_beginning_again_clears_the_failed_mark() -> None:
    ledger = _ledger()
    generations = ledger.begin(frozenset({"a1"}), reason="membership", owner="o")
    ledger.fail(generations, reason="membership", owner="o")

    ledger.begin(frozenset({"a1"}), reason="membership", owner="o")

    assert ledger.failed_operations_for_artifact(ArtifactId("a1"), "membership") == {}


def test_artifacts_lists_those_held_by_one_reason_and_owner() -> None:
    ledger = _ledger()
    ledger.begin(frozenset({"a1"}), reason="membership", owner="project:p")
    ledger.begin(frozenset({"a2"}), reason="membership", owner="project:q")
    ledger.begin(frozenset({"a3"}), reason="delete", owner="project:p")

    assert ledger.artifacts("membership", "project:p") == frozenset({"a1"})


def test_the_snapshot_is_cached_until_the_revision_moves() -> None:
    ledger = _ledger()
    ledger.begin(frozenset({"a1"}), reason="delete", owner="o")
    first = ledger.snapshot()

    assert ledger.snapshot() is not None
    assert ledger.snapshot() == first
    ledger.begin(frozenset({"a2"}), reason="delete", owner="o")
    assert ledger.snapshot() == (2, frozenset({"a1", "a2"}))


def test_bumping_the_corpus_revision_counts_up() -> None:
    ledger = _ledger()

    ledger.bump_corpus_revision()
    ledger.bump_corpus_revision()

    assert ledger.corpus_revision() == 2
