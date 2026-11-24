from collections.abc import Iterable
from pathlib import Path

import pytest

from corpus.source_filter import SourceExclusions
from store.source_state_store import SourceStateStore


@pytest.fixture
def store(tmp_path: Path) -> Iterable[SourceStateStore]:
    store = SourceStateStore(path=str(tmp_path / "source_state.db"))
    yield store
    store.close()


def test_exclusions_are_empty_by_default(store: SourceStateStore) -> None:
    assert store.get_exclusions() == SourceExclusions()


def test_disabling_a_connector_excludes_it(store: SourceStateStore) -> None:
    store.set_connector_enabled("github", False)

    assert store.get_exclusions() == SourceExclusions(connectors=frozenset({"github"}))


def test_re_enabling_a_connector_removes_the_exclusion(
    store: SourceStateStore,
) -> None:
    store.set_connector_enabled("github", False)
    store.set_connector_enabled("github", True)

    assert store.get_exclusions() == SourceExclusions()


def test_disabling_sources_excludes_only_those_sources(
    store: SourceStateStore,
) -> None:
    store.set_sources_enabled("github", {"owner/repo-a": False, "owner/repo-b": True})

    assert store.get_exclusions() == SourceExclusions(
        sources=frozenset({("github", "owner/repo-a")})
    )


def test_re_enabling_a_source_removes_the_exclusion(store: SourceStateStore) -> None:
    store.set_sources_enabled("github", {"owner/repo": False})
    store.set_sources_enabled("github", {"owner/repo": True})

    assert store.get_exclusions() == SourceExclusions()


def test_connector_and_source_exclusions_combine(store: SourceStateStore) -> None:
    store.set_connector_enabled("jira", False)
    store.set_sources_enabled("github", {"owner/repo": False})

    assert store.get_exclusions() == SourceExclusions(
        connectors=frozenset({"jira"}),
        sources=frozenset({("github", "owner/repo")}),
    )
