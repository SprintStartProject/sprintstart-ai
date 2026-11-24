from collections.abc import Iterable
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_source_state_store
from corpus.source_filter import SourceExclusions
from store.source_state_store import SourceStateStore


@pytest.fixture
def source_state_store(tmp_path: Path) -> Iterable[SourceStateStore]:
    store = SourceStateStore(path=str(tmp_path / "source_state.db"))
    yield store
    store.close()


@pytest.fixture
def client(source_state_store: SourceStateStore) -> Iterable[TestClient]:
    app.dependency_overrides[get_source_state_store] = lambda: source_state_store

    yield TestClient(app)

    app.dependency_overrides.clear()


def test_configure_connector_disables_connector(
    client: TestClient, source_state_store: SourceStateStore
) -> None:
    response = client.patch(
        "/api/v1/connectors/github",
        json={"enabled": False},
    )

    assert response.status_code == 200
    assert response.json() == {"connector_id": "github", "enabled": False}
    assert source_state_store.get_exclusions() == SourceExclusions(
        connectors=frozenset({"github"})
    )


def test_configure_connector_re_enables_connector(
    client: TestClient, source_state_store: SourceStateStore
) -> None:
    source_state_store.set_connector_enabled("github", False)

    response = client.patch(
        "/api/v1/connectors/github",
        json={"enabled": True},
    )

    assert response.status_code == 200
    assert source_state_store.get_exclusions() == SourceExclusions()


def test_patch_sources_disables_individual_sources(
    client: TestClient, source_state_store: SourceStateStore
) -> None:
    response = client.patch(
        "/api/v1/sources/github",
        json={"sources": {"owner/repo-a": False, "owner/repo-b": True}},
    )

    assert response.status_code == 200
    assert response.json() == {
        "connector_id": "github",
        "sources": {"owner/repo-a": False, "owner/repo-b": True},
    }
    assert source_state_store.get_exclusions() == SourceExclusions(
        sources=frozenset({("github", "owner/repo-a")})
    )
