import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_BASE = "/api/v1/onboarding/blueprints"
_PROJECT = "project-1"
_SCOPE = f"project:{_PROJECT}|area:backend"


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "steps": [
                    {
                        "id": "deploy-runbook",
                        "title": "Read the deploy runbook",
                        "requirement": "required",
                        "chunk_ids": ["c1"],
                    }
                ]
            }
        )
    )
    llm.embedding = _EMBED
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="deploy.md",
                text="backend onboarding deploy runbook",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
        ]
    )

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), llm, store
    app.dependency_overrides.clear()


def _generate(http: TestClient, **body: Any) -> dict[str, Any]:
    body.setdefault("scopes", [_SCOPE])
    body.setdefault("projectId", _PROJECT)
    response = http.post(f"{_BASE}/generate", json=body)
    assert response.status_code == 200, response.text
    return response.json()


def test_generate_returns_active_blueprint(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    outcome = _generate(http)["outcomes"][0]

    assert outcome["scope"] == _SCOPE
    assert outcome["status"] == "created"
    bp = outcome["blueprint"]
    assert bp["scope"] == _SCOPE
    assert bp["source"] == "generated"
    assert bp["version"] == "1"
    assert bp["steps"][0]["title"] == "Read the deploy runbook"


def test_generate_unchanged_corpus_is_a_noop(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    first = _generate(http)["outcomes"][0]["blueprint"]
    # The backend persists the result and passes it back as the active blueprint.
    again = _generate(http, active=[first])["outcomes"][0]

    assert again["status"] == "unchanged"
    assert again["blueprint"] is None


def test_generate_bumps_version_against_active(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, store = client

    first = _generate(http)["outcomes"][0]["blueprint"]
    store.add(
        [
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="x.md",
                text="new",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
        ]
    )
    outcome = _generate(http, active=[first])["outcomes"][0]

    assert outcome["status"] == "updated"
    assert outcome["blueprint"]["version"] == "2"
