import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.base import Message
from llm.errors import LLMUnavailableError
from onboarding.models import content_id
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

# Non-zero embedding so the stub store returns a perfect cosine match.
_EMBED = [1.0] + [0.0] * 767
_PROJECT = "project-1"

_ACCOUNTS = "Set up your accounts and access"
_LOCAL_DB = "Set up your local database"

# The AI service is stateless: the backend supplies active blueprints on every
# request. These mirror what the backend would send for a backend user.
_BLUEPRINTS: list[dict[str, Any]] = [
    {
        "scope": f"project:{_PROJECT}|global",
        "version": "1",
        "source": "authored",
        "steps": [
            {
                "id": content_id(_ACCOUNTS),
                "title": _ACCOUNTS,
                "requirement": "required",
            }
        ],
    },
    {
        "scope": f"project:{_PROJECT}|area:backend",
        "version": "1",
        "source": "authored",
        "steps": [
            {
                "id": content_id(_LOCAL_DB),
                "title": _LOCAL_DB,
                "requirement": "required",
            }
        ],
    },
]


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient()
    store = StubVectorStore()

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store

    yield TestClient(app), llm, store

    app.dependency_overrides.clear()


def _post(http: TestClient, **body: Any) -> list[dict[str, Any]]:
    body.setdefault("blueprints", _BLUEPRINTS)
    body.setdefault("projectId", _PROJECT)
    response = http.post("/api/v1/onboarding/path", json=body)
    assert response.status_code == 200
    return parse_sse_events(response.text)


def _path_event(events: list[dict[str, Any]]) -> dict[str, Any]:
    return next(e for e in events if e["type"] == "path")


def _all_step_ids(path: dict[str, Any]) -> list[str]:
    return [s["id"] for phase in path["phases"] for s in phase["steps"]]


def test_streams_stages_path_and_done(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(http, working_area="backend")
    types = [e["type"] for e in events]

    assert "stage" in types
    assert types.count("path") == 1
    assert types[-1] == "done"


def test_required_steps_always_present(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(http, working_area="backend")
    path = _path_event(events)["path"]
    ids = _all_step_ids(path)

    assert content_id("Set up your accounts and access") in ids
    assert content_id("Set up your local database") in ids


def test_unknown_working_area_falls_back_to_global_only(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(http, working_area="unknown-area")
    path = _path_event(events)["path"]
    titles = [phase["title"] for phase in path["phases"]]

    assert titles == ["Getting started"]
    assert content_id("Set up your accounts and access") in _all_step_ids(path)


def test_unseen_skill_level_does_not_crash(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    events = _post(
        http,
        working_area="backend",
        skills=[{"name": "kotlin", "level": "wizard"}],
    )
    path = _path_event(events)["path"]

    assert content_id("Set up your accounts and access") in _all_step_ids(path)


def test_empty_corpus_produces_blueprint_only_path(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, store = client
    assert store.count() == 0

    events = _post(http, working_area="backend")
    path = _path_event(events)["path"]
    origins = {s["origin"] for phase in path["phases"] for s in phase["steps"]}

    assert origins == {"blueprint"}


def test_invalid_llm_output_falls_back_without_error(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, store = client
    llm.embedding = _EMBED
    llm.generate_response = "this is not json at all"
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

    events = _post(http, working_area="backend")
    types = [e["type"] for e in events]
    path = _path_event(events)["path"]

    assert "error" not in types
    assert types[-1] == "done"
    origins = {s["origin"] for phase in path["phases"] for s in phase["steps"]}
    assert origins == {"blueprint"}


def test_grounded_llm_steps_are_added_and_cited(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, store = client
    llm.embedding = _EMBED
    llm.generate_response = json.dumps(
        {
            "steps": [
                {
                    "id": content_id("Set up your local database"),
                    "rewritten": "Configure and verify the local database connection.",
                    "chunk_ids": ["c1"],
                }
            ],
            "added": [
                {
                    "title": "Read the deploy runbook",
                    "description": "How code ships to prod.",
                    "tags": ["deploy"],
                    "chunk_ids": ["c1"],
                }
            ],
        }
    )
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="deploy.md",
                text="backend onboarding deploy runbook local db",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
        ]
    )

    events = _post(http, working_area="backend")
    path = _path_event(events)["path"]
    quality = _path_event(events)["quality"]

    llm_steps = [
        s for phase in path["phases"] for s in phase["steps"] if s["origin"] == "llm"
    ]
    assert len(llm_steps) == 1
    assert llm_steps[0]["citations"][0]["filename"] == "deploy.md"
    assert quality["grounded_ratio"] == 1.0

    # Enriched blueprint step carries the citation too.
    db_step = next(
        s
        for phase in path["phases"]
        for s in phase["steps"]
        if s["id"] == content_id("Set up your local database")
    )
    assert db_step["citations"][0]["chunk_id"] == "c1"


def test_synthesis_rewrites_step_description(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, store = client
    llm.embedding = _EMBED
    llm.generate_response = json.dumps(
        {
            "steps": [
                {
                    "id": content_id(_LOCAL_DB),
                    "rewritten": (
                        "Personalized: spin up the local DB with docker-compose."
                    ),
                    "chunk_ids": ["c1"],
                }
            ],
            "added": [],
        }
    )
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="setup.md",
                text="backend db setup docker-compose",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
        ]
    )

    events = _post(http, working_area="backend")
    path = _path_event(events)["path"]

    db_step = next(
        s
        for phase in path["phases"]
        for s in phase["steps"]
        if s["id"] == content_id(_LOCAL_DB)
    )
    assert (
        db_step["description"]
        == "Personalized: spin up the local DB with docker-compose."
    )


def test_ungrounded_llm_step_is_dropped(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, llm, store = client
    llm.embedding = _EMBED
    # Added step references a chunk id that does not exist -> no citation -> dropped.
    llm.generate_response = json.dumps(
        {"steps": [], "added": [{"title": "Ungrounded", "chunk_ids": ["missing"]}]}
    )
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="deploy.md",
                text="backend onboarding",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
        ]
    )

    events = _post(http, working_area="backend")
    path = _path_event(events)["path"]

    origins = {s["origin"] for phase in path["phases"] for s in phase["steps"]}
    assert origins == {"blueprint"}


def test_missing_request_field_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post("/api/v1/onboarding/path", json={"skills": []})

    assert response.status_code == 422


def test_missing_project_id_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post(
        "/api/v1/onboarding/path",
        json={"working_area": "backend", "blueprints": _BLUEPRINTS},
    )

    assert response.status_code == 422


def test_path_is_not_grounded_in_another_projects_corpus(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    """A step may only cite evidence from the requesting project."""
    http, llm, store = client
    llm.embedding = _EMBED
    llm.generate_response = json.dumps(
        {
            "steps": [],
            "added": [
                {
                    "title": "Read the other project's runbook",
                    "chunk_ids": ["foreign-1"],
                }
            ],
        }
    )
    store.add(
        [
            Chunk(
                id="foreign-1",
                artifact_id="foreign-artifact",
                filename="secret-runbook.md",
                text="backend onboarding deploy runbook local db",
                embedding=_EMBED,
                project_ids=("project-2",),
            )
        ]
    )

    events = _post(http, working_area="backend")
    path = _path_event(events)["path"]

    citations = [
        citation
        for phase in path["phases"]
        for step in phase["steps"]
        for citation in step["citations"]
    ]
    assert citations == []
    origins = {s["origin"] for phase in path["phases"] for s in phase["steps"]}
    assert origins == {"blueprint"}


def test_other_projects_blueprints_are_ignored(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    foreign: list[dict[str, Any]] = [
        {
            "scope": "project:project-2|global",
            "version": "1",
            "source": "authored",
            "steps": [
                {
                    "id": content_id("Foreign step"),
                    "title": "Foreign step",
                    "requirement": "required",
                }
            ],
        }
    ]

    events = _post(http, working_area="backend", blueprints=_BLUEPRINTS + foreign)
    path = _path_event(events)["path"]

    assert content_id("Foreign step") not in _all_step_ids(path)


# --- /path/yaml (synchronous YAML endpoint) ---


def test_yaml_endpoint_returns_valid_yaml(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post(
        "/api/v1/onboarding/path/yaml",
        json={
            "projectId": _PROJECT,
            "working_area": "backend",
            "blueprints": _BLUEPRINTS,
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-yaml"

    import yaml

    path = yaml.safe_load(response.text)
    assert path["working_area"] == "backend"
    assert any(
        s["id"] == content_id("Set up your accounts and access")
        for phase in path["phases"]
        for s in phase["steps"]
    )


def test_yaml_endpoint_includes_quality_report(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post(
        "/api/v1/onboarding/path/yaml",
        json={
            "projectId": _PROJECT,
            "working_area": "backend",
            "blueprints": _BLUEPRINTS,
        },
    )

    import yaml

    path = yaml.safe_load(response.text)
    assert "quality" in path
    assert "coverage" in path["quality"]
    assert "score" in path["quality"]


def test_yaml_endpoint_missing_field_returns_422(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    response = http.post("/api/v1/onboarding/path/yaml", json={"skills": []})

    assert response.status_code == 422


def test_yaml_endpoint_returns_503_when_llm_unavailable(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    class _DownLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            raise LLMUnavailableError("LLM backend unreachable")

        def embed(self, text: str) -> list[float]:
            raise LLMUnavailableError("LLM backend unreachable")

    app.dependency_overrides[get_llm] = lambda: _DownLLM()

    response = http.post(
        "/api/v1/onboarding/path/yaml",
        json={
            "projectId": _PROJECT,
            "working_area": "backend",
            "blueprints": _BLUEPRINTS,
        },
    )

    assert response.status_code == 503
