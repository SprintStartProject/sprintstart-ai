import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_BASE = "/api/v1/skills"
_PROJECT = "proj-1"


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient, StubVectorStore], Any, None]:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "suggestions": [
                    {
                        "name": "React",
                        "category": "Frontend & UI",
                        "reason": "Used across web components",
                        "confidence": "high",
                        "isNew": False,
                        "chunkIds": ["c-react"],
                    },
                    {
                        "name": "Communication",
                        "category": "Soft Skills",
                        "reason": "Essential for teamwork",
                        "confidence": "high",
                        "isNew": False,
                        "chunkIds": [],
                    },
                ]
            }
        )
    )
    llm.embedding = _EMBED

    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c-react",
                artifact_id="a1",
                filename="App.tsx",
                text="import React from 'react'",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            ),
        ]
    )

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app), llm, store
    app.dependency_overrides.clear()


def test_suggest_skills_api_200(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    payload = {
        "projectId": _PROJECT,
        "roleName": "Frontend Developer",
        "roleDescription": "Builds user interfaces",
        "projectIndustry": "Healthcare",
        "availableSkills": [
            {
                "id": "s1",
                "name": "React",
                "category": "Frontend & UI",
                "universal": False,
            },
            {
                "id": "s2",
                "name": "Communication",
                "category": "Soft Skills",
                "universal": True,
            },
        ],
    }

    response = http.post(f"{_BASE}/suggest", json=payload)
    assert response.status_code == 200, response.text
    data = response.json()
    assert "suggestions" in data
    assert len(data["suggestions"]) == 2
    names = {s["name"] for s in data["suggestions"]}
    assert "React" in names
    assert "Communication" in names


def test_suggest_skills_api_503_on_llm_error(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    class FailingLLM(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            raise LLMUnavailableError("LLM daemon is down")

    app.dependency_overrides[get_llm] = lambda: FailingLLM()

    payload = {
        "projectId": _PROJECT,
        "roleName": "Backend Developer",
    }
    response = http.post(f"{_BASE}/suggest", json=payload)
    assert response.status_code == 503
    assert "LLM daemon is down" in response.json()["detail"]


def test_suggest_skills_api_degradation_on_parse_error(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    llm = StubLLMClient(generate_response="Not valid JSON at all")
    app.dependency_overrides[get_llm] = lambda: llm

    payload = {
        "projectId": _PROJECT,
        "roleName": "Backend Developer",
    }
    response = http.post(f"{_BASE}/suggest", json=payload)
    assert response.status_code == 200
    assert response.json() == {"suggestions": []}


def test_suggest_skills_api_422_missing_fields(
    client: tuple[TestClient, StubLLMClient, StubVectorStore],
) -> None:
    http, _, _ = client

    # Missing roleName and projectId
    response = http.post(f"{_BASE}/suggest", json={})
    assert response.status_code == 422
