import json
from collections.abc import Callable, Generator

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_URL = "/api/v1/offboarding/interview-questions"
_BODY = {
    "projectId": "p1",
    "files": [
        {
            "repository": "github:acme/app",
            "path": "src/auth/Login.kt",
            "level": "ORPHANED",
            "score": 0.9,
            "kind": "CODE",
            "hotspot": 0.7,
            "abandonedShare": 0.2,
            "coupledWith": ["src/auth/Token.kt"],
            "reasons": ["Nobody who knows it is left."],
        }
    ],
}

_GENERATED = {
    "questions": [
        {"text": "Why retry?", "type": "rationale", "triggers": ["hotspot"]},
        {"text": "How to deploy?", "type": "operational"},
    ]
}
_JUDGED = {
    "verdicts": [
        {"index": 1, "coverage": "answered", "answer": "Helm.", "sources": ["d.md"]}
    ]
}


class _TwoStepLLM(StubLLMClient):
    def __init__(self) -> None:
        super().__init__()
        self.answers = [json.dumps(_GENERATED), json.dumps(_JUDGED)]

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        return self.answers.pop(0)


class _DownLLM(StubLLMClient):
    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        raise LLMUnavailableError("model down")


type ClientFor = Callable[[StubLLMClient], TestClient]


@pytest.fixture
def client_for() -> Generator[ClientFor]:
    def make(llm: StubLLMClient) -> TestClient:
        app.dependency_overrides[get_llm] = lambda: llm
        app.dependency_overrides[get_store] = StubVectorStore
        return TestClient(app)

    yield make
    app.dependency_overrides.clear()


def test_returns_open_questions_and_those_the_records_answer(
    client_for: ClientFor,
) -> None:
    response = client_for(_TwoStepLLM()).post(_URL, json=_BODY)

    assert response.status_code == 200
    body = response.json()
    [asked] = body["questions"]
    assert asked["text"] == "Why retry?"
    assert asked["type"] == "rationale"
    assert asked["triggers"] == ["hotspot"]
    assert asked["coverage"] == "open"
    assert asked["component"] == "github:acme/app/src/auth"
    assert asked["priority"] == pytest.approx(0.72)
    [answered] = body["answeredByRecords"]
    assert answered["recordAnswer"] == "Helm."
    assert answered["sources"] == ["d.md"]


def test_a_request_without_files_is_rejected(client_for: ClientFor) -> None:
    response = client_for(_TwoStepLLM()).post(
        _URL, json={"projectId": "p1", "files": []}
    )

    assert response.status_code == 422


def test_an_unavailable_model_answers_503(client_for: ClientFor) -> None:
    response = client_for(_DownLLM()).post(_URL, json=_BODY)

    assert response.status_code == 503
