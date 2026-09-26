from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm
from llm.base import Message
from llm.errors import LLMUnavailableError
from tests.stubs.llm import StubLLMClient


@pytest.fixture
def client() -> Generator[tuple[TestClient, StubLLMClient], Any, None]:
    llm = StubLLMClient()

    app.dependency_overrides[get_llm] = lambda: llm

    yield TestClient(app), llm

    app.dependency_overrides.clear()


def test_grades_answers_from_llm(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client

    class TestLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            return (
                '{"results": ['
                '{"id": "q1", "correct": true, "confidence": 0.9, '
                '"feedback": "Correct: gradlew bootRun."}]}'
            )

    app.dependency_overrides[get_llm] = lambda: TestLLM()

    response = http_client.post(
        "/api/v1/grade-answers",
        json={
            "answers": [
                {
                    "id": "q1",
                    "question": "How do you start the dev server?",
                    "reference_answer": "gradlew bootRun",
                    "user_answer": "run gradlew bootRun",
                }
            ]
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "results": [
            {
                "id": "q1",
                "correct": True,
                "confidence": 0.9,
                "feedback": "Correct: gradlew bootRun.",
            }
        ]
    }


def test_blank_answer_is_incorrect_without_llm_call(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client

    class ExplodingLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            raise AssertionError("blank answers must not be sent to the LLM")

    app.dependency_overrides[get_llm] = lambda: ExplodingLLM()

    response = http_client.post(
        "/api/v1/grade-answers",
        json={
            "answers": [
                {
                    "id": "q1",
                    "question": "How do you start the dev server?",
                    "reference_answer": "gradlew bootRun",
                    "user_answer": "   ",
                }
            ]
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["correct"] is False


def test_preserves_request_order_across_mixed_answers(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client

    class TestLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            return '{"results": [{"id": "q2", "correct": true, "feedback": "ok"}]}'

    app.dependency_overrides[get_llm] = lambda: TestLLM()

    response = http_client.post(
        "/api/v1/grade-answers",
        json={
            "answers": [
                {
                    "id": "q1",
                    "question": "Q1",
                    "reference_answer": "a1",
                    "user_answer": "",
                },
                {
                    "id": "q2",
                    "question": "Q2",
                    "reference_answer": "a2",
                    "user_answer": "a2, basically",
                },
            ]
        },
    )

    assert response.status_code == 200
    ids = [r["id"] for r in response.json()["results"]]
    assert ids == ["q1", "q2"]


def test_unparseable_llm_output_marks_ungraded_as_incorrect(
    client: tuple[TestClient, StubLLMClient],
):
    http_client, _ = client

    class GarbageLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            return "not json at all"

    app.dependency_overrides[get_llm] = lambda: GarbageLLM()

    response = http_client.post(
        "/api/v1/grade-answers",
        json={
            "answers": [
                {
                    "id": "q1",
                    "question": "Q1",
                    "reference_answer": "a1",
                    "user_answer": "some answer",
                }
            ]
        },
    )

    assert response.status_code == 200
    result = response.json()["results"][0]
    assert result["correct"] is False


def test_broken_json_is_corrected_once_before_an_answer_counts_as_wrong(
    client: tuple[TestClient, StubLLMClient],
):
    """An unreadable reply used to record a right answer as wrong."""
    http_client, _ = client
    calls: list[list[Message]] = []

    class FlakyLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            calls.append(messages)
            if len(calls) == 1:
                return '{"results": [{"id": "q1", "correct": true'
            return '{"results": [{"id": "q1", "correct": true, "feedback": "Yes."}]}'

    app.dependency_overrides[get_llm] = lambda: FlakyLLM()

    response = http_client.post(
        "/api/v1/grade-answers",
        json={
            "answers": [
                {
                    "id": "q1",
                    "question": "Q1",
                    "reference_answer": "a1",
                    "user_answer": "a1",
                }
            ]
        },
    )

    assert response.json()["results"][0]["correct"] is True
    assert len(calls) == 2
    assert "could not be validated" in str(calls[1][-1])


def test_llm_failure_returns_503(client: tuple[TestClient, StubLLMClient]):
    http_client, _ = client

    class FailingLLM(StubLLMClient):
        def generate(self, messages: list[Message]) -> str:
            raise LLMUnavailableError("http://localhost:11434")

    app.dependency_overrides[get_llm] = lambda: FailingLLM()

    response = http_client.post(
        "/api/v1/grade-answers",
        json={
            "answers": [
                {
                    "id": "q1",
                    "question": "Q1",
                    "reference_answer": "a1",
                    "user_answer": "some answer",
                }
            ]
        },
    )

    assert response.status_code == 503
