import json
from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm
from tests.stubs.llm import StubLLMClient

_BASE = "/api/v1/onboarding/verify"


@pytest.fixture
def http() -> Generator[TestClient, None, None]:
    _override_llm("stub answer, unused unless 'knowledge' grading is exercised")
    yield TestClient(app)
    app.dependency_overrides.clear()


def _override_llm(response: str) -> None:
    app.dependency_overrides[get_llm] = lambda: StubLLMClient(
        generate_response=response
    )


def test_verify_exact_match(http: TestClient) -> None:
    response = http.post(
        f"{_BASE}",
        json={
            "type": "exact",
            "answer": "chroma",
            "canonical_answer": "Chroma",
        },
    )

    assert response.status_code == 200, response.text
    assert response.json()["passed"] is True


def test_verify_exact_missing_canonical_answer_is_422(http: TestClient) -> None:
    response = http.post(f"{_BASE}", json={"type": "exact", "answer": "chroma"})

    assert response.status_code == 422


def test_verify_attest(http: TestClient) -> None:
    response = http.post(f"{_BASE}", json={"type": "attest", "answer": "done"})

    assert response.status_code == 200, response.text
    assert response.json()["passed"] is True


def test_verify_knowledge(http: TestClient) -> None:
    _override_llm(
        json.dumps({"passed": True, "score": 1.0, "feedback": "Good.", "hint": None})
    )

    response = http.post(
        f"{_BASE}",
        json={
            "type": "knowledge",
            "question": "Why?",
            "rubric": "Because X.",
            "evidence": "X is true.[c1]",
            "answer": "Because X.",
            "attempt_no": 1,
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] is True
    assert body["hint"] is None


def test_verify_knowledge_missing_rubric_is_422(http: TestClient) -> None:
    _override_llm("irrelevant")

    response = http.post(
        f"{_BASE}", json={"type": "knowledge", "question": "q", "answer": "a"}
    )

    assert response.status_code == 422


def test_verify_unknown_type_is_422(http: TestClient) -> None:
    response = http.post(f"{_BASE}", json={"type": "bogus", "answer": "x"})

    assert response.status_code == 422


def test_verify_artifact(http: TestClient) -> None:
    _override_llm(
        json.dumps(
            {"passed": True, "score": 1.0, "feedback": "Addresses it.", "hint": None}
        )
    )

    response = http.post(
        f"{_BASE}",
        json={
            "type": "artifact",
            "question": "Fix the typo in the README.",
            "rubric": "The README install section no longer has a typo.",
            "artifact_evidence": {
                "pr_title": "Fix typo",
                "pr_body": "Fixes the typo in the install section.",
                "pr_state": "MERGED",
                "files_changed": ["README.md"],
                "checks_passed": True,
            },
        },
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] is True
    assert body["hint"] is None


def test_verify_artifact_missing_rubric_is_422(http: TestClient) -> None:
    _override_llm("irrelevant")

    response = http.post(
        f"{_BASE}",
        json={"type": "artifact", "question": "q", "artifact_evidence": {}},
    )

    assert response.status_code == 422


def test_verify_artifact_no_evidence_skips_llm_call(http: TestClient) -> None:
    _override_llm("should never be parsed")

    response = http.post(
        f"{_BASE}",
        json={"type": "artifact", "question": "q", "rubric": "r"},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["passed"] is False
    assert "pull request" in body["feedback"].lower()
