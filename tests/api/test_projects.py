import json
from collections.abc import Generator
from typing import Any

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_PROJECT_ID = "3f1c0b1e-1f4d-4a5e-9b6a-0d2c8f7e5a11"
_URL = f"/api/v1/projects/{_PROJECT_ID}/industry/evaluate"
_EMBED = [1.0, 0.0] * 384


def _chunk(chunk_id: str, text: str, project_id: str = _PROJECT_ID) -> Chunk:
    return Chunk(
        id=chunk_id,
        artifact_id=f"art-{chunk_id}",
        filename="README.md",
        text=text,
        embedding=_EMBED,
        project_ids=(project_id,),
    )


def _llm(payload: dict[str, object] | str) -> StubLLMClient:
    raw = json.dumps(payload) if isinstance(payload, dict) else payload
    client = StubLLMClient(generate_response=raw)
    client.embedding = _EMBED
    return client


def _client(llm: Any, store: StubVectorStore) -> Generator[TestClient, Any, None]:
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_evaluate_industry_endpoint_success() -> None:
    store = StubVectorStore()
    store.add([_chunk("c1", "Financial ledger and accounting reconciliation system.")])

    llm = _llm(
        {
            "industry": "Fintech / Accounting",
            "confidence": "high",
            "evidence": ["Financial ledger and accounting system"],
        }
    )

    client = next(_client(llm, store))
    response = client.post(_URL, json={"projectId": _PROJECT_ID})

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["industry"] == "Fintech / Accounting"
    assert data["confidence"] == "high"
    assert data["evidence"] == ["Financial ledger and accounting system"]


def test_evaluate_industry_endpoint_degrades_on_malformed_llm() -> None:
    store = StubVectorStore()
    store.add([_chunk("c1", "Robotics simulation engine.")])
    llm = _llm("invalid non-json output")

    client = next(_client(llm, store))
    response = client.post(_URL, json={"projectId": _PROJECT_ID})

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["industry"] == ""
    assert data["confidence"] == "low"
    assert data["evidence"] == []


def test_evaluate_industry_endpoint_llm_unavailable_503() -> None:
    store = StubVectorStore()
    store.add([_chunk("c1", "Robotics simulation engine.")])

    class FailingLLM(StubLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = _EMBED

        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            raise LLMUnavailableError("LLM provider down")

    client = next(_client(FailingLLM(), store))
    response = client.post(_URL, json={"projectId": _PROJECT_ID})

    assert response.status_code == 503, response.text
    assert "LLM provider down" in response.text


def test_evaluate_industry_endpoint_missing_body_422() -> None:
    store = StubVectorStore()
    client = next(_client(_llm({}), store))

    response = client.post(_URL, json={})
    assert response.status_code == 422


def test_evaluate_industry_endpoint_project_isolation() -> None:
    store = StubVectorStore()
    store.add(
        [_chunk("c1", "Biotechnology gene sequencing.", project_id="other-project")]
    )
    llm = _llm({"industry": "Biotech", "confidence": "high", "evidence": []})

    client = next(_client(llm, store))
    response = client.post(_URL, json={"projectId": _PROJECT_ID})

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["industry"] == ""
    assert data["confidence"] == "low"
    assert data["evidence"] == []
