import json

import pytest

from llm.base import Message
from llm.errors import LLMUnavailableError
from projects.evaluation import _build_prompt, evaluate_industry
from rag.types import Chunk, ScoredChunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_PROJECT_ID = "3f1c0b1e-1f4d-4a5e-9b6a-0d2c8f7e5a11"
_EMBED = [1.0, 0.0] * 384


def _chunk(
    chunk_id: str,
    text: str,
    project_id: str = _PROJECT_ID,
    filename: str = "README.md",
    source_role: str = "primary",
) -> Chunk:
    return Chunk(
        id=chunk_id,
        artifact_id=f"art-{chunk_id}",
        filename=filename,
        text=text,
        embedding=_EMBED,
        project_ids=(project_id,),
        source_role=source_role,  # type: ignore[arg-type]
    )


def _llm(payload: dict[str, object] | str) -> StubLLMClient:
    raw = json.dumps(payload) if isinstance(payload, dict) else payload
    client = StubLLMClient(generate_response=raw)
    client.embedding = _EMBED
    return client


def test_evaluate_industry_happy_path() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk("c1", "Core banking transaction engine and ledger service."),
            _chunk("c2", "PCI-DSS compliant payment processing gateway."),
        ]
    )

    llm = _llm(
        {
            "industry": "Fintech / Banking",
            "confidence": "high",
            "evidence": ["Core banking transaction engine", "PCI-DSS payment gateway"],
        }
    )

    result = evaluate_industry(llm, store, _PROJECT_ID)

    assert result.industry == "Fintech / Banking"
    assert result.confidence == "high"
    assert result.evidence == [
        "Core banking transaction engine",
        "PCI-DSS payment gateway",
    ]


def test_evaluate_industry_project_isolation() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk(
                "c1",
                "Medical imaging and patient record service.",
                project_id="other-project",
            ),
        ]
    )
    llm = _llm({"industry": "Healthcare", "confidence": "high", "evidence": []})

    result = evaluate_industry(llm, store, _PROJECT_ID)

    # Scoped to _PROJECT_ID, which has no chunks -> graceful fallback
    assert result.industry == ""
    assert result.confidence == "low"
    assert result.evidence == []


def test_evaluate_industry_empty_store() -> None:
    store = StubVectorStore()
    llm = _llm({"industry": "E-Commerce", "confidence": "high", "evidence": []})

    result = evaluate_industry(llm, store, _PROJECT_ID)

    assert result.industry == ""
    assert result.confidence == "low"
    assert result.evidence == []


def test_evaluate_industry_malformed_json_degrades_gracefully() -> None:
    store = StubVectorStore()
    store.add([_chunk("c1", "Quantum computing circuit simulation library.")])
    llm = _llm("I think this project is about Quantum Computing, but not in JSON.")

    result = evaluate_industry(llm, store, _PROJECT_ID)

    assert result.industry == ""
    assert result.confidence == "low"
    assert result.evidence == []


def test_evaluate_industry_invalid_schema_degrades_gracefully() -> None:
    store = StubVectorStore()
    store.add([_chunk("c1", "Quantum computing circuit simulation library.")])
    # Missing required 'confidence' field or wrong type
    llm = _llm({"industry": 123, "confidence": "invalid_level"})

    result = evaluate_industry(llm, store, _PROJECT_ID)

    assert result.industry == ""
    assert result.confidence == "low"
    assert result.evidence == []


def test_evaluate_industry_llm_unavailable_propagates() -> None:
    store = StubVectorStore()
    store.add([_chunk("c1", "Some software project.")])

    class FailingLLM(StubLLMClient):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = _EMBED

        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            raise LLMUnavailableError("Ollama daemon down")

    with pytest.raises(LLMUnavailableError, match="Ollama daemon down"):
        evaluate_industry(FailingLLM(), store, _PROJECT_ID)


def test_evaluate_industry_excludes_test_role_chunks() -> None:
    store = StubVectorStore()
    store.add(
        [
            _chunk("c1", "Test mock for payment processing.", source_role="test"),
        ]
    )
    llm = _llm({"industry": "Fintech", "confidence": "high", "evidence": []})

    result = evaluate_industry(llm, store, _PROJECT_ID)

    # Only test chunks exist -> excluded from grounding -> empty fallback
    assert result.industry == ""
    assert result.confidence == "low"
    assert result.evidence == []


def test_build_prompt_structure() -> None:
    chunks = [
        ScoredChunk(
            id="chk-1",
            artifact_id="art-1",
            filename="architecture.md",
            text="Distributed event broker using Kafka.",
            score=0.85,
        )
    ]
    messages = _build_prompt(chunks)

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "STRICT JSON" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "[chk-1]" in messages[1]["content"]
    assert "architecture.md" in messages[1]["content"]
    assert "Distributed event broker" in messages[1]["content"]
