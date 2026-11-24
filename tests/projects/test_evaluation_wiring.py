import json
import logging

import pytest

from corpus.types import Chunk
from llm.base import Message
from projects.evaluation import evaluate_industry
from rag.hybrid import BM25Index, BM25IndexCache
from store.base import VectorStore
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_PROJECT = "3f1c0b1e-1f4d-4a5e-9b6a-0d2c8f7e5a11"
_EMBED = [1.0, 0.0] * 384


class _RecordingLLM(StubLLMClient):
    def __init__(self, raw: str) -> None:
        super().__init__(generate_response=raw)
        self.embedding = _EMBED
        self.prompts: list[list[Message]] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.prompts.append(messages)
        return super().generate(messages, temperature=temperature)


class _CountingCache(BM25IndexCache):
    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0

    def get(self, store: VectorStore) -> BM25Index:
        self.lookups += 1
        return super().get(store)


def _store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="art-c1",
                filename="README.md",
                text="Core banking ledger.",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
        ]
    )
    return store


def test_the_model_is_sent_the_retrieved_evidence() -> None:
    llm = _RecordingLLM(json.dumps({"industry": "Fintech", "confidence": "high"}))

    evaluate_industry(llm, _store(), _PROJECT)

    assert llm.prompts[0][1]["content"] == (
        "Artifact evidence:\n[c1] (README.md) Core banking ledger."
    )


def test_the_given_bm25_cache_is_the_one_used() -> None:
    cache = _CountingCache()

    evaluate_industry(
        _RecordingLLM(json.dumps({"industry": "Fintech"})),
        _store(),
        _PROJECT,
        bm25_cache=cache,
    )

    assert cache.lookups == 1


def test_an_unparseable_answer_is_logged_with_the_project(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = evaluate_industry(_RecordingLLM("not json"), _store(), _PROJECT)

    assert result.industry == ""
    assert (
        caplog.records[0]
        .getMessage()
        .startswith(
            f"Industry evaluation for project {_PROJECT} failed to parse LLM output, "
            "falling back to empty: "
        )
    )
