"""End-to-end golden question / trick prompt test harness.

Ingests the demo corpus against a real Ollama instance, then drives the
retrieval layer directly (``rag.retriever.retrieve``) — the layer the corpus
actually pins:

- Asserts each golden question retrieves the expected source file.
- Asserts each trick prompt retrieves nothing (nothing relevant found).

Marked pytest.mark.integration — skipped automatically when Ollama is not
reachable, so offline CI stays green.
"""

from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
import yaml
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_store
from rag.retriever import retrieve
from store.chroma_store import ChromaVectorStore
from tests.conftest import llm_required

CORPUS_DIR = Path(__file__).parent / "demo-corpus"
GOLDEN_QUESTIONS_FILE = Path(__file__).parent / "golden_questions.yaml"
TRICK_PROMPTS_FILE = Path(__file__).parent / "trick_prompts.yaml"

_GOLDEN_MIN_SCORE = 0.6
_TRICK_MIN_SCORE = 0.65


@pytest.fixture(scope="module")
def ingested_store() -> Generator[ChromaVectorStore, Any, None]:
    """Ingest the full demo corpus once per module against a real LLM.

    Uses a fresh in-memory ChromaDB store so tests are isolated from any
    persistent data directory on the developer's machine. The assertions below
    retrieve from this store directly; the app's dependency override just keeps
    the ingestion requests on the same one.
    """
    store = ChromaVectorStore(collection_name="golden-test-chunks")
    app.dependency_overrides[get_store] = lambda: store

    client = TestClient(app)

    for doc_file in sorted(CORPUS_DIR.glob("*.md")):
        response = client.post(
            "/api/v1/ingest",
            json={
                "artifact_id": doc_file.stem,
                "filename": doc_file.name,
                "content": doc_file.read_text(),
            },
        )
        assert response.status_code == 200, (
            f"Corpus ingestion failed for {doc_file.name}: {response.text}"
        )

    yield store

    app.dependency_overrides.clear()


@pytest.mark.integration
@llm_required
def test_golden_questions(ingested_store: ChromaVectorStore) -> None:
    """Every golden question must retrieve at least one chunk from the
    expected file.
    """
    cases: list[dict[str, str]] = yaml.safe_load(GOLDEN_QUESTIONS_FILE.read_text())

    failures: list[str] = []
    for case in cases:
        question = case["question"]
        expected = case["expected_citation_filename"]

        chunks = retrieve(
            question, get_llm(), ingested_store, min_score=_GOLDEN_MIN_SCORE
        )
        retrieved = {chunk.filename for chunk in chunks}

        if expected not in retrieved:
            failures.append(
                f"\nQuestion  : {question!r}"
                f"\nExpected  : {expected}"
                f"\nRetrieved : {retrieved or '(none)'}\n"
            )

    assert not failures, "Golden question assertion(s) failed:\n" + "".join(failures)


@pytest.mark.integration
@llm_required
def test_trick_prompts_retrieve_nothing(ingested_store: ChromaVectorStore) -> None:
    """Trick prompts about absent content must retrieve nothing."""
    cases: list[dict[str, str]] = yaml.safe_load(TRICK_PROMPTS_FILE.read_text())

    failures: list[str] = []
    for case in cases:
        prompt = case["prompt"]

        chunks = retrieve(prompt, get_llm(), ingested_store, min_score=_TRICK_MIN_SCORE)

        if chunks:
            retrieved = {chunk.filename for chunk in chunks}
            failures.append(
                f"\nPrompt    : {prompt!r}"
                f"\nExpected  : no chunks"
                f"\nRetrieved : {retrieved}\n"
            )

    assert not failures, "Trick prompt(s) unexpectedly retrieved chunks:\n" + "".join(
        failures
    )
