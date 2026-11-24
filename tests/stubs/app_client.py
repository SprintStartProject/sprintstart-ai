"""A TestClient for the app with a scripted model and a one-chunk corpus."""

from collections.abc import Generator
from typing import Any

from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_llm, get_source_state_store, get_store
from corpus.types import Chunk
from store.source_state_store import SourceStateStore
from tests.stubs.llm import ScriptedLLMClient
from tests.stubs.store import StubVectorStore


def scripted_client(llm: ScriptedLLMClient) -> Generator[TestClient, Any, None]:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="auth.md",
                text="the login handler lives in auth.py",
                embedding=[0.0] * 768,
                source_system="GITHUB",
                project_ids=("p1",),
            )
        ]
    )
    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_source_state_store] = lambda: SourceStateStore(
        ":memory:"
    )
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()
