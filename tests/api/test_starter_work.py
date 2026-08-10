import json
from collections.abc import Generator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import app
from api.dependencies import get_ingestion_metadata_store, get_llm, get_store
from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from rag.types import Chunk
from tests.conftest import parse_sse_events
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_BASE = "/api/v1/onboarding/starter-work"


def _issue_artifact(**overrides: object) -> ArtifactRecord:
    defaults: dict[str, object] = dict(
        id="a1",
        filename="issue-1.md",
        content_type="text/plain",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        artifact_type="ISSUE",
        state="OPEN",
        source_id="github:org/repo:ISSUE:1",
        source_url="https://github.com/org/repo/issues/1",
        labels=["good first issue"],
    )
    defaults.update(overrides)
    return ArtifactRecord(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def mine_client() -> Generator[tuple[TestClient, StubLLMClient], Any, None]:
    llm = StubLLMClient(
        generate_response=json.dumps(
            {
                "tasks": [
                    {
                        "source_id": "github:org/repo:ISSUE:1",
                        "safely_scoped": True,
                        "summary": "Fix a typo in the README.",
                        "competency_keys": [],
                        "rationale": "Small, well-scoped text fix.",
                    }
                ]
            }
        )
    )
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-a1",
                artifact_id="a1",
                filename="a1.md",
                text="# Fix typo in README\n\nThe install section has a typo.",
                embedding=_EMBED,
            )
        ]
    )
    metadata_store = IngestionMetadataStore(path=":memory:")
    metadata_store.save_artifact(_issue_artifact())

    app.dependency_overrides[get_llm] = lambda: llm
    app.dependency_overrides[get_store] = lambda: store
    app.dependency_overrides[get_ingestion_metadata_store] = lambda: metadata_store
    yield TestClient(app), llm
    app.dependency_overrides.clear()


def test_mine_returns_proposed_tasks(
    mine_client: tuple[TestClient, StubLLMClient],
) -> None:
    http, _ = mine_client

    response = http.post(f"{_BASE}/mine", json={})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "proposed"
    assert body["tasks"][0]["source_id"] == "github:org/repo:ISSUE:1"


def test_mine_skips_already_pooled_issue(
    mine_client: tuple[TestClient, StubLLMClient],
) -> None:
    http, _ = mine_client

    response = http.post(
        f"{_BASE}/mine",
        json={"active_source_ids": ["github:org/repo:ISSUE:1"]},
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "skipped"
    assert body["tasks"] == []


def test_mine_stream_yields_a_task_item_and_a_done_matching_the_sync_endpoint(
    mine_client: tuple[TestClient, StubLLMClient],
) -> None:
    http, _ = mine_client

    stream = http.post(f"{_BASE}/mine/stream", json={})
    assert stream.status_code == 200, stream.text
    assert stream.headers["content-type"].startswith("text/event-stream")

    events = parse_sse_events(stream.text)
    types = [e["type"] for e in events]
    assert "stage" in types
    assert "item" in types
    assert types[-1] == "done"
    assert events[-1]["result"]["tasks"][0]["source_id"] == "github:org/repo:ISSUE:1"
    assert events[-1]["result"]["status"] == "proposed"
