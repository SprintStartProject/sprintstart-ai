from dataclasses import replace
from typing import Any

import pytest

from corpus.filters import PROJECT_IDS_METADATA_KEY, project_metadata_key
from corpus.ids import ChunkId
from corpus.types import Chunk
from store.chroma_records import (
    NO_POSITION,
    chunk_metadata,
    chunks_from_get_result,
    chunks_without_embeddings_from_get_result,
    retrieval_fingerprint,
    scored_chunk,
    visible_chunks,
)


def _chunk(**overrides: Any) -> Chunk:
    base = Chunk(
        id="c1",
        artifact_id="a1",
        filename="f.py",
        text="body",
        embedding=[0.5, 1.5],
        kind="code",
        position=3,
        source_role="test",
        source_url="https://x/y",
        artifact_type="file",
        language="python",
        connector_id="conn",
        connector_source_id="src",
        source_system="GITHUB",
        created_at="2024-05-06T07:08:09Z",
        project_ids=("p1", "p2"),
        start_line=7,
        start_page=2,
    )
    return replace(base, **overrides)


def _get_result(chunk: Chunk) -> dict[str, object]:
    return {
        "ids": [chunk.id],
        "documents": [chunk.text],
        "metadatas": [chunk_metadata(chunk)],
        "embeddings": [chunk.embedding],
    }


def test_chunk_metadata_round_trips_through_get_result() -> None:
    chunk = _chunk()

    assert chunks_from_get_result(_get_result(chunk)) == [chunk]


def test_chunk_metadata_without_optional_fields_round_trips() -> None:
    chunk = _chunk(
        kind="text",
        position=None,
        source_role="primary",
        source_url=None,
        artifact_type=None,
        language=None,
        connector_id=None,
        connector_source_id=None,
        source_system=None,
        created_at=None,
        project_ids=(),
        start_line=None,
        start_page=None,
    )
    metadata = chunk_metadata(chunk)

    assert metadata["position"] == NO_POSITION
    assert "start_line" not in metadata
    assert "start_page" not in metadata
    assert chunks_from_get_result(_get_result(chunk)) == [chunk]


def test_chunk_metadata_marks_each_project_for_server_side_filtering() -> None:
    metadata = chunk_metadata(_chunk())

    assert metadata[project_metadata_key("p1")] is True
    assert metadata[project_metadata_key("p2")] is True
    assert PROJECT_IDS_METADATA_KEY in metadata
    assert project_metadata_key("other") not in metadata


def test_get_result_accepts_numpy_like_embeddings() -> None:
    class Embeddings:
        def tolist(self) -> list[list[float]]:
            return [[0.25]]

    raw = _get_result(_chunk(embedding=[0.0]))
    raw["embeddings"] = Embeddings()

    assert chunks_from_get_result(raw)[0].embedding == [0.25]


def test_get_result_rejects_an_unknown_chunk_kind() -> None:
    raw = _get_result(_chunk())
    metadata = dict(chunk_metadata(_chunk()))
    metadata["kind"] = "video"
    raw["metadatas"] = [metadata]

    with pytest.raises(ValueError, match="Unknown chunk kind 'video'"):
        chunks_from_get_result(raw)


def test_unknown_source_role_reads_back_as_primary() -> None:
    raw = _get_result(_chunk())
    metadata = dict(chunk_metadata(_chunk()))
    metadata["source_role"] = "mystery"
    raw["metadatas"] = [metadata]

    assert chunks_from_get_result(raw)[0].source_role == "primary"


def test_chunks_without_embeddings_leave_the_embedding_empty() -> None:
    chunk = _chunk()
    raw = _get_result(chunk)
    del raw["embeddings"]

    assert chunks_without_embeddings_from_get_result(raw) == [
        replace(chunk, embedding=[])
    ]


def test_scored_chunk_carries_the_score_and_the_metadata() -> None:
    chunk = _chunk()

    scored = scored_chunk(ChunkId("c1"), "body", chunk_metadata(chunk), 0.75)

    assert scored.score == 0.75
    assert (scored.id, scored.text, scored.kind, scored.position) == (
        "c1",
        "body",
        "code",
        3,
    )
    assert scored.project_ids == ("p1", "p2")
    assert (scored.start_line, scored.start_page) == (7, 2)
    assert scored.source_system == "GITHUB"


def test_visible_chunks_drops_revoked_artifacts() -> None:
    keep = _chunk(id="k", artifact_id="keep")
    drop = _chunk(id="d", artifact_id="drop")

    assert visible_chunks([keep, drop], frozenset({"drop"})) == [keep]
    assert visible_chunks([keep, drop], frozenset()) == [keep, drop]


def test_retrieval_fingerprint_changes_with_filter_metadata() -> None:
    base = chunk_metadata(_chunk())
    moved = chunk_metadata(_chunk(project_ids=("p3",)))
    retested = chunk_metadata(_chunk(source_role="primary"))

    fingerprints = {
        retrieval_fingerprint("c1", base),
        retrieval_fingerprint("c1", moved),
        retrieval_fingerprint("c1", retested),
        retrieval_fingerprint("c2", base),
    }
    assert len(fingerprints) == 4
    assert retrieval_fingerprint("c1", base) == retrieval_fingerprint("c1", base)
