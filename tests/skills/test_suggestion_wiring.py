import json
import logging

import pytest

from corpus.source_role import SourceRole
from corpus.types import Chunk
from llm.base import Message
from rag.hybrid import BM25Index, BM25IndexCache
from skills.models import SkillCatalogEntry
from skills.suggestion import suggest_skills
from store.base import VectorStore
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_PROJECT = "proj-1"


class _RecordingLLM(StubLLMClient):
    def __init__(self, payload: object) -> None:
        raw = payload if isinstance(payload, str) else json.dumps(payload)
        super().__init__(generate_response=raw)
        self.embedding = _EMBED
        self.prompts: list[list[Message]] = []
        self.embedded: list[str] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.prompts.append(messages)
        return super().generate(messages, temperature=temperature)

    def embed(self, text: str) -> list[float]:
        self.embedded.append(text)
        return super().embed(text)


class _CountingCache(BM25IndexCache):
    def __init__(self) -> None:
        super().__init__()
        self.lookups = 0

    def get(self, store: VectorStore) -> BM25Index:
        self.lookups += 1
        return super().get(store)


def _chunk(chunk_id: str, filename: str, role: SourceRole = "primary") -> Chunk:
    return Chunk(
        id=chunk_id,
        artifact_id=f"art-{chunk_id}",
        filename=filename,
        text=f"text of {filename}",
        embedding=_EMBED,
        project_ids=(_PROJECT,),
        source_role=role,
    )


def _store() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            _chunk("c-react", "package.json"),
            _chunk("c-test", "app.test.ts", role="test"),
            Chunk(
                id="c-other",
                artifact_id="art-other",
                filename="other.md",
                text="text of other.md",
                embedding=_EMBED,
                project_ids=("proj-2",),
            ),
        ]
    )
    return store


def _suggestion(name: str, **fields: object) -> dict[str, object]:
    return {"name": name, "reason": "r", "confidence": "high", **fields}


def test_the_model_is_sent_a_prompt_built_from_the_request_and_the_evidence() -> None:
    llm = _RecordingLLM({"suggestions": []})

    suggest_skills(
        llm,
        _store(),
        project_id=_PROJECT,
        role_name="Frontend Developer",
        role_description="Builds UI",
        project_industry="Fintech",
        available_skills=[
            SkillCatalogEntry(id="s1", name="React", category="Frontend")
        ],
    )

    system, user = llm.prompts[0][0]["content"], llm.prompts[0][1]["content"]
    assert "- [s1] React [Frontend]" in system
    assert "[c-react] (package.json) text of package.json" in system
    assert "c-test" not in system, "test material is never grounding evidence"
    assert "c-other" not in system, "another project's material is never visible"
    assert user == (
        "Role: Frontend Developer\n"
        "Description: Builds UI\n"
        "Project industry/domain: Fintech"
    )


def test_the_retrieval_query_joins_role_description_and_industry() -> None:
    llm = _RecordingLLM({"suggestions": []})

    suggest_skills(
        llm,
        _store(),
        project_id=_PROJECT,
        role_name="Dev",
        role_description="Builds UI",
        project_industry="Fintech",
    )

    assert "Dev Builds UI Fintech" in llm.embedded


def test_a_missing_description_is_shown_as_not_provided() -> None:
    llm = _RecordingLLM({"suggestions": []})

    suggest_skills(llm, _store(), project_id=_PROJECT, role_name="Dev")

    assert "Description: (none provided)" in llm.prompts[0][1]["content"]
    assert llm.embedded == ["Dev"]


def test_nothing_is_retrieved_for_an_empty_store_or_without_a_project() -> None:
    empty = _RecordingLLM({"suggestions": []})
    suggest_skills(empty, StubVectorStore(), project_id=_PROJECT, role_name="Dev")

    no_project = _RecordingLLM({"suggestions": []})
    suggest_skills(no_project, _store(), role_name="Dev")

    assert empty.embedded == []
    assert no_project.embedded == []
    assert "(no project artifacts retrieved)" in no_project.prompts[0][0]["content"]


def test_the_given_bm25_cache_is_the_one_used() -> None:
    cache = _CountingCache()

    suggest_skills(
        _RecordingLLM({"suggestions": []}),
        _store(),
        project_id=_PROJECT,
        role_name="Dev",
        bm25_cache=cache,
    )

    assert cache.lookups == 1


def test_an_unparseable_answer_is_logged_with_role_and_project(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        result = suggest_skills(
            _RecordingLLM("not json at all"),
            _store(),
            project_id=_PROJECT,
            role_name="Dev",
        )

    assert result.suggestions == []
    assert (
        caplog.records[0]
        .getMessage()
        .startswith(
            "Failed to parse LLM skill suggestions for role 'Dev' on project 'proj-1': "
        )
    )


def test_blank_and_repeated_names_are_skipped_without_ending_the_list() -> None:
    llm = _RecordingLLM(
        {
            "suggestions": [
                _suggestion("   ", universal=True),
                _suggestion("React", chunkIds=["c-react"]),
                _suggestion("react", chunkIds=["c-react"]),
                _suggestion("Docker", universal=True),
                _suggestion("DOCKER", universal=True),
                _suggestion("Git", universal=True),
            ]
        }
    )

    result = suggest_skills(llm, _store(), project_id=_PROJECT, role_name="Dev")

    assert [s.name for s in result.suggestions] == ["React", "Docker", "Git"]
    assert result.suggestions[0].chunk_ids == ["c-react"]


def test_a_known_universal_category_needs_no_evidence_whatever_its_case() -> None:
    llm = _RecordingLLM(
        {
            "suggestions": [
                _suggestion("Empathy", category=" SOFT Skills "),
                _suggestion("Kafka", category="Messaging"),
                _suggestion("Estimation", universal=True),
            ]
        }
    )

    result = suggest_skills(llm, _store(), project_id=_PROJECT, role_name="Dev")

    assert [s.name for s in result.suggestions] == ["Empathy", "Estimation"]
    assert all(s.is_new for s in result.suggestions)
    assert result.suggestions[0].category == " SOFT Skills "


def test_only_chunks_that_were_retrieved_count_as_evidence() -> None:
    llm = _RecordingLLM(
        {
            "suggestions": [
                _suggestion("React", chunkIds=["c-react", "invented", "c-other"]),
                _suggestion("Kafka", chunkIds=["invented"]),
            ]
        }
    )

    result = suggest_skills(llm, _store(), project_id=_PROJECT, role_name="Dev")

    assert [(s.name, s.chunk_ids) for s in result.suggestions] == [
        ("React", ["c-react"])
    ]


def test_a_dropped_ungrounded_skill_is_logged(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = _RecordingLLM({"suggestions": [_suggestion("Kafka", chunkIds=[])]})

    with caplog.at_level(logging.DEBUG, logger="skills.suggestion"):
        suggest_skills(llm, _store(), project_id=_PROJECT, role_name="Dev")

    assert "Dropping ungrounded skill suggestion 'Kafka' for role 'Dev'" in [
        record.getMessage() for record in caplog.records
    ]
