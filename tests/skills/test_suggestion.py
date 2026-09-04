import json

import pytest

from api.schemas import SkillCatalogItem
from llm.base import Message
from llm.errors import LLMUnavailableError
from rag.types import Chunk
from skills.suggestion import _build_prompt, suggest_skills
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767
_PROJECT = "proj-1"


def _make_store_with_chunks() -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c-react",
                artifact_id="a1",
                filename="package.json",
                text="dependencies include react react-dom typescript",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            ),
            Chunk(
                id="c-spring",
                artifact_id="a2",
                filename="build.gradle.kts",
                text="implementation org.springframework.boot spring-boot-starter-web",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            ),
            Chunk(
                id="c-other-proj",
                artifact_id="a3",
                filename="secret.md",
                text="secret project 2 only kotlin rust",
                embedding=_EMBED,
                project_ids=("proj-2",),
            ),
        ]
    )
    return store


def test_suggest_skills_happy_path() -> None:
    store = _make_store_with_chunks()
    catalog = [
        SkillCatalogItem(
            id="s1", name="React", category="Frontend & UI", universal=False
        ),
        SkillCatalogItem(
            id="s2",
            name="Agile/Scrum",
            category="Processes, Methods & Architecture",
            universal=True,
        ),
        SkillCatalogItem(
            id="s3", name="Python", category="Languages & Paradigms", universal=False
        ),
    ]

    llm_payload = {
        "suggestions": [
            {
                "name": "React",
                "category": "Frontend & UI",
                "reason": "Used in frontend components",
                "confidence": "high",
                "isNew": False,
                "chunkIds": ["c-react"],
            },
            {
                "name": "Agile/Scrum",
                "category": "Processes, Methods & Architecture",
                "reason": "Standard agile team collaboration",
                "confidence": "high",
                "isNew": False,
                "chunkIds": [],
            },
        ]
    }
    llm = StubLLMClient(generate_response=json.dumps(llm_payload))
    llm.embedding = _EMBED

    result = suggest_skills(
        llm=llm,
        store=store,
        project_id=_PROJECT,
        role_name="Frontend Developer",
        role_description="Builds modern web UI",
        project_industry="Fintech",
        available_skills=catalog,
    )

    assert len(result.suggestions) == 2
    react_sugg = next(s for s in result.suggestions if s.name == "React")
    assert react_sugg.category == "Frontend & UI"
    assert react_sugg.chunk_ids == ["c-react"]
    assert not react_sugg.is_new
    assert react_sugg.confidence == "high"

    agile_sugg = next(s for s in result.suggestions if s.name == "Agile/Scrum")
    assert agile_sugg.category == "Processes, Methods & Architecture"
    assert agile_sugg.chunk_ids == []
    assert not agile_sugg.is_new


def test_grounding_gate_drops_ungrounded_project_specific_skill() -> None:
    store = _make_store_with_chunks()
    catalog = [
        SkillCatalogItem(id="s1", name="Docker", category="DevOps", universal=False),
        SkillCatalogItem(
            id="s2", name="Code Review", category="Processes", universal=True
        ),
    ]

    # LLM hallucinates Docker without citing chunk IDs, or cites a non-existent chunk ID
    llm_payload = {
        "suggestions": [
            {
                "name": "Docker",
                "category": "DevOps",
                "reason": "Containers are good",
                "confidence": "medium",
                "isNew": False,
                "chunkIds": [],  # missing evidence
            },
            {
                "name": "Kubernetes",
                "category": "DevOps",
                "reason": "Cluster orchestration",
                "confidence": "medium",
                "isNew": True,
                "chunkIds": ["c-non-existent"],  # invalid chunk id
            },
            {
                "name": "Code Review",
                "category": "Processes",
                "reason": "Universal teamwork practice",
                "confidence": "high",
                "isNew": False,
                "chunkIds": [],  # universal -> allowed without chunks
            },
        ]
    }
    llm = StubLLMClient(generate_response=json.dumps(llm_payload))
    llm.embedding = _EMBED

    result = suggest_skills(
        llm=llm,
        store=store,
        project_id=_PROJECT,
        role_name="Backend Developer",
        available_skills=catalog,
    )

    # Docker and Kubernetes should be dropped by grounding gate; Code Review passes
    assert len(result.suggestions) == 1
    assert result.suggestions[0].name == "Code Review"


def test_new_project_specific_skill_from_evidence() -> None:
    store = _make_store_with_chunks()
    # Catalog does not contain Spring Boot
    catalog = [
        SkillCatalogItem(id="s1", name="React", category="Frontend", universal=False),
    ]

    llm_payload = {
        "suggestions": [
            {
                "name": "Spring Boot",
                "category": "Backend & APIs",
                "reason": "Detected in build.gradle.kts",
                "confidence": "high",
                "isNew": True,
                "chunkIds": ["c-spring"],
            }
        ]
    }
    llm = StubLLMClient(generate_response=json.dumps(llm_payload))
    llm.embedding = _EMBED

    result = suggest_skills(
        llm=llm,
        store=store,
        project_id=_PROJECT,
        role_name="Backend Developer",
        available_skills=catalog,
    )

    assert len(result.suggestions) == 1
    sugg = result.suggestions[0]
    assert sugg.name == "Spring Boot"
    assert sugg.category == "Backend & APIs"
    assert sugg.is_new is True
    assert sugg.chunk_ids == ["c-spring"]


def test_malformed_json_degrades_gracefully() -> None:
    store = _make_store_with_chunks()
    llm = StubLLMClient(generate_response="I cannot provide JSON right now.")
    llm.embedding = _EMBED

    result = suggest_skills(
        llm=llm,
        store=store,
        project_id=_PROJECT,
        role_name="Backend Developer",
    )

    assert result.suggestions == []


def test_invalid_schema_degrades_gracefully() -> None:
    store = _make_store_with_chunks()
    llm = StubLLMClient(generate_response=json.dumps({"unexpected_key": 123}))
    llm.embedding = _EMBED

    result = suggest_skills(
        llm=llm,
        store=store,
        project_id=_PROJECT,
        role_name="Backend Developer",
    )

    assert result.suggestions == []


def test_llm_unavailable_propagates() -> None:
    store = _make_store_with_chunks()

    class FailingLLM(StubLLMClient):
        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            raise LLMUnavailableError("Ollama connection refused")

    llm = FailingLLM()
    llm.embedding = _EMBED

    with pytest.raises(LLMUnavailableError, match="Ollama connection refused"):
        suggest_skills(
            llm=llm,
            store=store,
            project_id=_PROJECT,
            role_name="Backend Developer",
        )


def test_project_isolation_filter() -> None:
    store = _make_store_with_chunks()
    catalog = [
        SkillCatalogItem(
            id="s-rust", name="Rust", category="Languages", universal=False
        ),
    ]

    # Attempt to cite chunk belonging to proj-2 for proj-1
    llm_payload = {
        "suggestions": [
            {
                "name": "Rust",
                "category": "Languages",
                "reason": "From other project",
                "confidence": "high",
                "isNew": False,
                "chunkIds": ["c-other-proj"],
            }
        ]
    }
    llm = StubLLMClient(generate_response=json.dumps(llm_payload))
    llm.embedding = _EMBED

    result = suggest_skills(
        llm=llm,
        store=store,
        project_id=_PROJECT,
        role_name="Systems Engineer",
        available_skills=catalog,
    )

    # c-other-proj is not in proj-1 retrieved chunks, so Rust must be dropped
    assert len(result.suggestions) == 0


def test_build_prompt_structure() -> None:
    catalog = [
        SkillCatalogItem(id="s1", name="React", category="Frontend", universal=False),
        SkillCatalogItem(id="s2", name="Scrum", category="Agile", universal=True),
    ]
    messages = _build_prompt(
        role_name="Fullstack Developer",
        role_description="Works on backend and frontend",
        project_industry="Fintech",
        available_skills=catalog,
        chunks=[],
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "React [Frontend]" in messages[0]["content"]
    assert "Scrum [Agile] (universal)" in messages[0]["content"]
    assert "STRICT JSON" in messages[0]["content"]

    assert messages[1]["role"] == "user"
    assert "Role: Fullstack Developer" in messages[1]["content"]
    assert "Description: Works on backend and frontend" in messages[1]["content"]
    assert "Project industry/domain: Fintech" in messages[1]["content"]
