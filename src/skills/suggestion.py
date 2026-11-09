"""AI-supported skill suggestion for project roles."""

import logging
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from corpus.filters import RetrievalFilters
from corpus.source_role import GROUNDING_EXCLUDED_ROLES
from corpus.types import ScoredChunk
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.retriever import get_bm25_cache
from skills.models import SkillCatalogEntry, SkillSuggestion, SkillSuggestionResult
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 12
_MIN_SCORE = 0.3
_MAX_CATALOG_ITEMS = 200
_MAX_CATALOG_CHARS = 6000
_MAX_EVIDENCE_CHUNKS = 12
_MAX_CHUNK_TEXT_CHARS = 1000

_KNOWN_UNIVERSAL_CATEGORIES = frozenset(
    {
        "soft skills",
        "processes, methods & architecture",
        "agile",
        "agile & processes",
        "processes",
        "collaboration",
        "leadership & management",
    }
)


class _Suggestion(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1)
    category: str | None = None
    reason: str = Field(default="")
    confidence: Literal["high", "medium", "low"] = "medium"
    universal: bool = False
    is_new: bool = Field(alias="isNew", default=False)
    chunk_ids: list[str] = Field(alias="chunkIds", default_factory=list[str])


class _Payload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    suggestions: list[_Suggestion] = Field(default_factory=list[_Suggestion])


def _format_catalog(available_skills: list[SkillCatalogEntry]) -> str:
    if not available_skills:
        return "(empty catalog)"
    lines: list[str] = []
    total_chars = 0
    for s in available_skills[:_MAX_CATALOG_ITEMS]:
        cat = f" [{s.category}]" if s.category else ""
        univ = " (universal)" if s.universal else ""
        line = f"- [{s.id}] {s.name}{cat}{univ}"
        if total_chars + len(line) > _MAX_CATALOG_CHARS:
            lines.append("- ... (remaining catalog truncated)")
            break
        lines.append(line)
        total_chars += len(line) + 1
    return "\n".join(lines)


def _format_chunks(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return "(no project artifacts retrieved)"
    lines: list[str] = []
    for c in chunks[:_MAX_EVIDENCE_CHUNKS]:
        text = c.text[:_MAX_CHUNK_TEXT_CHARS]
        lines.append(f"[{c.id}] ({c.filename}) {text}")
    return "\n".join(lines)


def _build_prompt(
    role_name: str,
    role_description: str,
    project_industry: str | None,
    available_skills: list[SkillCatalogEntry],
    chunks: list[ScoredChunk],
) -> list[Message]:
    system = (
        "You recommend software skills a person in this role should self-assess for "
        "THIS project.\n\n"
        "Available skills catalog (existing skills in system):\n"
        f"{_format_catalog(available_skills)}\n\n"
        "Project artifacts (evidence):\n"
        f"{_format_chunks(chunks)}\n\n"
        "Task:\n"
        "Recommend skills relevant to both the role AND this project.\n"
        "Follow these two-class grounding rules strictly:\n"
        "1. For UNIVERSAL skills (marked '(universal)' in catalog, or role-typical "
        "soft skills / agile practices):\n"
        "   - Set universal=true.\n"
        "   - Recommend role-typical universal skills WITHOUT requiring evidence "
        "(chunkIds can be empty).\n"
        "   - Standard practices (e.g. Code Review, Agile/Scrum, Communication).\n"
        "2. For PROJECT-SPECIFIC skills (technologies, frameworks, databases, tools):\n"
        "   - Set universal=false.\n"
        "   - MUST cite at least one chunk_id from the evidence in chunkIds.\n"
        "   - Only suggest if project artifacts actually demonstrate their usage.\n"
        "   - If missing in catalog, suggest with isNew=true (and chunkIds).\n\n"
        "STRICT JSON only (no markdown fences, no explanatory text outside JSON):\n"
        '{"suggestions": [{"name": str, "category": str | null, "reason": str, '
        '"confidence": "high"|"medium"|"low", "universal": bool, "isNew": bool, '
        '"chunkIds": [str]}]}'
    )

    user_parts = [
        f"Role: {role_name}",
        f"Description: {role_description or '(none provided)'}",
    ]
    if project_industry:
        user_parts.append(f"Project industry/domain: {project_industry}")

    return [
        Message(role="system", content=system),
        Message(role="user", content="\n".join(user_parts)),
    ]


@dataclass(frozen=True)
class _Role:
    """The role skills are suggested for."""

    name: str
    description: str
    industry: str | None


@dataclass(frozen=True)
class _Search:
    """What is retrieved as evidence, and where from."""

    query: str
    filters: RetrievalFilters
    cache: BM25IndexCache


@dataclass(frozen=True)
class _Evidence:
    """What a suggestion is checked against: the catalog and the chunks retrieved."""

    catalog: dict[str, SkillCatalogEntry]
    retrieved_chunk_ids: set[str]
    role_name: str


def _query(role: _Role) -> str:
    parts = [role.name]
    if role.description:
        parts.append(role.description)
    if role.industry:
        parts.append(role.industry)
    return " ".join(parts)


def _retrieve(llm: LLMClient, store: VectorStore, search: _Search) -> list[ScoredChunk]:
    if search.filters.project_id is None or store.count() == 0:
        return []
    return hybrid_retrieve(
        question=search.query,
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=search.cache,
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
        filters=search.filters,
    )


def _read(raw: str, role: _Role, search: _Search) -> _Payload | None:
    try:
        return _Payload.model_validate_json(extract_json_object(raw))
    except (ValueError, ValidationError) as exc:
        logger.warning(
            "Failed to parse LLM skill suggestions for role %r on project %r: %s",
            role.name,
            search.filters.project_id,
            exc,
        )
        return None


def _is_universal_category(category: str | None) -> bool:
    return bool(category and category.strip().lower() in _KNOWN_UNIVERSAL_CATEGORIES)


def _vet(item: _Suggestion, evidence: _Evidence) -> SkillSuggestion | None:
    """The suggestion if the evidence allows it, else ``None``.

    Universal skills pass without citations. A project-specific skill must cite at
    least one chunk that was actually retrieved.
    """
    catalog_entry = evidence.catalog.get(item.name.strip().lower())
    if catalog_entry is not None:
        is_universal = catalog_entry.universal
        name = catalog_entry.name
        category = catalog_entry.category or item.category
        is_new = False
    else:
        is_universal = item.universal or _is_universal_category(item.category)
        name = item.name.strip()
        category = item.category
        is_new = True

    valid_chunk_ids = [
        cid for cid in item.chunk_ids if cid in evidence.retrieved_chunk_ids
    ]
    if is_universal:
        fallback = f"Recommended universal skill for {evidence.role_name}."
    elif valid_chunk_ids:
        fallback = (
            "Project-specific skill identified from artifacts for "
            f"{evidence.role_name}."
        )
    else:
        logger.debug(
            "Dropping ungrounded skill suggestion %r for role %r",
            item.name,
            evidence.role_name,
        )
        return None
    return SkillSuggestion(
        name=name,
        category=category,
        reason=item.reason or fallback,
        confidence=item.confidence,
        is_new=is_new,
        chunk_ids=valid_chunk_ids,
    )


def _ground(payload: _Payload, evidence: _Evidence) -> list[SkillSuggestion]:
    """Keep each skill once, in the model's order, if the evidence allows it."""
    suggestions: list[SkillSuggestion] = []
    seen_names: set[str] = set()
    for item in payload.suggestions:
        norm_name = item.name.strip().lower()
        if not norm_name or norm_name in seen_names:
            continue
        suggestion = _vet(item, evidence)
        if suggestion is None:
            continue
        seen_names.add(norm_name)
        suggestions.append(suggestion)
    return suggestions


def suggest_skills(
    llm: LLMClient,
    store: VectorStore,
    *,
    project_id: str | None = None,
    role_name: str,
    role_description: str = "",
    project_industry: str | None = None,
    available_skills: list[SkillCatalogEntry] | None = None,
    bm25_cache: BM25IndexCache | None = None,
) -> SkillSuggestionResult:
    """Suggest fitting skills for a project role, grounded in project artifacts."""
    if available_skills is None:
        available_skills = []
    role = _Role(role_name, role_description, project_industry)
    cache = bm25_cache if bm25_cache is not None else get_bm25_cache()
    search = _Search(_query(role), RetrievalFilters(project_id=project_id), cache)
    chunks = _retrieve(llm, store, search)

    prompt = _build_prompt(
        role_name=role_name,
        role_description=role_description,
        project_industry=project_industry,
        available_skills=available_skills,
        chunks=chunks,
    )
    # Let LLMUnavailableError propagate so route returns 503
    payload = _read(llm.generate(prompt), role, search)
    if payload is None:
        return SkillSuggestionResult(suggestions=[])

    evidence = _Evidence(
        catalog={s.name.strip().lower(): s for s in available_skills},
        retrieved_chunk_ids={c.id for c in chunks},
        role_name=role_name,
    )
    return SkillSuggestionResult(suggestions=_ground(payload, evidence))
