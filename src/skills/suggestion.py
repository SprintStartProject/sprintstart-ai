"""AI-supported skill suggestion for project roles."""

import logging
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from api.schemas import (
    SkillCatalogItem,
    SkillSuggestionItem,
    SkillSuggestionResponse,
)
from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from rag.filters import RetrievalFilters
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 15
_MIN_SCORE = 0.3


class _Suggestion(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(min_length=1)
    category: str | None = None
    reason: str = Field(default="")
    confidence: Literal["high", "medium", "low"] = "medium"
    is_new: bool = Field(alias="isNew", default=False)
    chunk_ids: list[str] = Field(alias="chunkIds", default_factory=list[str])


class _Payload(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    suggestions: list[_Suggestion] = Field(default_factory=list[_Suggestion])


def _format_catalog(available_skills: list[SkillCatalogItem]) -> str:
    if not available_skills:
        return "(empty catalog)"
    lines: list[str] = []
    for s in available_skills:
        cat = f" [{s.category}]" if s.category else ""
        univ = " (universal)" if s.universal else ""
        lines.append(f"- [{s.id}] {s.name}{cat}{univ}")
    return "\n".join(lines)


def _format_chunks(chunks: list[ScoredChunk]) -> str:
    if not chunks:
        return "(no project artifacts retrieved)"
    return "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)


def _build_prompt(
    role_name: str,
    role_description: str,
    project_industry: str | None,
    available_skills: list[SkillCatalogItem],
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
        "1. For UNIVERSAL skills (marked '(universal)' in catalog/agile/soft):\n"
        "   - Recommend role-typical universal skills WITHOUT requiring evidence "
        "(chunkIds can be empty).\n"
        "   - Standard practices (e.g. Code Review, Agile/Scrum, Communication).\n"
        "2. For PROJECT-SPECIFIC skills (technologies, frameworks, databases, tools):\n"
        "   - MUST cite at least one chunk_id from the evidence in chunkIds.\n"
        "   - Only suggest if project artifacts actually demonstrate their usage.\n"
        "   - If missing in catalog, suggest with isNew=true (and chunkIds).\n\n"
        "STRICT JSON only (no markdown fences, no explanatory text outside JSON):\n"
        '{"suggestions": [{"name": str, "category": str | null, "reason": str, '
        '"confidence": "high"|"medium"|"low", "isNew": bool, "chunkIds": [str]}]}'
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


def suggest_skills(
    llm: LLMClient,
    store: VectorStore,
    *,
    project_id: str,
    role_name: str,
    role_description: str = "",
    project_industry: str | None = None,
    available_skills: list[SkillCatalogItem] | None = None,
    bm25_cache: BM25IndexCache | None = None,
) -> SkillSuggestionResponse:
    """Suggest fitting skills for a project role, grounded in project artifacts."""
    if available_skills is None:
        available_skills = []

    # Map catalog by normalized name for lookup
    catalog_by_norm_name: dict[str, SkillCatalogItem] = {
        s.name.strip().lower(): s for s in available_skills
    }

    # Retrieve relevant project artifacts via hybrid RAG
    query_parts = [role_name]
    if role_description:
        query_parts.append(role_description)
    if project_industry:
        query_parts.append(project_industry)
    query = " ".join(query_parts)

    if bm25_cache is None:
        bm25_cache = BM25IndexCache()

    chunks: list[ScoredChunk] = []
    if store.count() > 0:
        chunks = hybrid_retrieve(
            question=query,
            llm=llm,
            store=store,
            top_k=_TOP_K,
            min_score=_MIN_SCORE,
            bm25_cache=bm25_cache,
            exclude_roles=GROUNDING_EXCLUDED_ROLES,
            filters=RetrievalFilters(project_id=project_id),
        )

    retrieved_chunk_ids = {c.id for c in chunks}

    prompt = _build_prompt(
        role_name=role_name,
        role_description=role_description,
        project_industry=project_industry,
        available_skills=available_skills,
        chunks=chunks,
    )

    # Let LLMUnavailableError propagate so route returns 503
    raw = llm.generate(prompt)

    try:
        json_str = extract_json_object(raw)
        payload = _Payload.model_validate_json(json_str)
    except (ValueError, ValidationError) as exc:
        logger.warning(
            "Failed to parse LLM skill suggestions for role %r on project %r: %s",
            role_name,
            project_id,
            exc,
        )
        return SkillSuggestionResponse(suggestions=[])

    # Grounding gate:
    # Universal skills (from catalog with universal=True) pass without evidence.
    # Project-specific skills (universal=False/new) MUST have >=1 valid chunk_id.
    validated_suggestions: list[SkillSuggestionItem] = []
    seen_names: set[str] = set()

    for item in payload.suggestions:
        norm_name = item.name.strip().lower()
        if not norm_name or norm_name in seen_names:
            continue

        catalog_entry = catalog_by_norm_name.get(norm_name)
        is_universal = catalog_entry.universal if catalog_entry else False

        # Filter chunk_ids to only those actually retrieved
        valid_chunk_ids = [cid for cid in item.chunk_ids if cid in retrieved_chunk_ids]

        if is_universal:
            # Universal skill: no chunk evidence required
            seen_names.add(norm_name)
            name = catalog_entry.name if catalog_entry else item.name.strip()
            category = item.category or (
                catalog_entry.category if catalog_entry else None
            )
            reason = item.reason or f"Recommended universal skill for {role_name}."
            validated_suggestions.append(
                SkillSuggestionItem(
                    name=name,
                    category=category,
                    reason=reason,
                    confidence=item.confidence,
                    is_new=False,
                    chunk_ids=valid_chunk_ids,
                )
            )
        else:
            # Project-specific skill: must have chunk evidence
            if not valid_chunk_ids:
                logger.debug(
                    "Dropping ungrounded skill suggestion %r for role %r",
                    item.name,
                    role_name,
                )
                continue

            seen_names.add(norm_name)
            is_new = item.is_new or (catalog_entry is None)
            name = (
                catalog_entry.name
                if (catalog_entry and not item.is_new)
                else item.name.strip()
            )
            category = item.category or (
                catalog_entry.category if catalog_entry else None
            )
            reason = (
                item.reason
                or f"Project-specific skill identified from artifacts for {role_name}."
            )

            validated_suggestions.append(
                SkillSuggestionItem(
                    name=name,
                    category=category,
                    reason=reason,
                    confidence=item.confidence,
                    is_new=is_new,
                    chunk_ids=valid_chunk_ids,
                )
            )

    return SkillSuggestionResponse(suggestions=validated_suggestions)
