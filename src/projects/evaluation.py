"""AI-supported evaluation of project industry/domain from ingested artifacts.

Retrieves project chunks via hybrid search and prompts the LLM to identify the
industry/domain as free text (e.g. "Fintech / Banking", "Quantum Computing Platform")
along with a confidence rating ("high", "medium", "low") and grounding evidence.

Fails closed by project scope: chunks from another project are never retrieved.
Degrades gracefully to an empty response with low confidence on parse errors or
when no eligible evidence is found.
"""

import json
import logging
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from api.schemas import IndustryEvaluationResponse
from ingestion.source_role import GROUNDING_EXCLUDED_ROLES
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from rag.hybrid import BM25IndexCache, hybrid_retrieve
from rag.types import RetrievalFilters, ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)

_TOP_K = 12
_MIN_SCORE = 0.3
_QUERY = "project industry domain overview purpose tech stack software architecture"


class _Payload(BaseModel):
    industry: str = ""
    confidence: Literal["high", "medium", "low"] = "low"
    evidence: list[str] = Field(default_factory=list)


def _fallback_response() -> IndustryEvaluationResponse:
    return IndustryEvaluationResponse(industry="", confidence="low", evidence=[])


def _build_prompt(chunks: list[ScoredChunk]) -> list[Message]:
    evidence = "\n".join(f"[{c.id}] ({c.filename}) {c.text}" for c in chunks)
    system = (
        "You analyze software project artifacts to determine the project's "
        "industry and domain.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences). Task:\n"
        "Based on the provided evidence artifacts, determine which industry/domain "
        "best describes this project (e.g. 'Fintech / Banking', 'Quantum Computing "
        "Platform', 'E-Commerce', 'Developer Tooling', 'Healthcare / MedTech').\n"
        "Assess your confidence as 'high', 'medium', or 'low'. Provide key evidence "
        "phrases or filenames grounding your assessment.\n\n"
        'JSON schema: {"industry": str, "confidence": "high"|"medium"|"low", '
        '"evidence": [str]}'
    )
    user = f"Artifact evidence:\n{evidence}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def evaluate_industry(
    llm: LLMClient,
    store: VectorStore,
    project_id: str,
    *,
    bm25_cache: BM25IndexCache | None = None,
) -> IndustryEvaluationResponse:
    """Evaluate the industry/domain of a project from its ingested corpus.

    Retrieves grounding evidence scoped to ``project_id`` and prompts the LLM for
    an industry assessment. Degrades gracefully on LLM parse errors or empty
    evidence; lets ``LLMUnavailableError`` propagate for the route to map to 503.
    """
    if store.count() == 0:
        return _fallback_response()

    cache = bm25_cache or BM25IndexCache()
    chunks = hybrid_retrieve(
        question=_QUERY,
        llm=llm,
        store=store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        bm25_cache=cache,
        exclude_roles=GROUNDING_EXCLUDED_ROLES,
        filters=RetrievalFilters(project_id=project_id),
    )

    if not chunks:
        return _fallback_response()

    raw = llm.generate(_build_prompt(chunks))
    try:
        payload = _Payload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Industry evaluation for project %s failed to parse LLM output, "
            "falling back to empty: %s",
            project_id,
            exc,
        )
        return _fallback_response()

    return IndustryEvaluationResponse(
        industry=payload.industry.strip(),
        confidence=payload.confidence,
        evidence=payload.evidence,
    )
