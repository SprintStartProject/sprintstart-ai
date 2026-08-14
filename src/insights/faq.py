"""Semantic grouping of recurring user questions into FAQ clusters.

Called by the backend's ``POST /insights/faq/refresh`` (pull-based, per
issue #66). ``/chat`` is stateless and this service does not retain question
history itself, so the backend sends the full set of questions to group on
every request.

Two levels of structure come out of this (issue #285): a *group* is one
recurring question (the same thing asked in different words), a *category* is
the topic bucket a group belongs to. Categories are what keeps the view
readable as the question set grows — a flat list of 200 groups is exactly the
"drowning in questions" problem the grouping is meant to solve.

This full rebuild is the expensive path: its cost grows with the total number
of questions, so it is the manual-refresh fallback rather than something to run
per chat message. The incremental counterpart lives in
:mod:`insights.faq_classification`.
"""

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from ingestion.metadata_store import IngestionMetadataStore
from insights.redaction import redact_pii
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from rag.retriever import retrieve
from rag.types import RetrievalFilters
from store.base import VectorStore

logger = logging.getLogger(__name__)

# Cap on redacted sample questions returned per group.
_MAX_SAMPLE_QUESTIONS = 5
# Retrieval settings used to find the documents that answer a group.
_DOCS_TOP_K = 5
_DOCS_MIN_SCORE = 0.3
# Cap on distinct documents returned per group.
_MAX_DOCUMENTS = 5
# Default ceiling on distinct categories. The caller (backend) owns the real
# limit and passes it in; this is only the fallback for direct calls.
DEFAULT_MAX_CATEGORIES = 12
# Category assigned when the model gave us nothing usable. Never invented by
# the model itself — it is the honest "we don't know" bucket, and the backend
# can re-classify those groups later instead of guessing a topic now.
UNCATEGORIZED = "Uncategorized"


@dataclass(frozen=True)
class FaqQuestionInput:
    id: str
    text: str


@dataclass(frozen=True)
class FaqDocument:
    id: str
    title: str
    source: str | None


@dataclass(frozen=True)
class FaqGroup:
    question: str
    count: int
    questions: list[str]
    documents: list[FaqDocument]
    category: str = UNCATEGORIZED


@dataclass
class _Cluster:
    members: list[FaqQuestionInput] = field(default_factory=list[FaqQuestionInput])
    category: str = UNCATEGORIZED


class _GroupEntry(BaseModel):
    ids: list[str] = []
    category: str = ""


class _GroupPayload(BaseModel):
    groups: list[_GroupEntry] = []
    discard_ids: list[str] = []


_GROUPING_SYSTEM_TEMPLATE = (
    "You group recurring end-user questions asked to a docs chatbot into FAQ "
    "clusters for a PM-facing dashboard, and sort those clusters into topic "
    "categories. Each input question is prefixed with its id in square "
    "brackets.\n\n"
    "Rules:\n"
    "1. First set aside anything that is not a genuine, documentation-relevant "
    "question — greetings, smalltalk, or chit-chat (e.g. 'hey', 'hey there, "
    "how you doing', 'thanks!'). List their ids in discard_ids instead of a "
    "group.\n"
    "2. Group the remaining questions by what they are actually asking, not by "
    "surface sentence structure. Two questions belong together only if the "
    "same piece of documentation would answer both. Questions that name "
    "different components, services, or products (e.g. 'how to start the "
    "frontend' vs 'how to start the backend') are DIFFERENT groups even "
    "though they share the same template — the named component is the "
    "distinguishing part, not noise.\n"
    "3. Minor rewordings, abbreviations, or added politeness for the *same* "
    "request belong in the same group.\n"
    "4. Every input id must appear exactly once, in exactly one group or in "
    "discard_ids.\n"
    "5. Give every group a category: a short topic label (2-4 words, Title "
    "Case) naming the area the question belongs to, e.g. 'Local Setup', "
    "'Deployment & CI', 'Authentication', 'Testing'. Categories are a level "
    "ABOVE groups — several groups share one category. Reuse the exact same "
    "spelling for every group in the same category.\n"
    "6. Use at most {max_categories} distinct categories in total. If you "
    "would need more, merge the least distinct ones into a broader label. "
    "Prefer categories that stay meaningful as the question set grows: name "
    "the topic, not the individual question.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{{"groups": [{{"ids": [id, ...], "category": str}}, ...], '
    '"discard_ids": [id, ...]}}'
)


def _build_grouping_prompt(
    questions: list[FaqQuestionInput], max_categories: int
) -> list[Message]:
    listing = "\n".join(f"[{q.id}] {q.text}" for q in questions)
    return [
        Message(
            role="system",
            content=_GROUPING_SYSTEM_TEMPLATE.format(max_categories=max_categories),
        ),
        Message(role="user", content=listing),
    ]


def _normalize_category(raw: str, seen: dict[str, str]) -> str:
    """Map a model-supplied category onto a single canonical spelling.

    The model is asked to reuse spellings but does not always: "Local Setup"
    and "local setup" would otherwise become two categories that a PM reads as
    one. First spelling wins, later case/whitespace variants fold into it.
    """
    label = " ".join(raw.split())
    if not label:
        return UNCATEGORIZED
    key = label.casefold()
    if key not in seen:
        seen[key] = label
    return seen[key]


def _cluster_questions(
    questions: list[FaqQuestionInput], llm: LLMClient, max_categories: int
) -> list[_Cluster]:
    """Group questions into FAQ clusters with a single batched LLM call.

    A single call over the whole set replaces the previous greedy,
    embedding-threshold clustering: it judges cluster membership by meaning
    (e.g. treating a named component like "frontend" vs "backend" as
    distinguishing) rather than a fixed cosine cutoff, isn't sensitive to
    input order, and filters out non-questions (greetings/smalltalk) before
    they can be surfaced as a group.
    """
    by_id = {q.id: q for q in questions if q.text.strip()}
    if not by_id:
        return []

    order = {qid: i for i, qid in enumerate(by_id)}
    raw = llm.generate(_build_grouping_prompt(list(by_id.values()), max_categories))
    try:
        payload = _GroupPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "FAQ grouping failed to parse LLM output, falling back to "
            "ungrouped questions: %s",
            exc,
        )
        return [_Cluster(members=[q]) for q in by_id.values()]

    clusters: list[_Cluster] = []
    claimed: set[str] = set(payload.discard_ids)
    canonical_categories: dict[str, str] = {}
    for entry in payload.groups:
        unique_ids = list(
            dict.fromkeys(
                gid for gid in entry.ids if gid in by_id and gid not in claimed
            )
        )
        claimed.update(unique_ids)
        if unique_ids:
            members = sorted(
                (by_id[gid] for gid in unique_ids), key=lambda q: order[q.id]
            )
            clusters.append(
                _Cluster(
                    members=members,
                    category=_normalize_category(entry.category, canonical_categories),
                )
            )

    # Defensive: never silently drop a question the model didn't classify.
    for qid, question in by_id.items():
        if qid not in claimed:
            clusters.append(_Cluster(members=[question]))

    return clusters


def documents_for(
    representative_text: str,
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    project_id: str,
) -> list[FaqDocument]:
    """Retrieve the project's documents that answer ``representative_text``.

    Shared with the incremental classification path, which needs the same
    document lookup when it opens a new group.
    """
    chunks = retrieve(
        question=representative_text,
        llm=llm,
        store=store,
        top_k=_DOCS_TOP_K,
        min_score=_DOCS_MIN_SCORE,
        filters=RetrievalFilters(project_id=project_id),
    )

    documents: dict[str, FaqDocument] = {}
    for chunk in chunks:
        if chunk.artifact_id in documents:
            continue
        record = metadata_store.get_artifact(chunk.artifact_id)
        documents[chunk.artifact_id] = FaqDocument(
            id=chunk.artifact_id,
            title=record.filename if record is not None else chunk.filename,
            source=record.source_type if record is not None else chunk.artifact_type,
        )
        if len(documents) >= _MAX_DOCUMENTS:
            break

    return list(documents.values())


def group_faqs(
    questions: list[FaqQuestionInput],
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    project_id: str,
    max_categories: int = DEFAULT_MAX_CATEGORIES,
) -> list[FaqGroup]:
    """Group questions, categorize the groups, and attach answering documents.

    The documents come from retrieval, so they are scoped to ``project_id`` —
    a group must never point a PM at a document from another project.
    """
    clusters = _cluster_questions(questions, llm, max_categories)

    # Redact every representative + sample question in a single batched LLM
    # call rather than one call per group.
    sample_texts: list[str] = []
    sample_bounds: list[tuple[int, int]] = []
    for cluster in clusters:
        seen: list[str] = []
        for member in cluster.members:
            if len(seen) >= _MAX_SAMPLE_QUESTIONS:
                break
            if member.text not in seen:
                seen.append(member.text)
        start = len(sample_texts)
        sample_texts.extend(seen)
        sample_bounds.append((start, start + len(seen)))

    redacted = redact_pii(sample_texts, llm)

    groups: list[FaqGroup] = []
    for cluster, (start, end) in zip(clusters, sample_bounds, strict=True):
        redacted_samples = redacted[start:end]
        representative_text = cluster.members[0].text
        representative_redacted = (
            redacted_samples[0] if redacted_samples else representative_text
        )
        groups.append(
            FaqGroup(
                question=representative_redacted,
                count=len(cluster.members),
                questions=redacted_samples,
                documents=documents_for(
                    representative_text, llm, store, metadata_store, project_id
                ),
                category=cluster.category,
            )
        )

    groups.sort(key=lambda g: g.count, reverse=True)
    return groups
