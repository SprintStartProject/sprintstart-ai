"""Semantic grouping of recurring user questions into FAQ clusters.

Called by the backend's ``POST /insights/faq/refresh`` (pull-based, per
issue #66). ``/chat`` is stateless and this service does not retain question
history itself, so the backend sends the full set of questions to group on
every request.

A *group* is one recurring question — the same thing asked in different words.
Each carries a short generated **title** naming what is being asked (issue
#285). The title is what keeps the view readable as the set grows: a list of
verbatim questions makes a PM read a whole sentence to work out what each entry
is about, while "Getting VPN access" is scannable, and one title stays stable
while the phrasings under it vary.

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
class FaqSampleQuestion:
    """One redacted sample question, carrying the id it came in as.

    The id lets the caller tie the sample back to the message that was asked,
    and with it to when it was asked — which is what the FAQ's trend and
    recency signals are built from.
    """

    id: str
    text: str


@dataclass(frozen=True)
class FaqGroup:
    question: str
    count: int
    questions: list[FaqSampleQuestion]
    documents: list[FaqDocument]
    title: str = ""
    question_ids: list[str] = field(default_factory=list[str])


@dataclass
class _Cluster:
    members: list[FaqQuestionInput] = field(default_factory=list[FaqQuestionInput])
    title: str = ""


class _GroupEntry(BaseModel):
    ids: list[str] = []
    title: str = ""


class _GroupPayload(BaseModel):
    groups: list[_GroupEntry] = []
    discard_ids: list[str] = []


TITLE_RULE = (
    "a short title naming what is being asked: 3-8 words, sentence case, no "
    "trailing punctuation, e.g. 'Getting VPN access', 'Starting the backend "
    "locally', 'Submitting a travel expense report'. It must summarise the "
    "request rather than copy a question verbatim, keep whatever component or "
    "product the questions name, and be specific enough to tell this entry "
    "apart from a neighbouring one — 'Setup' or 'Access' alone is too generic"
)

_GROUPING_SYSTEM = (
    "You group recurring end-user questions asked to a docs chatbot into FAQ "
    "entries for a PM-facing dashboard. Each input question is prefixed with "
    "its id in square brackets.\n\n"
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
    f"5. Give every group {TITLE_RULE}.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{"groups": [{"ids": [id, ...], "title": str}, ...], '
    '"discard_ids": [id, ...]}'
)


def _build_grouping_prompt(questions: list[FaqQuestionInput]) -> list[Message]:
    listing = "\n".join(f"[{q.id}] {q.text}" for q in questions)
    return [
        Message(role="system", content=_GROUPING_SYSTEM),
        Message(role="user", content=listing),
    ]


def _cluster_questions(
    questions: list[FaqQuestionInput], llm: LLMClient
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
    raw = llm.generate(_build_grouping_prompt(list(by_id.values())))
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
                _Cluster(members=members, title=" ".join(entry.title.split()))
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
) -> list[FaqGroup]:
    """Group questions, title the groups, and attach answering documents.

    The documents come from retrieval, so they are scoped to ``project_id`` —
    a group must never point a PM at a document from another project.
    """
    clusters = _cluster_questions(questions, llm)

    # Redact every representative + sample question in a single batched LLM
    # call rather than one call per group.
    sample_texts: list[str] = []
    sample_ids: list[str] = []
    sample_bounds: list[tuple[int, int]] = []
    for cluster in clusters:
        seen: dict[str, str] = {}
        for member in cluster.members:
            if len(seen) >= _MAX_SAMPLE_QUESTIONS:
                break
            # Keyed by text so a question asked twice verbatim contributes one
            # sample; the first asker's id is the one kept.
            seen.setdefault(member.text, member.id)
        start = len(sample_texts)
        sample_texts.extend(seen.keys())
        sample_ids.extend(seen.values())
        sample_bounds.append((start, start + len(seen)))

    redacted = redact_pii(sample_texts, llm)

    groups: list[FaqGroup] = []
    for cluster, (start, end) in zip(clusters, sample_bounds, strict=True):
        samples = [
            FaqSampleQuestion(id=qid, text=text)
            for qid, text in zip(
                sample_ids[start:end], redacted[start:end], strict=True
            )
        ]
        representative_text = cluster.members[0].text
        representative_redacted = samples[0].text if samples else representative_text
        groups.append(
            FaqGroup(
                question=representative_redacted,
                count=len(cluster.members),
                questions=samples,
                documents=documents_for(
                    representative_text, llm, store, metadata_store, project_id
                ),
                # Falls back to the representative question rather than to a
                # placeholder: a wordy entry still says what it is about, an
                # "Untitled" one says nothing at all.
                title=cluster.title or representative_redacted,
                question_ids=[member.id for member in cluster.members],
            )
        )

    groups.sort(key=lambda g: g.count, reverse=True)
    return groups
