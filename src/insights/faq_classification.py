"""Incremental FAQ maintenance: classify one question, then re-consolidate.

The full rebuild in :mod:`insights.faq` costs one LLM pass over *every*
question a project ever asked. That is fine as a manual fallback and impossible
as a per-message operation, which is what issues #284/#285 need: the FAQ should
be current the moment someone asks the AI Buddy something, without a PM
pressing refresh.

This module is the cheap online counterpart. Two operations, each with a prompt
whose size is bounded by the *structure* of the FAQ rather than by its history:

``classify_question``
    One question against the existing entries. Decides: relevant at all? does
    it join one of them or open a new one? Cost is one LLM call regardless of
    how many questions came before.

``merge_groups``
    Runs when the entry count exceeds its ceiling. Sees only the entries'
    titles and representative questions, never the question history, and
    returns a merge plan.

The design is deliberately self-healing rather than exact. A single question
carries little context, so the classifier will occasionally open an entry that
duplicates an existing one. Rather than paying for a precise (and expensive)
comparison against the full corpus on every message, the merge pass folds those
duplicates back together once the limit is crossed. Drift is bounded by the
limit; it is never allowed to accumulate.
"""

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from ingestion.metadata_store import IngestionMetadataStore
from insights.faq import TITLE_RULE, FaqDocument, documents_for
from insights.redaction import redact_pii
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from store.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExistingGroup:
    """An entry the incoming question could join, as the classifier sees it.

    Both the title and a representative question travel: the title is what the
    model is asked to match the *topic* against, while the verbatim question is
    what makes "start the frontend" and "start the backend" distinguishable —
    a summarised title can lose exactly that detail.
    """

    id: str
    question: str
    title: str = ""
    count: int = 1


@dataclass(frozen=True)
class Classification:
    """What to do with one incoming question.

    ``relevant`` false means the text was smalltalk or otherwise not a
    documentation question; the caller drops it and nothing else in this
    object is meaningful.

    ``group_id`` set means "add to that existing entry"; ``None`` means "open a
    new one", in which case ``title`` names it and ``documents`` holds the
    answering documents retrieved for it (an existing entry already has both).
    """

    relevant: bool
    question: str = ""
    title: str = ""
    group_id: str | None = None
    documents: list[FaqDocument] = field(default_factory=list[FaqDocument])


@dataclass(frozen=True)
class Merge:
    """Fold ``sources`` into ``into``.

    All are entry ids: the surviving one keeps the stored samples, title and
    documents, so it always has to be one that already exists.
    """

    into: str
    sources: list[str]


class _ClassifyPayload(BaseModel):
    relevant: bool = True
    group_id: str | None = None
    title: str = ""
    redacted_question: str = ""


class _MergeEntry(BaseModel):
    into: str = ""
    sources: list[str] = []


class _MergePayload(BaseModel):
    merges: list[_MergeEntry] = []


_CLASSIFY_SYSTEM = (
    "You maintain the FAQ of a documentation chatbot for a PM-facing "
    "dashboard. A user just asked one question. Decide where it belongs.\n\n"
    "You are given the FAQ's existing entries, each one a recurring question "
    "listed with its id, its title and the wording it is usually asked in.\n\n"
    "Rules:\n"
    "1. If the text is not a genuine, documentation-relevant question — a "
    "greeting, smalltalk, thanks, or chit-chat — set relevant to false and "
    "stop. Everything else is then ignored.\n"
    "2. Match the question to an existing entry ONLY if the same piece of "
    "documentation would answer both — i.e. it is the same request in "
    "different words. Questions naming different components, services, or "
    "products (e.g. 'how do I start the frontend' vs 'how do I start the "
    "backend') are NOT the same entry even when they share a sentence "
    "template. When in doubt, open a new entry: set group_id to null.\n"
    "3. If you matched an existing entry, copy its title into title "
    f"unchanged. If you opened a new one, give it {TITLE_RULE}.\n"
    "4. Set redacted_question to the question's text with every person's name "
    "replaced by [NAME], every email by [EMAIL] and every phone number by "
    "[PHONE]. Change nothing else: keep wording, punctuation and meaning "
    "identical. Do not redact product names, company names or technical "
    "terms.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{"relevant": bool, "group_id": str|null, "title": str, '
    '"redacted_question": str}'
)

_MERGE_GROUPS_SYSTEM_TEMPLATE = (
    "You clean up duplicate entries in a documentation FAQ. It holds more "
    "entries than it should (at most {target_max}), which usually means the "
    "same question was opened more than once in slightly different words.\n\n"
    "You are given each entry's id, its title, its representative question "
    "and how often it was asked.\n\n"
    "Rules:\n"
    "1. Merge two entries ONLY if the same piece of documentation would answer "
    "both — the same request in different words. Entries naming different "
    "components, services, or products stay separate even when the sentence "
    "template or the title looks similar.\n"
    "2. The 'into' id must be one of the given ids: prefer the one with the "
    "highest count, since its wording is the one users actually use.\n"
    "3. Every id in 'sources' must be one of the given ids, no id may appear "
    "in more than one merge, and an id used as 'into' must not also appear as "
    "a source.\n"
    "4. If nothing is genuinely duplicated, return an empty merges list. "
    "Leaving the FAQ over its limit is better than merging distinct "
    "questions into one.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{{"merges": [{{"into": str, "sources": [str, ...]}}, ...]}}'
)


def _describe(group: ExistingGroup) -> str:
    title = group.title or "(untitled)"
    return f"- [{group.id}] {title} (asked {group.count}x) — {group.question}"


def _build_classify_prompt(question: str, groups: list[ExistingGroup]) -> list[Message]:
    listing = (
        "\n".join(_describe(group) for group in groups)
        if groups
        else "(none yet — this is the first question)"
    )
    user = f"Existing entries:\n{listing}\n\nNew question:\n{question}"
    return [
        Message(role="system", content=_CLASSIFY_SYSTEM),
        Message(role="user", content=user),
    ]


def _fallback_classification(question: str, llm: LLMClient) -> Classification:
    """Keep the question as its own entry when the classifier output is unusable.

    Dropping it would silently lose a real question. The redaction still runs —
    an unclassifiable question is no reason to leak a name into a PM's
    dashboard — and ``redact_pii`` degrades to its regex pass on its own if the
    LLM is the thing that is broken. The redacted text doubles as the title,
    which is wordy but still says what the entry is about.
    """
    redacted = redact_pii([question], llm)
    text = redacted[0] if redacted else question
    return Classification(relevant=True, question=text, title=text, group_id=None)


def classify_question(
    question: str,
    groups: list[ExistingGroup],
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    project_id: str,
) -> Classification:
    """Place one freshly asked question into the project's FAQ.

    Retrieval only runs when a new entry is opened: an existing one already
    carries the documents that answer it, so re-retrieving them on every repeat
    ask would be the per-message cost this whole path is built to avoid.

    ``LLMUnavailableError`` propagates so the endpoint can answer 503 — the
    caller's retry is the right response to a temporarily missing AI service,
    whereas a silently mis-filed question would be permanent.
    """
    text = question.strip()
    if not text:
        return Classification(relevant=False)

    raw = llm.generate(_build_classify_prompt(text, groups))
    try:
        payload = _ClassifyPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "FAQ classification failed to parse LLM output, keeping the "
            "question as its own entry: %s",
            exc,
        )
        return _fallback_classification(text, llm)

    if not payload.relevant:
        return Classification(relevant=False)

    groups_by_id = {g.id: g for g in groups}
    matched = groups_by_id.get(payload.group_id) if payload.group_id else None
    if payload.group_id and matched is None:
        # A hallucinated id must not silently become a new entry carrying the
        # title of one the model thought it was joining.
        logger.warning(
            "FAQ classification returned unknown entry id %r, opening a new "
            "entry instead",
            payload.group_id,
        )

    # A blank redaction would replace the question with nothing at all, which
    # reads as a corrupted FAQ entry rather than a redacted one.
    redacted = payload.redacted_question.strip() or text
    # A matched entry keeps its own title: it is established, and re-titling it
    # every time someone rephrases its question would make the list churn.
    title = matched.title if matched else (" ".join(payload.title.split()) or redacted)

    return Classification(
        relevant=True,
        question=redacted,
        title=title,
        group_id=matched.id if matched else None,
        documents=(
            []
            if matched
            else documents_for(text, llm, store, metadata_store, project_id)
        ),
    )


def merge_groups(
    groups: list[ExistingGroup],
    target_max: int,
    llm: LLMClient,
) -> list[Merge]:
    """Propose merges of duplicate entries once the FAQ is over its ceiling."""
    if len(groups) <= target_max:
        return []

    listing = "\n".join(_describe(group) for group in groups)
    messages: list[Message] = [
        Message(
            role="system",
            content=_MERGE_GROUPS_SYSTEM_TEMPLATE.format(target_max=target_max),
        ),
        Message(role="user", content=f"Entries ({len(groups)}):\n{listing}"),
    ]

    raw = llm.generate(messages)
    try:
        payload = _MergePayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "FAQ merging failed to parse LLM output, leaving the entries unchanged: %s",
            exc,
        )
        return []

    return _validated_merges(payload, known={g.id for g in groups})


def _validated_merges(payload: _MergePayload, known: set[str]) -> list[Merge]:
    """Drop merges the caller could not apply safely.

    A merge plan is applied destructively, so anything ambiguous is discarded
    rather than guessed: unknown ids, a source claimed twice, or a target that
    is itself being merged away would each corrupt the result.
    """
    merges: list[Merge] = []
    claimed: set[str] = set()

    for entry in payload.merges:
        target = entry.into.strip()
        if not target or target not in known or target in claimed:
            continue

        sources = [
            source
            for source in dict.fromkeys(s.strip() for s in entry.sources)
            if source in known and source != target and source not in claimed
        ]
        if not sources:
            continue

        claimed.update(sources)
        merges.append(Merge(into=target, sources=sources))

    # A target a *later* merge consumed as a source escapes the check above, and
    # applying both would make the result depend on the order they run in.
    return [m for m in merges if m.into not in claimed]
