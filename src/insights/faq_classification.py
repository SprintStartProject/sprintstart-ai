"""Incremental FAQ maintenance: classify one question, then re-consolidate.

The full rebuild in :mod:`insights.faq` costs one LLM pass over *every*
question a project ever asked. That is fine as a manual fallback and impossible
as a per-message operation, which is what issues #284/#285 ask for: the FAQ and
its categories should be current the moment someone asks the AI Buddy something,
without a PM pressing refresh.

This module is the cheap online counterpart. Three operations, each with a
prompt whose size is bounded by the *structure* of the FAQ rather than by its
history:

``classify_question``
    One question against the existing categories and group representatives.
    Decides: relevant at all? which category? an existing group or a new one?
    Cost is one LLM call regardless of how many questions came before.

``consolidate_categories``
    Runs when the category count exceeds its ceiling. Sees only category names
    and counts — no question text at all — and returns a merge plan.

``merge_groups``
    Runs when one category holds too many groups. Sees only that category's
    group representatives and returns a merge plan.

The design is deliberately self-healing rather than exact. A single question
carries little context, so the classifier will occasionally open a group that
duplicates an existing one. Rather than paying for a precise (and expensive)
comparison against the full corpus on every message, the consolidation passes
fold those duplicates back together once a limit is crossed. Drift is bounded
by the limits; it is never allowed to accumulate.
"""

import json
import logging
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from ingestion.metadata_store import IngestionMetadataStore
from insights.faq import UNCATEGORIZED, FaqDocument, documents_for
from insights.redaction import redact_pii
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from store.base import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExistingCategory:
    """A category the project already uses, with its current weight."""

    name: str
    group_count: int = 0
    question_count: int = 0


@dataclass(frozen=True)
class ExistingGroup:
    """A group the incoming question could join, as the classifier sees it."""

    id: str
    question: str
    category: str = UNCATEGORIZED
    count: int = 1


@dataclass(frozen=True)
class Classification:
    """What to do with one incoming question.

    ``relevant`` false means the text was smalltalk or otherwise not a
    documentation question; the caller drops it and nothing else in this
    object is meaningful.

    ``group_id`` set means "add to that existing group"; ``None`` means "open a
    new group", in which case ``documents`` holds the answering documents that
    were retrieved for it (an existing group already has its own).
    """

    relevant: bool
    question: str = ""
    category: str = UNCATEGORIZED
    group_id: str | None = None
    documents: list[FaqDocument] = field(default_factory=list[FaqDocument])


@dataclass(frozen=True)
class Merge:
    """Fold ``sources`` into ``into``.

    For categories these are names, for groups they are ids. ``into`` may be a
    name that did not exist before (an umbrella label the model invented);
    for groups it is always one of the existing ids, since a group carries
    stored samples and documents that have to survive the merge.
    """

    into: str
    sources: list[str]


class _ClassifyPayload(BaseModel):
    relevant: bool = True
    group_id: str | None = None
    category: str = ""
    redacted_question: str = ""


class _MergeEntry(BaseModel):
    into: str = ""
    sources: list[str] = []


class _MergePayload(BaseModel):
    merges: list[_MergeEntry] = []


_CLASSIFY_SYSTEM = (
    "You maintain the FAQ of a documentation chatbot for a PM-facing "
    "dashboard. A user just asked one question. Decide where it belongs.\n\n"
    "You are given the FAQ's existing categories (topic buckets) and existing "
    "groups (one recurring question each, listed with their id).\n\n"
    "Rules:\n"
    "1. If the text is not a genuine, documentation-relevant question — a "
    "greeting, smalltalk, thanks, or chit-chat — set relevant to false and "
    "stop. Everything else is then ignored.\n"
    "2. Match the question to an existing group ONLY if the same piece of "
    "documentation would answer both — i.e. it is the same request in "
    "different words. Questions naming different components, services, or "
    "products (e.g. 'how do I start the frontend' vs 'how do I start the "
    "backend') are NOT the same group even when they share a sentence "
    "template. When in doubt, open a new group: set group_id to null.\n"
    "3. Set category to the topic bucket the question belongs to. Strongly "
    "prefer an existing category and copy its spelling EXACTLY. Only invent a "
    "new one when no existing category plausibly covers the question — a new "
    "category must be a short topic label (2-4 words, Title Case) that names "
    "an area, not this one question.\n"
    "4. If you matched an existing group, category must be that group's "
    "category.\n"
    "5. Set redacted_question to the question's text with every person's name "
    "replaced by [NAME], every email by [EMAIL] and every phone number by "
    "[PHONE]. Change nothing else: keep wording, punctuation and meaning "
    "identical. Do not redact product names, company names or technical "
    "terms.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{"relevant": bool, "group_id": str|null, "category": str, '
    '"redacted_question": str}'
)

_CONSOLIDATE_SYSTEM_TEMPLATE = (
    "You reorganize the topic categories of a documentation FAQ. The category "
    "set has grown past what a PM can scan, so it must be reduced to at most "
    "{target_max} categories by merging related ones.\n\n"
    "You are given each category with how many question groups and how many "
    "asked questions it holds.\n\n"
    "Rules:\n"
    "1. Merge categories that describe the same area or that are narrow "
    "facets of a broader one. Never merge unrelated areas just to hit the "
    "number.\n"
    "2. Prefer merging small, rarely-asked categories into larger related "
    "ones. A category holding many questions is a real topic — keep it.\n"
    "3. The 'into' name is what survives. Reuse an existing category's exact "
    "name when the merge is clearly absorption into it; invent a broader "
    "label (2-4 words, Title Case) only when merging peers with no natural "
    "parent among them.\n"
    "4. Every name in 'sources' must be one of the given categories, and no "
    "category may appear in more than one merge.\n"
    "5. Return only the merges you actually want. Categories you do not "
    "mention stay as they are.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{{"merges": [{{"into": str, "sources": [str, ...]}}, ...]}}'
)

_MERGE_GROUPS_SYSTEM_TEMPLATE = (
    "You clean up duplicate entries in one topic category of a documentation "
    "FAQ. The category holds more question groups than it should ("
    "at most {target_max}), which usually means the same question was opened "
    "more than once in slightly different words.\n\n"
    "You are given each group's id, its representative question and how often "
    "it was asked.\n\n"
    "Rules:\n"
    "1. Merge two groups ONLY if the same piece of documentation would answer "
    "both — the same request in different words. Groups naming different "
    "components, services, or products stay separate even when the sentence "
    "template matches.\n"
    "2. The 'into' id must be one of the given group ids: prefer the one with "
    "the highest count, since its wording is the one users actually use.\n"
    "3. Every id in 'sources' must be one of the given ids, no id may appear "
    "in more than one merge, and an id used as 'into' must not also appear as "
    "a source.\n"
    "4. If nothing is genuinely duplicated, return an empty merges list. "
    "Leaving the category over its limit is better than merging distinct "
    "questions into one.\n\n"
    "Return STRICT JSON only (no prose, no markdown fences): "
    '{{"merges": [{{"into": str, "sources": [str, ...]}}, ...]}}'
)


def _build_classify_prompt(
    question: str,
    categories: list[ExistingCategory],
    groups: list[ExistingGroup],
) -> list[Message]:
    if categories:
        category_lines = "\n".join(
            f"- {c.name} ({c.group_count} groups, {c.question_count} questions)"
            for c in categories
        )
    else:
        category_lines = "(none yet — this is the first question)"

    if groups:
        group_lines = "\n".join(
            f"- [{g.id}] ({g.category}, asked {g.count}x) {g.question}" for g in groups
        )
    else:
        group_lines = "(none yet)"

    user = (
        f"Existing categories:\n{category_lines}\n\n"
        f"Existing groups:\n{group_lines}\n\n"
        f"New question:\n{question}"
    )
    return [
        Message(role="system", content=_CLASSIFY_SYSTEM),
        Message(role="user", content=user),
    ]


def _fallback_classification(question: str, llm: LLMClient) -> Classification:
    """Keep the question, uncategorized, when the classifier output is unusable.

    Dropping it would silently lose a real question, and guessing a category
    would pollute the very structure this module exists to keep clean. The
    redaction still runs — an unclassified question is no reason to leak a name
    into a PM's dashboard. ``redact_pii`` degrades to its regex pass on its own
    if the LLM is the thing that is broken.
    """
    redacted = redact_pii([question], llm)
    return Classification(
        relevant=True,
        question=redacted[0] if redacted else question,
        category=UNCATEGORIZED,
        group_id=None,
    )


def classify_question(
    question: str,
    categories: list[ExistingCategory],
    groups: list[ExistingGroup],
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    project_id: str,
) -> Classification:
    """Place one freshly asked question into the project's FAQ structure.

    Retrieval only runs when a new group is opened: an existing group already
    carries the documents that answer it, so re-retrieving them on every repeat
    ask would be the per-message cost this whole path is built to avoid.

    ``LLMUnavailableError`` propagates so the endpoint can answer 503 — the
    caller's retry is the right response to a temporarily missing AI service,
    whereas a silently uncategorized question would be permanent.
    """
    text = question.strip()
    if not text:
        return Classification(relevant=False)

    raw = llm.generate(_build_classify_prompt(text, categories, groups))
    try:
        payload = _ClassifyPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "FAQ classification failed to parse LLM output, keeping the "
            "question uncategorized: %s",
            exc,
        )
        return _fallback_classification(text, llm)

    if not payload.relevant:
        return Classification(relevant=False)

    groups_by_id = {g.id: g for g in groups}
    matched = groups_by_id.get(payload.group_id) if payload.group_id else None
    if payload.group_id and matched is None:
        # A hallucinated id must not silently become a new group under a
        # category the model picked for a group it thought it was joining.
        logger.warning(
            "FAQ classification returned unknown group id %r, opening a new "
            "group instead",
            payload.group_id,
        )

    category = _resolve_category(payload.category, matched, categories)
    # A blank redaction would replace the question with nothing at all, which
    # reads as a corrupted FAQ entry rather than a redacted one.
    redacted = payload.redacted_question.strip() or text

    return Classification(
        relevant=True,
        question=redacted,
        category=category,
        group_id=matched.id if matched else None,
        documents=(
            []
            if matched
            else documents_for(text, llm, store, metadata_store, project_id)
        ),
    )


def _resolve_category(
    raw: str,
    matched: ExistingGroup | None,
    categories: list[ExistingCategory],
) -> str:
    """Settle on the category for a classified question.

    A matched group's own category wins over whatever the model wrote: a group
    lives in exactly one category, and honouring a conflicting answer here
    would move an established group every time someone rephrases its question.
    """
    if matched is not None:
        return matched.category

    label = " ".join(raw.split())
    if not label:
        return UNCATEGORIZED

    # Fold case/whitespace variants onto the existing spelling, otherwise
    # "local setup" would open a second bucket next to "Local Setup".
    for category in categories:
        if category.name.casefold() == label.casefold():
            return category.name
    return label


def _validated_merges(
    payload: _MergePayload,
    known: set[str],
    require_known_target: bool,
) -> list[Merge]:
    """Drop merges the caller could not apply safely.

    A merge plan is applied destructively, so anything ambiguous is discarded
    rather than guessed: unknown names, a source claimed twice, or a target
    that is itself being merged away would each corrupt the result.
    """
    merges: list[Merge] = []
    claimed: set[str] = set()

    for entry in payload.merges:
        target = " ".join(entry.into.split())
        if not target:
            continue
        if require_known_target and target not in known:
            logger.warning("Ignoring merge into unknown target %r", target)
            continue
        if target in claimed:
            logger.warning("Ignoring merge into %r, already merged away", target)
            continue

        sources = [
            source
            for source in dict.fromkeys(" ".join(s.split()) for s in entry.sources)
            if source in known and source != target and source not in claimed
        ]
        if not sources:
            continue

        claimed.update(sources)
        merges.append(Merge(into=target, sources=sources))

    # A target a *later* merge consumed as a source escapes the check above, and
    # applying both would make the result depend on the order they run in.
    return [m for m in merges if m.into not in claimed]


def consolidate_categories(
    categories: list[ExistingCategory],
    target_max: int,
    llm: LLMClient,
) -> list[Merge]:
    """Propose category merges that bring the set back under ``target_max``.

    Only category metadata is sent, never question text, which is what makes
    this affordable to run whenever the ceiling is crossed. The plan is a
    proposal: the caller applies it and stays the owner of the data.
    """
    if len(categories) <= target_max:
        return []

    listing = "\n".join(
        f"- {c.name} ({c.group_count} groups, {c.question_count} questions)"
        for c in categories
    )
    messages: list[Message] = [
        Message(
            role="system",
            content=_CONSOLIDATE_SYSTEM_TEMPLATE.format(target_max=target_max),
        ),
        Message(
            role="user",
            content=f"Categories ({len(categories)}):\n{listing}",
        ),
    ]

    raw = llm.generate(messages)
    try:
        payload = _MergePayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Category consolidation failed to parse LLM output, leaving the "
            "categories unchanged: %s",
            exc,
        )
        return []

    return _validated_merges(
        payload,
        known={c.name for c in categories},
        # An umbrella label the model invented is a legitimate merge target
        # here, unlike for groups where the target has to carry stored data.
        require_known_target=False,
    )


def merge_groups(
    groups: list[ExistingGroup],
    target_max: int,
    llm: LLMClient,
) -> list[Merge]:
    """Propose merges of duplicate groups inside one over-full category."""
    if len(groups) <= target_max:
        return []

    listing = "\n".join(f"- [{g.id}] (asked {g.count}x) {g.question}" for g in groups)
    messages: list[Message] = [
        Message(
            role="system",
            content=_MERGE_GROUPS_SYSTEM_TEMPLATE.format(target_max=target_max),
        ),
        Message(role="user", content=f"Groups ({len(groups)}):\n{listing}"),
    ]

    raw = llm.generate(messages)
    try:
        payload = _MergePayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        logger.warning(
            "Group merging failed to parse LLM output, leaving the groups "
            "unchanged: %s",
            exc,
        )
        return []

    return _validated_merges(
        payload,
        known={g.id for g in groups},
        # The surviving group keeps its samples and documents, so it has to be
        # a real one — an invented id would have nothing behind it.
        require_known_target=True,
    )
