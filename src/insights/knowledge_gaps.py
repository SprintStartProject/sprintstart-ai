"""Structural documentation-coverage gap detection per component.

Called by the backend's Knowledge-Gaps insight refresh (pull-based, issue #137).
This service is stateless and sources everything from its ingestion index, so
the request carries nothing but the project the scan is scoped to.

"Insufficient" is scoped here as *structural coverage*: for each component of
that project we determine which expected documentation categories (readme,
setup, adr, …) are present versus missing. Detection is hybrid — the LLM
classifies a component's documents into categories, with a filename heuristic as
a fallback when the LLM output can't be used.

Every component of the project is reported, including the ones missing nothing;
those carry the "covered" severity. The result is therefore a coverage roster
rather than a list of problems: a component in good shape is a finding of its
own, and leaving it out made it look like it had never been ingested.

Owners and related-question counts are deliberately NOT produced here: the
ingestion index holds no user/ownership data and this service retains no
question history. The backend enriches the returned ``component`` with those.
"""

import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from llm.base import LLMClient, Message
from llm.parsing import extract_json_object
from store.base import VectorStore

logger = logging.getLogger(__name__)

Severity = Literal["high", "medium", "low", "covered"]

# Documentation categories every component should ideally have, ordered from
# most to least foundational. This is the "expected-type checklist" the corpus
# is measured against; ``missingTypes = expected - present``.
EXPECTED_TYPES: tuple[str, ...] = (
    "readme",
    "setup",
    "architecture",
    "adr",
    "api",
    "runbook",
)

# Categories whose absence is especially damaging for onboarding/operations and
# therefore weighs heavier in severity scoring.
CRITICAL_TYPES: frozenset[str] = frozenset({"readme", "setup"})

# A component whose newest artifact is older than this is considered stale, which
# bumps its gap severity by one notch.
_STALE_AFTER_DAYS = 180

# Bound on how many documents (and how much text per document) we feed the
# classifier, to keep the per-component prompt within a reasonable token budget.
_MAX_DOCS_PER_COMPONENT = 60
_SNIPPET_CHARS = 600

# Filename extensions that are documentation-like, used to prioritize the
# sample fed to the classifier (see ``_doc_priority``).
_DOC_EXTENSIONS = (".md", ".mdx", ".rst", ".txt")

# Filename keywords mapped to categories, used as the fallback classifier.
# Matched at word boundaries rather than anywhere in the path -- see
# ``_mentions_keyword`` for why.
_HEURISTIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "readme": ("readme",),
    "setup": (
        "setup",
        "install",
        "installation",
        "getting-started",
        "getting_started",
        "quickstart",
        "contributing",
        "help",
        "agents",
        "env.example",
    ),
    "architecture": ("architecture", "design"),
    "adr": ("adr", "adrs", "decision-record", "decision_record"),
    # "reference" deliberately absent: it matched every file under a
    # ``references/`` directory, so a wiki that files its conventions and
    # working agreements there read as fully API-documented. An actual API
    # reference is caught by "api" anyway.
    "api": ("api", "apis", "openapi", "swagger"),
    "runbook": (
        "runbook",
        "runbooks",
        "playbook",
        "operations",
        "ops",
        "devops",
        "oncall",
        "on-call",
    ),
}


def _mentions_keyword(name: str, keyword: str) -> bool:
    """Whether ``name`` contains ``keyword`` as a whole word.

    A plain substring test is far too eager on paths: "api" hides inside
    "rapid" and "capital", "ops" inside "props", "adr" inside "adrenaline".
    The heuristic can only ever *add* categories (see ``_classify_present``),
    so a false positive here is one the LLM can never take back -- it silently
    turns a real gap into full coverage, which is the failure mode this whole
    function exists to prevent.

    Both ends are anchored. Legitimate suffixed forms are not inferred from an
    open-ended prefix match but listed outright in ``_HEURISTIC_KEYWORDS``
    ("apis", "installation", "adrs"), so the set of things that count as
    documentation stays something you can read off the table rather than a
    consequence of how the matcher happens to work.
    """
    pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
    return re.search(pattern, name) is not None


@dataclass(frozen=True)
class KnowledgeGap:
    component: str
    missing_types: list[str]
    present_types: list[str]
    last_updated: str
    severity: Severity


def _component_of(record: ArtifactRecord) -> str | None:
    """Derive an ``owner/repo`` component from an artifact's ``source_id``.

    Source ids from the GitHub ingestion run have the shape
    ``"github:owner/repo:TYPE:..."``; the second colon-separated segment is the
    repository. Artifacts without such a segment (e.g. ad-hoc uploads) have no
    derivable component and are skipped by the caller.
    """
    source_id = record.source_id
    if not source_id:
        return None
    parts = source_id.split(":")
    if len(parts) >= 2 and "/" in parts[1]:
        return parts[1]
    return None


def _doc_priority(record: ArtifactRecord) -> int:
    """Lower sorts first: readme, then other doc-like files, then everything
    else. Applied before the ``_MAX_DOCS_PER_COMPONENT`` cap so a component
    with many files doesn't have its README or other docs truncated out of
    the classifier's sample."""
    name = record.filename.lower()
    if "readme" in name:
        return 0
    if name.endswith(_DOC_EXTENSIONS):
        return 1
    return 2


def _doc_snippets(
    records: list[ArtifactRecord],
    store: VectorStore,
) -> list[tuple[str, str]]:
    """Return ``(filename, text snippet)`` pairs for a component's documents."""
    prioritized = sorted(records, key=_doc_priority)
    snippets: list[tuple[str, str]] = []
    for record in prioritized[:_MAX_DOCS_PER_COMPONENT]:
        text = ""
        chunks = store.list_chunks_by_artifact(record.id, limit=1)
        if chunks:
            text = chunks[0].text[:_SNIPPET_CHARS].replace("\n", " ").strip()
        snippets.append((record.filename, text))
    return snippets


def _heuristic_present(records: list[ArtifactRecord]) -> set[str]:
    """Filename-based fallback classification of present categories."""
    present: set[str] = set()
    for record in records:
        name = record.filename.lower()
        for category, keywords in _HEURISTIC_KEYWORDS.items():
            if any(_mentions_keyword(name, keyword) for keyword in keywords):
                present.add(category)
    return present


def _build_classify_prompt(
    component: str,
    snippets: list[tuple[str, str]],
) -> list[Message]:
    docs = "\n".join(
        f"- {filename}: {text}" if text else f"- {filename}"
        for filename, text in snippets
    )
    categories = ", ".join(EXPECTED_TYPES)
    system = (
        "You assess the documentation coverage of a software component. You are "
        "given the component's documents (filename and an optional snippet). "
        "Decide which of these documentation categories the component already "
        f"has substantive coverage for: {categories}.\n\n"
        "A category counts as present only if at least one document genuinely "
        "serves that purpose — do not guess from a filename alone if the snippet "
        "contradicts it. Return STRICT JSON only (no prose, no markdown fences) "
        'with this schema: {"present": [<category>, ...]}. Use only categories '
        "from the list above."
    )
    user = f"Component: {component}\n\nDocuments:\n{docs}"
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _classify_present(
    component: str,
    records: list[ArtifactRecord],
    llm: LLMClient,
    store: VectorStore,
) -> set[str]:
    """Classify which expected categories the component covers.

    The filename heuristic runs unconditionally and its result is *unioned*
    with the LLM's, rather than only substituting for it when the LLM output
    fails to parse. This means an obvious signal (README.md exists) can never
    be overridden by a confidently-wrong LLM classification — the LLM can
    only add categories the heuristic missed, never remove ones it found.
    ``LLMUnavailableError`` is allowed to propagate so the endpoint can
    surface a 503.
    """
    heuristic = _heuristic_present(records)
    snippets = _doc_snippets(records, store)
    raw = llm.generate(_build_classify_prompt(component, snippets))
    try:
        payload = json.loads(extract_json_object(raw))
        if not isinstance(payload, dict):
            raise ValueError("classification output is not an object")
        present = cast(dict[str, object], payload).get("present")
        if not isinstance(present, list):
            raise ValueError("'present' is not a list")
        llm_present = {str(item) for item in cast(list[object], present)} & set(
            EXPECTED_TYPES
        )
    except (json.JSONDecodeError, ValueError, TypeError) as exc:
        logger.warning(
            "Knowledge-gap classification for %s fell back to heuristic: %s",
            component,
            exc,
        )
        return heuristic
    return llm_present | heuristic


def _is_stale(last_updated: str) -> bool:
    try:
        updated = datetime.fromisoformat(last_updated)
    except ValueError:
        return False
    if updated.tzinfo is None:
        updated = updated.replace(tzinfo=UTC)
    age_days = (datetime.now(UTC) - updated).days
    return age_days > _STALE_AFTER_DAYS


def _severity(missing: list[str], last_updated: str) -> Severity:
    """Rank a gap.

    A component with nothing missing is "covered" rather than scored: staleness
    alone would otherwise push a fully documented component to "low", which is
    indistinguishable from one that is actually missing something.

    Missing critical categories (readme/setup) weigh far more than missing
    optional ones: each missing critical type is worth 3 points, so missing
    even one already reaches "medium" and missing both reaches "high" on its
    own. Missing optional categories are capped at 3 points total so that a
    component with solid critical coverage but several missing optional
    categories doesn't get inflated to "high" purely from their count —
    without the cap, e.g. 4 missing optional categories alone would already
    cross the "high" threshold even though readme/setup both exist.
    """
    if not missing:
        return "covered"

    missing_set = set(missing)
    critical_missing = len(missing_set & CRITICAL_TYPES)
    noncritical_missing = min(len(missing_set - CRITICAL_TYPES), 3)
    score = 3 * critical_missing + noncritical_missing
    if _is_stale(last_updated):
        score += 1
    if score >= 4:
        return "high"
    if score >= 2:
        return "medium"
    return "low"


_SEVERITY_RANK = {"high": 0, "medium": 1, "low": 2, "covered": 3}


def detect_knowledge_gaps(
    llm: LLMClient,
    store: VectorStore,
    metadata_store: IngestionMetadataStore,
    project_id: str,
) -> list[KnowledgeGap]:
    """Report documentation coverage for every component of one project.

    Components missing nothing are included with severity "covered", so the
    result is the project's full roster and an absence means "not ingested"
    rather than "nothing to report".

    Only artifacts belonging to ``project_id`` are considered, so a component
    is never reported to — or judged by the documents of — another project.
    """
    components: dict[str, list[ArtifactRecord]] = {}
    for record in metadata_store.list_artifacts(project_id=project_id):
        component = _component_of(record)
        if component is None:
            continue
        components.setdefault(component, []).append(record)

    gaps: list[KnowledgeGap] = []
    for component, records in sorted(components.items()):
        present = _classify_present(component, records, llm, store)
        missing = [t for t in EXPECTED_TYPES if t not in present]
        # Fully covered components are reported too, as "covered". Dropping them
        # made a component that is in good shape indistinguishable from one that
        # was never ingested, and "this repository has no gaps" is a finding a PM
        # wants to see rather than infer from an absence.
        last_updated = max(record.updated_at for record in records)
        gaps.append(
            KnowledgeGap(
                component=component,
                missing_types=missing,
                present_types=[t for t in EXPECTED_TYPES if t in present],
                last_updated=last_updated,
                severity=_severity(missing, last_updated),
            )
        )

    gaps.sort(key=lambda g: (_SEVERITY_RANK[g.severity], g.component))
    return gaps
