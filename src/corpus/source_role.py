"""Classification of a source document's *role* in the corpus.

Orthogonal to :data:`ingestion.models.ChunkKind` (which describes the *content*
type — text, code, pdf, image). ``SourceRole`` describes what the file *is to
the team*: ``primary`` project material vs ``test`` material (test code and test
fixtures/sample data).

Test material is still ingested and searchable (e.g. for codebase Q&A), but it
is not representative of how the project actually works, so consumers like the
onboarding generator exclude it from grounding evidence.
"""

import re
from typing import Literal, TypeGuard

SourceRole = Literal["primary", "test"]

DEFAULT_SOURCE_ROLE: SourceRole = "primary"

# Roles that are searchable (for codebase Q&A) but never used as grounding
# evidence for onboarding, because they don't represent how the project actually
# works. Single source of truth for every consumer that grounds against the
# corpus, so their evidence policy can't drift apart.
GROUNDING_EXCLUDED_ROLES: frozenset[SourceRole] = frozenset({"test"})

# Basename patterns that reliably indicate test material. The ingest API only
# receives a basename (no path), so classification is best-effort; callers that
# know the original path should pass ``source_role`` explicitly instead.
_TEST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^test_.*", re.IGNORECASE),  # pytest:    test_foo.py
    re.compile(r".*_test\.[^.]+$", re.IGNORECASE),  # go/js:     foo_test.go
    re.compile(r".*\.(test|spec)\.[^.]+$", re.IGNORECASE),  # js:        foo.spec.ts
    re.compile(r"^conftest\.py$", re.IGNORECASE),  # pytest config
    re.compile(r".*_(sample|fixture)s?\.[^.]+$", re.IGNORECASE),  # fixtures/sample data
)


def is_source_role(value: str) -> TypeGuard[SourceRole]:
    return value in ("primary", "test")


def classify_source_role(filename: str) -> SourceRole:
    """Best-effort role from a basename; ``primary`` unless it looks like a test."""
    name = filename.strip()
    if any(pattern.match(name) for pattern in _TEST_PATTERNS):
        return "test"
    return DEFAULT_SOURCE_ROLE
