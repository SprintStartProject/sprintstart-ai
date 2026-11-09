"""Token-set overlap (Jaccard): the deterministic duplicate check of the assembly jobs.

No embeddings and no external calls, so a re-run gives the same answer.
"""

import re

# Two texts whose Jaccard index exceeds this are considered duplicates.
OVERLAP_THRESHOLD: float = 0.55

# Minimal stopword set — common English function words only.  Domain terms
# like "install", "setup", "verify" are intentionally excluded because they
# carry semantic weight for distinguishing onboarding steps.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "do",
        "for",
        "from",
        "in",
        "is",
        "it",
        "not",
        "of",
        "on",
        "or",
        "set",
        "that",
        "the",
        "this",
        "to",
        "up",
        "was",
        "with",
        "you",
        "your",
    }
)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower())) - _STOPWORDS


def text_overlap(a: str, b: str) -> float:
    """Jaccard index over normalised, stopword-filtered token sets."""
    tokens_a = _tokenize(a)
    tokens_b = _tokenize(b)
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
