from dataclasses import dataclass, field
from typing import Literal

Confidence = Literal["high", "medium", "low"]


@dataclass(frozen=True)
class SkillCatalogEntry:
    """An existing skill offered to the model as catalog context."""

    id: str
    name: str
    category: str | None = None
    universal: bool = False


@dataclass(frozen=True)
class SkillSuggestion:
    """A recommended skill with its rationale and the chunks that ground it."""

    name: str
    category: str | None
    reason: str
    confidence: Confidence
    is_new: bool = False
    chunk_ids: list[str] = field(default_factory=list[str])


@dataclass(frozen=True)
class SkillSuggestionResult:
    suggestions: list[SkillSuggestion] = field(default_factory=list[SkillSuggestion])
