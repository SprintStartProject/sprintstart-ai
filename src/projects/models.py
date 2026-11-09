from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class IndustryEvaluation:
    """The industry or domain detected for a project; empty when undetermined."""

    industry: str
    confidence: Literal["high", "medium", "low"]
    evidence: list[str] = field(default_factory=list[str])
