"""Models the onboarding assembly jobs share: citations and check questions."""

from typing import Literal

from pydantic import BaseModel, Field

# Matches the backend's ``CheckQuestionType`` enum constants exactly, so the
# generated check needs no case translation on the consuming side.
CheckQuestionType = Literal["MULTIPLE_CHOICE", "SHORT_TEXT"]


class CitationRef(BaseModel):
    """A resolved reference to an ingested document chunk."""

    filename: str
    chunk_id: str
    source_url: str | None = None


class CheckOption(BaseModel):
    """One answer option of a MULTIPLE_CHOICE check question."""

    position: int
    label: str
    correct: bool = False


class CheckQuestion(BaseModel):
    """One knowledge-check question, grounded in its phase's content.

    ``correct_answer`` is only meaningful for ``SHORT_TEXT`` questions;
    ``options`` is only meaningful for ``MULTIPLE_CHOICE`` ones. When a phase
    is AI-assembled, ``key`` and ``blocked_by`` carry optional dependency
    edges: ``blocked_by`` lists the keys of earlier steps/questions this
    question must wait for.
    """

    position: int
    type: CheckQuestionType
    question: str
    explanation: str | None = None
    correct_answer: str | None = None
    options: list[CheckOption] = Field(default_factory=list[CheckOption])
    key: str = ""
    blocked_by: list[str] = Field(default_factory=list[str])
