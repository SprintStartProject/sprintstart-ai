"""Request and response of the exit-interview planner (``POST
/offboarding/interview-questions``).

Mirrored by the backend's ``offboarding`` module, which sends the files of a stored or
computed risk map and shows the plan to the project manager.
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel

from api.schemas import ProjectScopedRequest
from insights.interview_components import Trigger
from insights.interview_prompts import Coverage, QuestionType

_CAMEL = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class InterviewFileSchema(BaseModel):
    model_config = _CAMEL

    repository: str = Field(
        description="Repository or space key, e.g. github:acme/app."
    )
    path: str = Field(description="File path, or page title for a documentation page.")
    level: str = Field(description="ORPHANED or HOARDED, as the risk map says.")
    score: float = Field(ge=0, description="The file's risk score from the risk map.")
    kind: str | None = Field(default=None, description="CODE, PROSE or CONFIG.")
    hotspot: float | None = None
    abandoned_share: float | None = None
    coupled_with: list[str] = Field(
        default_factory=list[str], description="Paths of files that change with it."
    )
    reasons: list[str] = Field(
        default_factory=list[str],
        description="The risk map's sentences; they name nobody.",
    )


class InterviewQuestionsRequest(ProjectScopedRequest):
    model_config = _CAMEL

    files: list[InterviewFileSchema] = Field(
        min_length=1, description="The files at risk, as ranked by the backend."
    )
    max_components: int = Field(default=8, ge=1, le=20)
    per_component: int = Field(default=4, ge=1, le=10)


class InterviewQuestionSchema(BaseModel):
    """One question; ``triggers`` are the risk-map findings it answers, and
    ``recordAnswer`` with ``sources`` is what the records already say when
    ``coverage`` is partial or answered."""

    model_config = _CAMEL

    component: str
    text: str
    type: QuestionType
    triggers: list[Trigger]
    files: list[str]
    priority: float
    coverage: Coverage
    record_answer: str | None = None
    sources: list[str] = []


class InterviewQuestionsResponse(BaseModel):
    """``questions`` for the leaver, most urgent first; ``answeredByRecords`` the
    ones the project's records answer, so they are not asked."""

    model_config = _CAMEL

    questions: list[InterviewQuestionSchema]
    answered_by_records: list[InterviewQuestionSchema]
