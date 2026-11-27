"""Prompts and parsers of the two model calls behind an exit-interview plan.

The first call writes questions for one component from what the risk map found; the
second judges, for each question, whether the project's own records already answer it.
Both answer in strict JSON that is validated here, so a malformed answer degrades the
plan instead of breaking it.
"""

import json
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ValidationError

from insights.interview_components import Component, RiskFile, Trigger
from llm.base import Message
from llm.parsing import extract_json_object


class QuestionType(StrEnum):
    """What a question tries to get out of the leaver."""

    RATIONALE = "rationale"
    OPERATIONAL = "operational"
    GOTCHA = "gotcha"
    HANDOVER = "handover"


class Coverage(StrEnum):
    """How much of a question the project's records already answer."""

    ANSWERED = "answered"
    PARTIAL = "partial"
    OPEN = "open"


@dataclass(frozen=True)
class DraftQuestion:
    text: str
    type: QuestionType
    triggers: tuple[Trigger, ...]
    files: tuple[str, ...]


@dataclass(frozen=True)
class Verdict:
    coverage: Coverage
    answer: str | None
    sources: tuple[str, ...]


@dataclass(frozen=True)
class Excerpt:
    filename: str
    text: str


_TRIGGER_MEANING = {
    Trigger.ORPHANED: "nobody who stays knows these files",
    Trigger.ABANDONED: "most of their lines were written by people who left",
    Trigger.HOTSPOT: "they are complicated and change often",
    Trigger.COUPLED: "they usually change together with other files",
    Trigger.CONCENTRATED: "one person holds almost all of the knowledge",
}

_GENERATE_RULES = (
    "You prepare an exit interview with a developer who is leaving a software "
    "project. The project's risk analysis found files whose knowledge leaves with "
    "this developer. Write the questions only this developer can answer and that a "
    "successor would need: why it is built this way (rationale), how to run, deploy"
    " or fix it (operational), what breaks in surprising ways (gotcha), and who or "
    "what to watch when taking it over (handover). Every question must be specific "
    "to the files and findings given, name the file or concept it is about, and be "
    "answerable in a few sentences. Never ask what the excerpts already state "
    "plainly. Never ask about people's performance. Return STRICT JSON only, no "
    "prose and no markdown fences: "
    '{"questions": [{"text": str, '
    '"type": "rationale"|"operational"|"gotcha"|"handover", '
    '"triggers": [<finding names>], "files": [<paths>]}]}. '
    "Use only the finding names and paths given."
)

_JUDGE_RULES = (
    "You check whether a project's records already answer interview questions. For "
    "each question you get excerpts from the project's documents and code. Judge "
    "only from the excerpts: 'answered' when they state the full answer, 'partial' "
    "when they state part of it, 'open' otherwise. Do not use your own knowledge. "
    "For answered or partial give the answer in one or two sentences and the "
    "filenames it comes from. Return STRICT JSON only: "
    '{"verdicts": [{"index": int, "coverage": "answered"|"partial"|"open", '
    '"answer": str|null, "sources": [<filenames>]}]}.'
)


def _file_line(file: RiskFile) -> str:
    reasons = " ".join(file.reasons)
    kind = f" ({file.kind.lower()})" if file.kind else ""
    return f"- {file.path}{kind}: {reasons}".rstrip()


def generate_prompt(
    component: Component, excerpts: list[Excerpt], per_component: int
) -> list[Message]:
    """Asks for at most ``per_component`` questions about one component."""
    findings = "\n".join(
        f"- {t.value}: {_TRIGGER_MEANING[t]}" for t in sorted(component.triggers)
    )
    files = "\n".join(_file_line(file) for file in component.files)
    context = "\n\n".join(f"[{e.filename}]\n{e.text}" for e in excerpts) or "(none)"
    user = (
        f"Component: {component.name}\n\nFindings:\n{findings}"
        f"\n\nFiles at risk:\n{files}"
        f"\n\nExcerpts from the project's records:\n{context}\n\n"
        f"Write at most {per_component} questions, most important first."
    )
    return [
        Message(role="system", content=_GENERATE_RULES),
        Message(role="user", content=user),
    ]


def judge_prompt(
    questions: list[DraftQuestion], evidence: list[list[Excerpt]]
) -> list[Message]:
    """Asks whether each question is answered by its own excerpts."""
    blocks: list[str] = []
    for index, (question, excerpts) in enumerate(zip(questions, evidence, strict=True)):
        found = "\n".join(f"[{e.filename}] {e.text}" for e in excerpts) or "(none)"
        blocks.append(f"Question {index}: {question.text}\nExcerpts:\n{found}")
    return [
        Message(role="system", content=_JUDGE_RULES),
        Message(role="user", content="\n\n".join(blocks)),
    ]


class _QuestionEntry(BaseModel):
    text: str
    type: QuestionType
    triggers: list[str] = []
    files: list[str] = []


class _QuestionPayload(BaseModel):
    questions: list[_QuestionEntry] = []


class _VerdictEntry(BaseModel):
    index: int
    coverage: Coverage
    answer: str | None = None
    sources: list[str] = []


class _VerdictPayload(BaseModel):
    verdicts: list[_VerdictEntry] = []


def parse_questions(
    raw: str, component: Component, per_component: int
) -> list[DraftQuestion]:
    """The questions of a generation answer, keeping only findings and files of
    ``component``.

    Raises ``ValueError`` when the answer is not the expected JSON.
    """
    try:
        payload = _QuestionPayload.model_validate(json.loads(extract_json_object(raw)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(str(exc)) from exc
    paths = {file.path for file in component.files}
    allowed = {t.value for t in component.triggers}
    drafts = [
        DraftQuestion(
            text=entry.text.strip(),
            type=entry.type,
            triggers=tuple(Trigger(t) for t in entry.triggers if t in allowed)
            or (Trigger.CONCENTRATED,),
            files=tuple(f for f in entry.files if f in paths),
        )
        for entry in payload.questions
        if entry.text.strip()
    ]
    return drafts[:per_component]


def parse_verdicts(raw: str, count: int) -> list[Verdict]:
    """One verdict per question; a question the judge skipped stays open.

    Raises ``ValueError`` when the answer is not the expected JSON.
    """
    try:
        payload = _VerdictPayload.model_validate(json.loads(extract_json_object(raw)))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(str(exc)) from exc
    verdicts = [Verdict(Coverage.OPEN, None, ()) for _ in range(count)]
    for entry in payload.verdicts:
        if 0 <= entry.index < count:
            answer = entry.answer if entry.coverage != Coverage.OPEN else None
            verdicts[entry.index] = Verdict(
                entry.coverage, answer, tuple(entry.sources)
            )
    return verdicts
