"""What the model drafts for an orientation packet, and how that draft is read.

The generation models, the prompt, the correction round, and the resolution of a
draft against the retrieved evidence: everything that depends on the model's output
rather than on the order of the pipeline stages in assembly.orientation.job.
"""

import json
import logging

from pydantic import BaseModel, Field, ValidationError

from assembly.citations import resolve_citations
from assembly.orientation.models import (
    STEP_ORDER,
    OrientationSection,
    OrientationStep,
)
from assembly.similarity import OVERLAP_THRESHOLD, text_overlap
from assembly.stages import (
    AssemblyError,
    evidence_line,
)
from corpus.types import ScoredChunk
from llm.base import Message
from llm.parsing import extract_json_object

logger = logging.getLogger(__name__)

TOP_K_PER_STEP = 6
_MAX_TASK_TEXT = 4000

# What each step's retrieval is looking for. These are *queries*, not prose the
# hire ever sees — they exist so the evidence pool spans the whole path to a PR
# instead of five rephrasings of the task's own subject.
STEP_QUERIES: dict[OrientationStep, str] = {
    "SET_UP": (
        "development environment setup prerequisites install dependencies "
        "build and run the project locally configuration"
    ),
    "FIND_THE_CODE": (
        "project structure where modules live architecture packages layout "
        "which file is responsible for"
    ),
    "MAKE_THE_CHANGE": (
        "coding conventions style guide patterns used in this codebase "
        "how changes are usually implemented"
    ),
    "CHECK_LOCALLY": (
        "run the tests lint format type check verification commands before "
        "pushing definition of done"
    ),
    "OPEN_THE_PR": (
        "contributing guide pull request process branch naming commit message "
        "code review expectations"
    ),
}

# Short human labels for the live progress stream — what the watcher sees a step
# called while it is searched / grounded. Distinct from `_STEP_GUIDE` (prompt text)
# and `STEP_QUERIES` (retrieval text): this is display only.
STEP_LABEL: dict[OrientationStep, str] = {
    "SET_UP": "setting up",
    "FIND_THE_CODE": "finding the code",
    "MAKE_THE_CHANGE": "making the change",
    "CHECK_LOCALLY": "checking locally",
    "OPEN_THE_PR": "opening the PR",
}

# What the model is told each step is *for*, in the packet the hire reads.
_STEP_GUIDE = (
    "  SET_UP          - what must be working before touching this task\n"
    "  FIND_THE_CODE   - where in this repository the work lives\n"
    "  MAKE_THE_CHANGE - the conventions that apply to a change like this\n"
    "  CHECK_LOCALLY   - what to run, and what a good result looks like\n"
    "  OPEN_THE_PR     - how a change gets proposed and reviewed here\n"
)


class _GenSection(BaseModel):
    step: str = ""
    title: str = ""
    body: str = ""
    chunk_ids: list[str] = Field(default_factory=list[str])


class GenPayload(BaseModel):
    summary: str = ""
    sections: list[_GenSection] = Field(default_factory=list[_GenSection])


def task_query(
    task_title: str, task_body: str, labels: list[str], touched_paths: list[str]
) -> str:
    parts = [task_title, " ".join(labels), " ".join(touched_paths), task_body]
    return " ".join(p for p in parts if p.strip())[:_MAX_TASK_TEXT]


def build_prompt(
    task_title: str,
    task_body: str,
    labels: list[str],
    touched_paths: list[str],
    chunks: list[ScoredChunk],
) -> list[Message]:
    evidence = "\n".join(evidence_line(c) for c in chunks)
    steps = "|".join(STEP_ORDER)

    system = (
        "You prepare a new team member to do one specific task in a codebase "
        "they have never worked in. You do NOT write documentation: everything "
        "you produce is a restatement of material this team already has, "
        "reorganised around the task and the order somebody actually does "
        "things in.\n\n"
        "The packet is segmented by step:\n"
        f"{_STEP_GUIDE}\n"
        "Rules:\n"
        "1. Write only what the evidence below supports. Every section MUST "
        "list the evidence chunk ids it draws on; a section citing nothing is "
        "dropped, however useful it sounds. If the evidence says nothing about "
        "a step, omit that step -- an honest gap beats an invented answer.\n"
        "2. Be concrete and specific to this task: exact paths, exact commands, "
        "exact conventions. Generic software advice is worse than nothing "
        "because it costs the reader attention.\n"
        "3. Do not repeat yourself across sections. When two sources say the "
        "same thing, say it once and cite both.\n"
        "4. Keep each section short: a reader is doing the task, not studying.\n"
        "5. You may emit more than one section for a step, or none at all.\n\n"
        "Return STRICT JSON only (no prose, no markdown fences):\n"
        '{"summary": str, "sections": [{"step": '
        f"{steps}"
        ', "title": str, "body": str (markdown), "chunk_ids": [str]}]}'
    )
    label_line = f"Labels: {', '.join(labels)}\n" if labels else ""
    paths_line = (
        f"Paths this task is expected to touch: {', '.join(touched_paths)}\n"
        if touched_paths
        else ""
    )
    user = (
        f"Task: {task_title}\n"
        f"{label_line}"
        f"{paths_line}"
        f"\n{task_body[:_MAX_TASK_TEXT] or '(no description)'}\n\n"
        f"Evidence:\n{evidence}"
    )
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def correction_prompt(
    messages: list[Message], raw: str, error: AssemblyError
) -> list[Message]:
    """Ask once for the same packet with only its JSON corrected."""
    return [
        *messages,
        Message(role="assistant", content=raw),
        Message(
            role="user",
            content=(
                f"That response could not be validated: {error}. Return the same "
                "grounded packet again as one valid JSON object matching the schema "
                "exactly. Return JSON only, with all quotes escaped and all commas "
                "present."
            ),
        ),
    ]


def parse_payload(raw: str) -> GenPayload:
    try:
        return GenPayload.model_validate_json(extract_json_object(raw))
    except (ValidationError, json.JSONDecodeError, ValueError) as exc:
        raise AssemblyError(f"invalid orientation output: {exc}") from exc


def resolve_sections(
    payload: GenPayload, chunks: list[ScoredChunk]
) -> tuple[list[OrientationSection], int]:
    """Keep the usable sections in step order; return them with the count dropped.

    A section goes when its step is not one of ours, when it has no body, when
    it cites nothing real, or when it restates a section already kept for the
    same step.
    """
    chunks_by_id = {c.id: c for c in chunks}
    kept: list[OrientationSection] = []
    dropped = 0

    for item in payload.sections:
        step = item.step.strip().upper()
        if step not in STEP_ORDER:
            logger.info("Dropped orientation section of unknown step %r", item.step)
            dropped += 1
            continue
        if not item.body.strip():
            dropped += 1
            continue

        citations = resolve_citations(item.chunk_ids, chunks_by_id)
        if not citations:
            logger.info("Dropped ungrounded orientation section %r", item.title)
            dropped += 1
            continue

        if any(
            s.step == step and text_overlap(s.body, item.body) > OVERLAP_THRESHOLD
            for s in kept
        ):
            logger.info("Dropped orientation section restating %r", item.title)
            dropped += 1
            continue

        kept.append(
            OrientationSection(
                step=step,  # type: ignore[arg-type]
                title=item.title.strip() or step.replace("_", " ").title(),
                body=item.body,
                citations=citations,
            )
        )

    kept.sort(key=lambda s: STEP_ORDER.index(s.step))
    return kept, dropped
