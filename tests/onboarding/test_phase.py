import json
from collections.abc import Generator

from onboarding.phase import assemble_phase, stream_phase
from onboarding.progress import ProgressEvent
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_PROJECT = "p1"
_EMBED = [1.0] + [0.0] * 767


def _collect[T](
    generator: Generator[ProgressEvent, None, T],
) -> tuple[list[ProgressEvent], T]:
    """Drain a progress generator, keeping both the events and the returned value."""
    events: list[ProgressEvent] = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        return events, stop.value


def _llm(payload: dict[str, object]) -> StubLLMClient:
    llm = StubLLMClient(generate_response=json.dumps(payload))
    llm.embedding = _EMBED
    return llm


def _store(*texts: str) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id=f"a{i}",
                filename=f"doc{i}.md",
                text=text,
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


def _step(title: str, chunk_ids: list[str] | None = None) -> dict[str, object]:
    step: dict[str, object] = {
        "title": title,
        "description": f"{title} description",
        "tasks": [],
        "resources": [],
        "estimated_minutes": 10,
        "expected_outcome": f"{title} outcome",
    }
    if chunk_ids is not None:
        step["chunk_ids"] = chunk_ids
    return step


def _question(q_type: str = "SHORT_TEXT", **fields: object) -> dict[str, object]:
    question: dict[str, object] = {
        "type": q_type,
        "question": f"Explain {q_type}",
    }
    if q_type == "SHORT_TEXT":
        question["correct_answer"] = "Because the evidence says so"
    else:
        question["options"] = [
            {"label": "Right", "correct": True},
            {"label": "Wrong", "correct": False},
        ]
    question.update(fields)
    return question


def _payload(
    steps: list[dict[str, object]] | None = None,
    questions: list[dict[str, object]] | None = None,
    **extra: object,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "steps": steps or [],
        "questions": questions or [],
    }
    payload.update(extra)
    return payload


def test_assembles_grounded_steps_and_questions() -> None:
    store = _store(
        "the README explains the project and how to run it locally",
        "coding conventions require a PR to pass review",
    )
    llm = _llm(
        _payload(
            steps=[_step("Read the README", ["c1"])],
            questions=[_question()],
        )
    )

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview for a new member.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(outcome.steps) == 1
    step = outcome.steps[0]
    assert step.title == "Read the README"
    assert step.estimated_minutes == 10
    assert len(outcome.check_questions) == 1
    assert outcome.check_questions[0].question == "Explain SHORT_TEXT"


def test_drops_an_ungrounded_step_but_keeps_the_phase() -> None:
    """A step with no source does not ship -- that is the phase's grounding rule."""
    store = _store("the README explains the project and how to run it locally")
    llm = _llm(
        _payload(
            steps=[
                _step("Read the README", ["c1"]),
                _step("Invented step", ["nope"]),
            ]
        )
    )

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    assert [s.title for s in outcome.steps] == ["Read the README"]
    assert outcome.steps_dropped == 1


def test_drops_invalid_questions_and_keeps_the_valid_ones() -> None:
    """A quiz with a flawed question is not thrown away wholesale."""
    store = _store("project security conventions are documented in CONTRIBUTING.md")
    llm = _llm(
        _payload(
            steps=[_step("Follow the conventions", ["c1"])],
            questions=[
                _question(),
                # All-correct MULTIPLE_CHOICE discriminates nothing: dropped.
                _question(
                    q_type="MULTIPLE_CHOICE",
                    question="Pick all",
                    options=[
                        {"label": "A", "correct": True},
                        {"label": "B", "correct": True},
                    ],
                ),
            ],
        )
    )

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Conventions",
        phase_prompt="Cover the security conventions.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(outcome.check_questions) == 1
    assert outcome.questions_dropped == 1


def test_skipped_when_the_corpus_is_empty() -> None:
    store = StubVectorStore()
    llm = _llm({})

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
    )

    assert outcome.status == "skipped"
    assert outcome.steps == []


def test_skipped_when_every_step_is_ungrounded() -> None:
    store = _store("the README explains the project")
    llm = _llm(_payload(steps=[_step("Invented step", ["missing"])]))

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
    )

    assert outcome.status == "skipped"
    assert outcome.steps == []
    assert outcome.steps_dropped == 1


def test_unchanged_when_the_corpus_fingerprint_matches() -> None:
    """Idempotency: an unchanged corpus short-circuits retrieval and generation."""
    store = _store("the README explains the project and how to run it locally")
    first = assemble_phase(
        _llm(_payload(steps=[_step("Read the README", ["c1"])])),
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
    )
    assert first.status == "assembled"
    fingerprint = first.provenance.corpus_fingerprint
    assert fingerprint is not None

    second = assemble_phase(
        _llm(json.dumps({"steps": []})),
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
        last_fingerprint=fingerprint,
    )

    assert second.status == "unchanged"


def test_stream_emits_item_per_grounded_step_then_terminal_done() -> None:
    store = _store("the README explains the project and how to run it locally")
    llm = _llm(_payload(steps=[_step("Read the README", ["c1"])]))

    events, outcome = _collect(
        stream_phase(
            llm,
            store,
            phase_title="Project Overview",
            phase_prompt="Generate an overview.",
            project_id=_PROJECT,
        )
    )

    assert outcome.status == "assembled"
    types = [e["type"] for e in events]
    assert types[0] == "stage"
    assert "item" in types
    assert types[-1] == "done"
    item = next(e for e in events if e["type"] == "item")
    assert item["item"]["title"] == "Read the README"  # type: ignore[typeddict-item]
