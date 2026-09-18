import json
from collections.abc import Generator

from llm.base import Message
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


def test_accepts_literal_control_characters_in_generated_strings() -> None:
    """Model output with an unescaped newline remains usable phase content."""
    payload = _payload(steps=[_step("Read the README", ["c1"])])
    payload["steps"][0]["description"] = "Read first\nThen run the application"  # type: ignore[index]
    raw = json.dumps(payload).replace("\\n", "\n")
    llm = StubLLMClient(generate_response=raw, embedding=_EMBED)

    outcome = assemble_phase(
        llm,
        _store("the README explains the project and how to run it locally"),
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    assert outcome.steps[0].description == "Read first\nThen run the application"


def test_treats_null_short_text_options_as_an_empty_list() -> None:
    """A model's explicit null for inapplicable options does not skip the phase."""
    outcome = assemble_phase(
        _llm(
            _payload(
                steps=[_step("Read the README", ["c1"])],
                questions=[_question(options=None)],
            )
        ),
        _store("the README explains the project and how to run it locally"),
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(outcome.check_questions) == 1
    assert outcome.check_questions[0].options == []


def test_accepts_tasks_as_plain_strings() -> None:
    """A compact model task remains usable instead of invalidating the phase."""
    step = _step("Review the repository", ["c1"])
    step["tasks"] = ["Open README.md", "Search for meeting references"]

    outcome = assemble_phase(
        _llm(_payload(steps=[step])),
        _store("the README documents the repository structure"),
        phase_title="Architecture",
        phase_prompt="Review the repository.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    assert [task.title for task in outcome.steps[0].tasks] == [
        "Open README.md",
        "Search for meeting references",
    ]
    assert all(task.description == "" for task in outcome.steps[0].tasks)


def test_retries_once_when_generated_json_is_invalid() -> None:
    """A transient syntax error gets one correction pass before giving up."""
    valid = json.dumps(_payload(steps=[_step("Read the README", ["c1"])]))

    class _CorrectingLLM(StubLLMClient):
        def __init__(self) -> None:
            super().__init__(embedding=_EMBED)
            self.calls: list[list[Message]] = []

        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            self.calls.append(messages)
            return '{"steps": [' if len(self.calls) == 1 else valid

    llm = _CorrectingLLM()
    outcome = assemble_phase(
        llm,
        _store("the README explains the project"),
        phase_title="Architecture",
        phase_prompt="Review the repository.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(llm.calls) == 2
    assert "could not be validated" in llm.calls[1][-1]["content"]


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


def test_resolves_dependency_edges_between_grounded_items() -> None:
    """blocked_by keys survive only when they name an earlier kept item."""
    store = _store(
        "the README explains the project and how to run it locally",
        "a build requires the environment to be configured first",
    )
    readme = _step("Read the README", ["c1"])
    readme["key"] = "s1"
    setup = _step("Set up the environment", ["c2"])
    setup["key"] = "s2"
    setup["blocked_by"] = ["s1", "missing"]
    ghost = _step("Ungrounded step", ["nope"])
    ghost["key"] = "s3"
    later = _step("Depends on a dropped step", ["c1"])
    later["key"] = "s4"
    later["blocked_by"] = ["s3", "s2"]
    selfie = _step("Self-referencing step", ["c1"])
    selfie["key"] = "s5"
    selfie["blocked_by"] = ["s5"]

    outcome = assemble_phase(
        _llm(
            _payload(
                steps=[readme, setup, ghost, later, selfie],
                questions=[_question(key="q1", blocked_by=["s1", "unknown"])],
            )
        ),
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    by_key = {s.key: s for s in outcome.steps}
    assert by_key["s2"].blocked_by == ["s1"]
    # Reference to an ungrounded step and a self-reference are dropped.
    assert by_key["s4"].blocked_by == ["s2"]
    assert by_key["s5"].blocked_by == []
    assert outcome.check_questions[0].blocked_by == ["s1"]


def test_dependency_edges_survive_end_to_end() -> None:
    """A genuine ordering chain comes back as usable blocked_by edges, not [].

    This is the behavior the prompt now asks for: when items have a real order
    (read the README -> run its build -> follow the conventions), the phase
    should expose it as dependency edges rather than dropping them all.
    """
    store = _store(
        "the README explains the project and how to run it locally",
        "the build only works once the README's environment setup is done",
        "coding conventions live in CONTRIBUTING.md",
    )
    readme = _step("Read the README", ["c1"])
    readme["key"] = "s1"
    build = _step("Run the build", ["c2"])
    build["key"] = "s2"
    build["blocked_by"] = ["s1"]
    conventions = _step("Follow the conventions", ["c3"])
    conventions["key"] = "s3"
    conventions["blocked_by"] = ["s2"]

    outcome = assemble_phase(
        _llm(
            _payload(
                steps=[readme, build, conventions],
                questions=[
                    _question(key="q1", blocked_by=["s1"]),
                    _question(key="q2", blocked_by=["s3"]),
                ],
            )
        ),
        store,
        phase_title="Project Setup",
        phase_prompt=(
            "Guide a new member from reading the README to following the conventions."
        ),
        project_id=_PROJECT,
    )

    assert outcome.status == "assembled"
    by_key = {s.key: s for s in outcome.steps}
    # The chain survives: build is blocked by the README, conventions by the build.
    assert by_key["s1"].blocked_by == []
    assert by_key["s2"].blocked_by == ["s1"]
    assert by_key["s3"].blocked_by == ["s2"]
    # Knowledge checks are blocked by the step that teaches them.
    assert outcome.check_questions[0].blocked_by == ["s1"]
    assert outcome.check_questions[1].blocked_by == ["s3"]


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
