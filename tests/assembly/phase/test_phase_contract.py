"""What a phase assembly tells the people around it.

`test_phase.py` proves which phase comes out. This module pins what the watcher,
the operator and the model are told on the way: the progress script of every
path, the outcome notes, the log lines, how each generated item is cleaned, and
the prompt itself.
"""

import logging
from datetime import UTC, datetime

import pytest

from assembly.fingerprint import corpus_fingerprint
from assembly.models import CheckOption, CheckQuestion
from assembly.phase.job import assemble_phase, stream_phase
from assembly.phase.models import PhaseOutcome, PhaseResource, PhaseTask
from assembly.progress import ProgressEvent
from corpus.types import Chunk
from tests.assembly.phase.phase_fixtures import PROJECT, payload, question, step
from tests.assembly.pipeline_fixtures import (
    EMBED,
    collect,
    make_llm,
    make_store,
    script,
    unreadable_llm,
)
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore
from tests.support.golden import assert_matches_golden

_TITLE = "Architecture"
_PROMPT = "Walk the newcomer through the repository layout."
_RETRIEVING = ("stage", "retrieving", "Searching the project for: Architecture")
_GROUNDING = ("stage", "grounding", "Checking every step cites its source")
_NOT_ASSEMBLED = ("done", None, "No phase content could be assembled")
_LONG_QUESTION = (
    "Why does the reports module compute its cache header once at boot "
    "rather than per request?"
)


def _two_doc_store() -> StubVectorStore:
    return make_store(
        "the README explains the repository layout",
        "the reports module lives in src/api/reports.py",
    )


def _assemble(llm: StubLLMClient, store: StubVectorStore | None = None) -> PhaseOutcome:
    return assemble_phase(
        llm,
        store or _two_doc_store(),
        phase_title=_TITLE,
        phase_prompt=_PROMPT,
        project_id=PROJECT,
    )


def _stream(
    llm: StubLLMClient, store: StubVectorStore | None = None
) -> tuple[list[ProgressEvent], PhaseOutcome]:
    return collect(
        stream_phase(
            llm,
            store or _two_doc_store(),
            phase_title=_TITLE,
            phase_prompt=_PROMPT,
            project_id=PROJECT,
        )
    )


def test_an_assembled_phase_streams_its_steps_then_its_questions() -> None:
    llm = make_llm(
        payload(
            steps=[step("Read the README", ["c1"])],
            questions=[question(question=_LONG_QUESTION)],
        )
    )

    events, outcome = _stream(llm)

    assert script(events) == [
        _RETRIEVING,
        ("stage", "generating", "Writing the phase from 2 source(s)"),
        _GROUNDING,
        ("item", None, "Step: Read the README"),
        ("item", None, f"Check question: {_LONG_QUESTION[:60]}"),
        ("done", None, "Phase content ready"),
    ]
    assert {e["operation"] for e in events} == {"phase"}
    items = [e["item"] for e in events if e["type"] == "item"]
    assert items == [
        outcome.steps[0].model_dump(mode="json"),
        outcome.check_questions[0].model_dump(mode="json"),
    ]
    assert outcome.notes == []


def test_dropped_steps_and_questions_are_announced_and_noted() -> None:
    llm = make_llm(
        payload(
            steps=[step("Read the README", ["c1"]), step("Invented", ["nope"])],
            questions=[question(), question(correct_answer=" ")],
        )
    )

    events, outcome = _stream(llm)

    assert script(events)[-3:] == [
        ("warning", None, "Dropped 1 ungrounded or duplicate step(s)"),
        ("warning", None, "Dropped 1 invalid question(s)"),
        ("done", None, "Phase content ready"),
    ]
    assert outcome.notes == [
        "dropped 1 ungrounded or duplicate step(s)",
        "dropped 1 invalid question(s)",
    ]
    assert outcome.provenance is not None
    assert outcome.provenance.notes == outcome.notes


def test_unreadable_output_is_corrected_once_then_given_up() -> None:
    duplicate = "the README explains the repository layout in detail"
    llm = unreadable_llm()
    calls = llm.calls

    events, outcome = _stream(llm, make_store(duplicate, duplicate + " ."))

    assert script(events) == [
        _RETRIEVING,
        ("stage", "generating", "Writing the phase from 1 source(s)"),
        ("stage", "generating", "Correcting invalid generated JSON"),
        ("warning", None, "The generated content could not be read"),
        _NOT_ASSEMBLED,
    ]
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert len(calls) == 2
    assert outcome.status == "skipped"
    assert (outcome.chunks_retrieved, outcome.chunks_collapsed) == (1, 1)
    assert outcome.notes == ["invalid phase output: no JSON object found in LLM output"]


def test_a_phase_with_nothing_grounded_says_so() -> None:
    llm = make_llm(
        payload(steps=[step("Invented", ["nope"])], questions=[question(question="")])
    )

    events, outcome = _stream(llm)

    assert script(events)[-5:] == [
        _GROUNDING,
        ("warning", None, "Dropped 1 ungrounded or duplicate step(s)"),
        ("warning", None, "Dropped 1 invalid question(s)"),
        ("warning", None, "Nothing usable survived grounding"),
        _NOT_ASSEMBLED,
    ]
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.notes == ["no grounded steps or valid questions assembled"]
    assert (outcome.steps_dropped, outcome.questions_dropped) == (1, 1)
    assert (outcome.chunks_retrieved, outcome.chunks_collapsed) == (2, 0)


def test_an_empty_corpus_says_there_is_nothing_indexed() -> None:
    events, outcome = _stream(make_llm(payload()), StubVectorStore())

    assert script(events) == [
        ("warning", None, "The project has no indexed material yet"),
        _NOT_ASSEMBLED,
    ]
    assert outcome.notes == ["corpus is empty"]


def test_a_phase_nothing_matches_says_so() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="zoo.md",
                text="zebra quokka",
                embedding=[0.0, 1.0] + [0.0] * 766,
                project_ids=(PROJECT,),
            )
        ]
    )

    events, outcome = _stream(make_llm(payload()), store)

    assert script(events) == [
        _RETRIEVING,
        ("warning", None, "Nothing in the project matched this phase"),
        _NOT_ASSEMBLED,
    ]
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.status == "skipped"
    assert outcome.notes == ["no grounding evidence retrieved for this phase"]


def test_an_unchanged_corpus_ends_at_once() -> None:
    store = _two_doc_store()
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))
    first = _assemble(llm, store)
    assert first.provenance is not None

    events, outcome = collect(
        stream_phase(
            llm,
            store,
            phase_title=_TITLE,
            phase_prompt=_PROMPT,
            project_id=PROJECT,
            last_fingerprint=first.provenance.corpus_fingerprint,
        )
    )

    assert script(events) == [
        ("done", None, "Nothing changed — the cached phase content is current")
    ]
    assert outcome.notes == ["corpus unchanged since the last assembly"]


def test_collapsed_sources_are_noted() -> None:
    duplicate = "the README explains the repository layout in detail"
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))

    outcome = _assemble(llm, make_store(duplicate, duplicate + " ."))

    assert outcome.notes == ["collapsed 1 redundant source chunk(s)"]


def test_provenance_names_the_model_a_utc_time_and_the_plain_corpus() -> None:
    store = _two_doc_store()
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))

    outcome = _assemble(llm, store)

    assert outcome.provenance is not None
    assert outcome.provenance.model == "stub-model"
    assert outcome.provenance.generated_at is not None
    generated = datetime.fromisoformat(outcome.provenance.generated_at)
    assert generated.utcoffset() == UTC.utcoffset(None)
    assert outcome.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_generation_is_sampled_deterministically() -> None:
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))

    _assemble(llm)

    assert llm.temperatures == [0.0]


def test_every_dropped_step_counts_and_the_rest_are_still_read() -> None:
    llm = make_llm(
        payload(
            steps=[
                step("Read the README", ["c1"]),
                step(" ", ["c1"]),
                step("", ["c1"]),
                step("Invented", ["nope"]),
                step("Also invented", []),
                step("Read the README", ["c1"]),
                step("Read the README", ["c2"]),
                step("Find the reports module", ["c2"]),
            ]
        )
    )

    outcome = _assemble(llm)

    assert [s.title for s in outcome.steps] == [
        "Read the README",
        "Find the reports module",
    ]
    assert outcome.steps_dropped == 6


def test_a_step_repeating_only_an_earlier_title_is_dropped() -> None:
    again = step("Read the README", ["c2"])
    again["description"] = "Open src/api/reports.py and follow the cache header."

    outcome = _assemble(
        make_llm(payload(steps=[step("Read the README", ["c1"]), again]))
    )

    assert outcome.steps_dropped == 1


def test_a_step_keeps_only_its_usable_tasks_and_resources() -> None:
    generated = step("Read the README", ["c1"])
    generated["tasks"] = [
        {"title": "Open README.md", "description": "It is in the root."},
        {"title": " ", "description": "untitled"},
    ]
    generated["resources"] = [
        {"title": "README", "url": "https://example.test/README.md"},
        {"title": "", "url": "https://example.test/untitled"},
        {"title": "No link", "url": " "},
    ]

    outcome = _assemble(make_llm(payload(steps=[generated])))

    kept = outcome.steps[0]
    assert kept.tasks == [
        PhaseTask(title="Open README.md", description="It is in the root.")
    ]
    assert kept.resources == [
        PhaseResource(title="README", url="https://example.test/README.md")
    ]


@pytest.mark.parametrize(
    ("estimate", "kept"), [(1, 1), (25, 25), (0, None), (-5, None), (None, None)]
)
def test_only_a_positive_time_estimate_is_kept(
    estimate: int | None, kept: int | None
) -> None:
    generated = step("Read the README", ["c1"])
    generated["estimated_minutes"] = estimate

    outcome = _assemble(make_llm(payload(steps=[generated])))

    assert outcome.steps[0].estimated_minutes == kept


def test_valid_questions_are_kept_whole_and_numbered_in_order() -> None:
    llm = make_llm(
        payload(
            steps=[step("Read the README", ["c1"])],
            questions=[
                question(
                    q_type="MULTIPLE_CHOICE",
                    question="Where does the reports module live?",
                    explanation="The README says so.",
                    key=" q1 ",
                    blocked_by=[" "],
                    options=[
                        {"label": " ", "correct": False},
                        {"label": "src/api", "correct": True},
                        {"label": "src/web", "correct": False},
                    ],
                ),
                question(
                    question="Why is the header computed once?",
                    explanation="Boot-time config.",
                    correct_answer="Because it never changes at runtime",
                ),
            ],
        )
    )

    outcome = _assemble(llm)

    assert outcome.check_questions == [
        CheckQuestion(
            position=0,
            type="MULTIPLE_CHOICE",
            question="Where does the reports module live?",
            explanation="The README says so.",
            options=[
                CheckOption(position=0, label="src/api", correct=True),
                CheckOption(position=1, label="src/web", correct=False),
            ],
            key="q1",
        ),
        CheckQuestion(
            position=1,
            type="SHORT_TEXT",
            question="Why is the header computed once?",
            explanation="Boot-time config.",
            correct_answer="Because it never changes at runtime",
        ),
    ]


@pytest.mark.parametrize(
    "options",
    [
        [{"label": "Only", "correct": True}],
        [{"label": "A", "correct": False}, {"label": "B", "correct": False}],
        [{"label": "A", "correct": True}, {"label": "B", "correct": True}],
    ],
    ids=["one option", "none correct", "all correct"],
)
def test_a_multiple_choice_question_that_cannot_discriminate_is_dropped(
    options: list[dict[str, object]],
) -> None:
    llm = make_llm(
        payload(
            steps=[step("Read the README", ["c1"])],
            questions=[question(q_type="MULTIPLE_CHOICE", options=options)],
        )
    )

    outcome = _assemble(llm)

    assert outcome.check_questions == []
    assert outcome.questions_dropped == 1


def test_a_question_of_an_unknown_type_is_dropped() -> None:
    llm = make_llm(
        payload(
            steps=[step("Read the README", ["c1"])],
            questions=[question(q_type="multiple_choice")],
        )
    )

    assert _assemble(llm).questions_dropped == 1


def test_each_dropped_step_is_logged_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    llm = make_llm(
        payload(
            steps=[
                step("Read the README", ["c1"]),
                step("Invented", ["nope"]),
                step("Read the README", ["c2"]),
            ]
        )
    )

    with caplog.at_level(logging.INFO, logger="assembly.phase.draft"):
        _assemble(llm)

    assert [r.getMessage() for r in caplog.records] == [
        "Dropped ungrounded phase step 'Invented'",
        "Dropped phase step restating 'Read the README'",
    ]


def test_each_failed_attempt_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="assembly.phase.job"):
        _assemble(unreadable_llm())

    reason = "invalid phase output: no JSON object found in LLM output"
    assert [r.getMessage() for r in caplog.records] == [
        f"Phase assembly attempt 1 failed for {_TITLE!r}: {reason}",
        f"Phase assembly attempt 2 failed for {_TITLE!r}: {reason}",
    ]


def test_the_industry_reaches_the_search() -> None:
    searched: list[str] = []

    def embed(text: str) -> list[float]:
        searched.append(text)
        return EMBED

    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))
    llm.embed_fn = embed

    assemble_phase(
        llm,
        _two_doc_store(),
        phase_title=_TITLE,
        phase_prompt=_PROMPT,
        project_id=PROJECT,
        industry=" Fintech ",
    )

    assert searched == [f"Industry: Fintech {_TITLE} {_PROMPT}"]


def test_material_from_other_projects_and_tests_never_reaches_the_prompt() -> None:
    store = make_store("the README explains the repository layout")
    store.add(
        [
            Chunk(
                id="other",
                artifact_id="a-other",
                filename="other.md",
                text="the README of the other project explains its layout",
                embedding=EMBED,
                project_ids=("p2",),
            ),
            Chunk(
                id="spec",
                artifact_id="a-spec",
                filename="test_layout.py",
                text="the README layout fixture for tests",
                embedding=EMBED,
                project_ids=(PROJECT,),
                source_role="test",
            ),
        ]
    )
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))
    calls = llm.calls

    _assemble(llm, store)

    evidence = calls[0][1]["content"]
    assert "[c1]" in evidence
    assert "[other]" not in evidence
    assert "[spec]" not in evidence


def test_the_prompt_for_a_described_phase_in_an_industry() -> None:
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))
    calls = llm.calls

    assemble_phase(
        llm,
        _two_doc_store(),
        phase_title=_TITLE,
        phase_description="How the code is laid out.",
        phase_prompt=_PROMPT,
        project_id=PROJECT,
        industry="Fintech",
    )

    assert_matches_golden("phase_described_in_industry", calls[0])


def test_the_prompt_for_a_bare_phase() -> None:
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))
    calls = llm.calls

    _stream(llm)

    assert_matches_golden("phase_bare", calls[0])


def test_a_phase_without_a_prompt_says_so_to_the_model() -> None:
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))
    calls = llm.calls

    assemble_phase(
        llm, _two_doc_store(), phase_title=_TITLE, phase_prompt="", project_id=PROJECT
    )

    assert "Author's prompt:\n(none)\n" in calls[0][1]["content"]


def test_the_correction_prompt_carries_the_broken_reply_and_the_error() -> None:
    llm = unreadable_llm()
    calls = llm.calls

    _assemble(llm)

    assert_matches_golden("phase_correction", calls[1])
