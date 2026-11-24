"""What an orientation run tells the people around it.

`test_orientation.py` proves which packet comes out. This module pins what the
watcher, the operator and the model are told on the way: the progress script of
every path, the outcome notes, the log lines, and the prompt itself.
"""

import logging
from datetime import UTC, datetime

import pytest

from onboarding.orientation import (
    _task_query,
    assemble_orientation,
    stream_orientation,
)
from onboarding.orientation_models import OrientationSource
from rag.types import Chunk
from tests.onboarding.golden import assert_matches_golden
from tests.onboarding.orientation_fixtures import PIDS, TITLE, payload, section
from tests.onboarding.pipeline_fixtures import (
    EMBED,
    collect,
    make_llm,
    make_store,
    script,
    unreadable_llm,
)
from tests.stubs.store import StubVectorStore

_RETRIEVING = [
    ("stage", "retrieving", "Searching the project: setting up"),
    ("stage", "retrieving", "Searching the project: finding the code"),
    ("stage", "retrieving", "Searching the project: making the change"),
    ("stage", "retrieving", "Searching the project: checking locally"),
    ("stage", "retrieving", "Searching the project: opening the PR"),
]
_GROUNDING = ("stage", "grounding", "Checking every section cites its source")
_NOT_ASSEMBLED = ("done", None, "No orientation could be assembled")


def _two_doc_store() -> StubVectorStore:
    return make_store(
        "run make dev to start the service locally",
        "pull requests need one review",
    )


def test_an_assembled_packet_streams_each_step_then_its_sections() -> None:
    llm = make_llm(
        payload(
            section("SET_UP", "Run it", ["c1"]),
            section("OPEN_THE_PR", "Review", ["c2"]),
        )
    )

    events, outcome = collect(
        stream_orientation(llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS)
    )

    assert script(events) == [
        *_RETRIEVING,
        ("stage", "generating", "Writing the packet from 2 source(s)"),
        _GROUNDING,
        ("item", None, "setting up: Run it"),
        ("item", None, "opening the PR: Review"),
        ("done", None, "Orientation ready"),
    ]
    assert {e["operation"] for e in events} == {"orientation"}
    assert outcome.notes == []


def test_a_dropped_section_is_announced_and_noted() -> None:
    llm = make_llm(
        payload(
            section("SET_UP", "Run it", ["c1"]),
            section("CHECK_LOCALLY", "Invented", ["nope"]),
        )
    )

    events, outcome = collect(
        stream_orientation(llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS)
    )

    assert script(events)[-3:] == [
        ("item", None, "setting up: Run it"),
        ("warning", None, "Dropped 1 section(s) with no source"),
        ("done", None, "Orientation ready"),
    ]
    assert outcome.notes == ["dropped 1 ungrounded or duplicate section(s)"]
    assert outcome.provenance is not None
    assert outcome.provenance.notes == outcome.notes


def test_unreadable_output_is_corrected_once_then_given_up() -> None:
    llm = unreadable_llm()
    calls = llm.calls

    events, outcome = collect(
        stream_orientation(llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS)
    )

    assert script(events) == [
        *_RETRIEVING,
        ("stage", "generating", "Writing the packet from 2 source(s)"),
        ("stage", "generating", "Correcting invalid generated JSON"),
        ("warning", None, "The generated packet could not be read"),
        _NOT_ASSEMBLED,
    ]
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert len(calls) == 2
    assert outcome.status == "skipped"
    assert outcome.chunks_retrieved == 2
    assert outcome.notes == [
        "invalid orientation output: no JSON object found in LLM output"
    ]


def test_a_packet_with_every_section_dropped_says_so() -> None:
    llm = make_llm(payload(section("SET_UP", "Invented", ["nope"])))

    events, outcome = collect(
        stream_orientation(llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS)
    )

    assert script(events)[-3:] == [
        _GROUNDING,
        ("warning", None, "Every section was dropped for lack of a source"),
        _NOT_ASSEMBLED,
    ]
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.notes == ["no grounded sections in the assembled packet"]
    assert outcome.sections_dropped == 1
    assert outcome.chunks_retrieved == 2


def test_an_empty_corpus_says_there_is_nothing_indexed() -> None:
    events, outcome = collect(
        stream_orientation(
            make_llm(payload()), StubVectorStore(), task_title=TITLE, project_ids=PIDS
        )
    )

    assert script(events) == [
        ("warning", None, "The project has no indexed material yet"),
        _NOT_ASSEMBLED,
    ]
    assert outcome.notes == ["corpus is empty"]


def test_a_task_nothing_matches_says_so_after_searching_every_step() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="zoo.md",
                text="zebra quokka",
                embedding=[0.0, 1.0] + [0.0] * 766,
                project_ids=("p1",),
            )
        ]
    )

    events, outcome = collect(
        stream_orientation(
            make_llm(payload()), store, task_title=TITLE, project_ids=PIDS
        )
    )

    assert script(events) == [
        *_RETRIEVING,
        ("warning", None, "Nothing in the project matched this task closely enough"),
        _NOT_ASSEMBLED,
    ]
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.notes == ["no grounding evidence retrieved for this task"]


def test_an_unchanged_corpus_ends_at_once_with_the_cached_packet() -> None:
    store = _two_doc_store()
    llm = make_llm(payload(section("SET_UP", "Run it", ["c1"])))
    first = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)
    assert first.provenance is not None

    events, outcome = collect(
        stream_orientation(
            llm,
            store,
            task_title=TITLE,
            project_ids=PIDS,
            last_fingerprint=first.provenance.corpus_fingerprint,
        )
    )

    assert script(events) == [
        ("done", None, "Nothing changed — the cached packet is current")
    ]
    assert outcome.notes == ["corpus unchanged since the cached packet"]


def test_collapsed_sources_are_noted() -> None:
    duplicate = "run make dev to start the reports service locally on port 8080"
    store = make_store(duplicate, duplicate + " .")
    llm = make_llm(payload(section("SET_UP", "Run it", ["c1"])))

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.notes == ["collapsed 1 redundant source chunk(s)"]


def test_provenance_names_the_model_and_a_utc_time() -> None:
    llm = make_llm(payload(section("SET_UP", "Run it", ["c1"])))

    outcome = assemble_orientation(
        llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS
    )

    assert outcome.provenance is not None
    assert outcome.provenance.model == "stub-model"
    assert outcome.provenance.generated_at is not None
    generated = datetime.fromisoformat(outcome.provenance.generated_at)
    assert generated.utcoffset() == UTC.utcoffset(None)


def test_every_dropped_section_counts_and_the_rest_are_still_read() -> None:
    body = "Run `make dev` to start the service locally on port 8080."
    llm = make_llm(
        payload(
            section("SET_UP", "Run it", ["c1"], body=body),
            section("DEPLOY_IT", "Unknown step", ["c1"]),
            section("SHIP_IT", "Unknown step", ["c1"]),
            section("SET_UP", "No body", ["c1"], body=" "),
            section("SET_UP", "No body", ["c1"], body=""),
            section("CHECK_LOCALLY", "Uncited", ["nope"]),
            section("CHECK_LOCALLY", "Uncited", []),
            section("SET_UP", "Restated", ["c1"], body=body + " Again."),
            section("SET_UP", "Restated", ["c1"], body=body + " Once more."),
            section("OPEN_THE_PR", "Review", ["c2"]),
        )
    )

    outcome = assemble_orientation(
        llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS
    )

    assert outcome.packet is not None
    assert [s.title for s in outcome.packet.sections] == ["Run it", "Review"]
    assert outcome.sections_dropped == 8


def test_the_same_text_under_two_steps_is_kept_under_both() -> None:
    body = "Run `make dev` to start the service locally on port 8080."
    llm = make_llm(
        payload(
            section("SET_UP", "Run it", ["c1"], body=body),
            section("CHECK_LOCALLY", "Run it again", ["c1"], body=body),
        )
    )

    outcome = assemble_orientation(
        llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS
    )

    assert outcome.packet is not None
    assert [s.step for s in outcome.packet.sections] == ["SET_UP", "CHECK_LOCALLY"]


def test_an_untitled_section_is_named_after_its_step() -> None:
    llm = make_llm(payload(section("set_up", " ", ["c1"], body="Run make dev.")))

    outcome = assemble_orientation(
        llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS
    )

    assert outcome.packet is not None
    assert [s.title for s in outcome.packet.sections] == ["Set Up"]


def test_each_dropped_section_is_logged_with_its_reason(
    caplog: pytest.LogCaptureFixture,
) -> None:
    body = "Run `make dev` to start the service locally on port 8080."
    llm = make_llm(
        payload(
            section("SET_UP", "Run it", ["c1"], body=body),
            section("DEPLOY_IT", "Ship", ["c1"]),
            section("CHECK_LOCALLY", "Invented", ["nope"]),
            section("SET_UP", "Restated", ["c1"], body=body + " Again."),
        )
    )

    with caplog.at_level(logging.INFO, logger="onboarding.orientation"):
        assemble_orientation(llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS)

    assert [r.getMessage() for r in caplog.records] == [
        "Dropped orientation section of unknown step 'DEPLOY_IT'",
        "Dropped ungrounded orientation section 'Invented'",
        "Dropped orientation section restating 'Restated'",
    ]


def test_each_failed_attempt_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="onboarding.orientation"):
        assemble_orientation(
            unreadable_llm(), _two_doc_store(), task_title=TITLE, project_ids=PIDS
        )

    reason = "invalid orientation output: no JSON object found in LLM output"
    assert [r.getMessage() for r in caplog.records] == [
        f"Orientation assembly attempt 1 failed for task {TITLE!r}: {reason}",
        f"Orientation assembly attempt 2 failed for task {TITLE!r}: {reason}",
    ]


def test_sources_are_listed_once_each_in_first_use_order() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id=f"a{i}",
                filename=name,
                text=text,
                embedding=EMBED,
                project_ids=("p1",),
                source_url=f"https://example.test/{name}",
                artifact_type=kind,
            )
            for i, (name, text, kind) in enumerate(
                [
                    ("README.md", "run make dev to start the service locally", "DOC"),
                    ("ci.yml", "the pipeline runs lint and tests", "CONFIG"),
                ],
                start=1,
            )
        ]
    )
    llm = make_llm(
        payload(
            section("SET_UP", "Run it", ["c1"]),
            section("CHECK_LOCALLY", "Checks", ["c1", "c2"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.packet is not None
    assert outcome.packet.sources == [
        OrientationSource(
            filename="README.md",
            source_url="https://example.test/README.md",
            artifact_type="DOC",
        ),
        OrientationSource(
            filename="ci.yml",
            source_url="https://example.test/ci.yml",
            artifact_type="CONFIG",
        ),
    ]


def test_the_task_query_joins_every_part_with_spaces() -> None:
    query = _task_query("Fix it", "", ["bug", "api"], ["src/a.py", "src/b.py"])

    assert query == "Fix it bug api src/a.py src/b.py"


def test_the_task_query_is_cut_to_its_cap() -> None:
    assert len(_task_query("T", "x" * 9000, [], [])) == 4000


def test_every_step_searches_for_the_task_and_what_the_step_needs() -> None:
    searched: list[str] = []

    def embed(text: str) -> list[float]:
        searched.append(text)
        return EMBED

    llm = make_llm(payload(section("SET_UP", "Run it", ["c1"])))
    llm.embed_fn = embed

    assemble_orientation(
        llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS, labels=["api"]
    )

    assert len(searched) == 5
    assert all(q.startswith(f"{TITLE} api ") for q in searched)
    assert searched[0].endswith("build and run the project locally configuration")


def test_material_from_other_projects_and_tests_never_reaches_the_prompt() -> None:
    store = make_store("run make dev to start the service locally")
    store.add(
        [
            Chunk(
                id="other",
                artifact_id="a-other",
                filename="other.md",
                text="run make dev for the other project",
                embedding=EMBED,
                project_ids=("p2",),
            ),
            Chunk(
                id="spec",
                artifact_id="a-spec",
                filename="test_dev.py",
                text="run make dev inside the test fixture",
                embedding=EMBED,
                project_ids=("p1",),
                source_role="test",
            ),
        ]
    )
    llm = make_llm(payload(section("SET_UP", "Run it", ["c1"])))
    calls = llm.calls

    assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    evidence = calls[0][1]["content"]
    assert "[c1]" in evidence
    assert "[other]" not in evidence
    assert "[spec]" not in evidence


def test_a_corpus_with_only_other_projects_counts_as_empty() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="other",
                artifact_id="a-other",
                filename="other.md",
                text="run make dev for the other project",
                embedding=EMBED,
                project_ids=("p2",),
            )
        ]
    )

    outcome = assemble_orientation(
        make_llm(payload()), store, task_title=TITLE, project_ids=PIDS
    )

    assert outcome.notes == ["corpus is empty"]


def test_the_prompt_for_a_task_with_labels_and_paths() -> None:
    llm = make_llm(payload(section("SET_UP", "Run it", ["c1"])))
    calls = llm.calls

    assemble_orientation(
        llm,
        _two_doc_store(),
        task_title=TITLE,
        project_ids=PIDS,
        labels=["good first issue", "api"],
        touched_paths=["src/api/reports.py", "src/api/cache.py"],
    )

    assert_matches_golden("orientation_labelled_task", calls[0])


def test_the_prompt_for_a_bare_task() -> None:
    llm = make_llm(payload(section("SET_UP", "Run it", ["c1"])))
    calls = llm.calls

    collect(
        stream_orientation(llm, _two_doc_store(), task_title=TITLE, project_ids=PIDS)
    )

    assert_matches_golden("orientation_bare_task", calls[0])


def test_the_correction_prompt_carries_the_broken_reply_and_the_error() -> None:
    llm = unreadable_llm()
    calls = llm.calls

    assemble_orientation(
        llm,
        _two_doc_store(),
        task_title=TITLE,
        project_ids=PIDS,
        task_body="The header is computed once at boot.",
    )

    assert_matches_golden("orientation_correction", calls[1])
