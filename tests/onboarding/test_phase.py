import json

from llm.base import Message
from onboarding.phase import assemble_phase, stream_phase
from rag.types import RetrievalFilters
from tests.onboarding.phase_fixtures import PROJECT, payload, question, step
from tests.onboarding.pipeline_fixtures import EMBED, collect, make_llm, make_store
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def test_assembles_grounded_steps_and_questions() -> None:
    store = make_store(
        "the README explains the project and how to run it locally",
        "coding conventions require a PR to pass review",
    )
    llm = make_llm(
        payload(
            steps=[step("Read the README", ["c1"])],
            questions=[question()],
        )
    )

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview for a new member.",
        project_id=PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(outcome.steps) == 1
    first = outcome.steps[0]
    assert first.title == "Read the README"
    assert first.estimated_minutes == 10
    assert len(outcome.check_questions) == 1
    assert outcome.check_questions[0].question == "Explain SHORT_TEXT"


def test_accepts_literal_control_characters_in_generated_strings() -> None:
    """Model output with an unescaped newline remains usable phase content."""
    generated = payload(steps=[step("Read the README", ["c1"])])
    generated["steps"][0]["description"] = "Read first\nThen run the application"  # type: ignore[index]
    raw = json.dumps(generated).replace("\\n", "\n")
    llm = StubLLMClient(generate_response=raw, embedding=EMBED)

    outcome = assemble_phase(
        llm,
        make_store("the README explains the project and how to run it locally"),
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
    )

    assert outcome.status == "assembled"
    assert outcome.steps[0].description == "Read first\nThen run the application"


def test_treats_null_short_text_options_as_an_empty_list() -> None:
    """A model's explicit null for inapplicable options does not skip the phase."""
    outcome = assemble_phase(
        make_llm(
            payload(
                steps=[step("Read the README", ["c1"])],
                questions=[question(options=None)],
            )
        ),
        make_store("the README explains the project and how to run it locally"),
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(outcome.check_questions) == 1
    assert outcome.check_questions[0].options == []


def test_accepts_tasks_as_plain_strings() -> None:
    """A compact model task remains usable instead of invalidating the phase."""
    compact = step("Review the repository", ["c1"])
    compact["tasks"] = ["Open README.md", "Search for meeting references"]

    outcome = assemble_phase(
        make_llm(payload(steps=[compact])),
        make_store("the README documents the repository structure"),
        phase_title="Architecture",
        phase_prompt="Review the repository.",
        project_id=PROJECT,
    )

    assert outcome.status == "assembled"
    assert [task.title for task in outcome.steps[0].tasks] == [
        "Open README.md",
        "Search for meeting references",
    ]
    assert all(task.description == "" for task in outcome.steps[0].tasks)


def test_retries_once_when_generated_json_is_invalid() -> None:
    """A transient syntax error gets one correction pass before giving up."""
    valid = json.dumps(payload(steps=[step("Read the README", ["c1"])]))

    class _CorrectingLLM(StubLLMClient):
        def __init__(self) -> None:
            super().__init__(embedding=EMBED)
            self.calls: list[list[Message]] = []

        def generate(
            self, messages: list[Message], *, temperature: float | None = None
        ) -> str:
            self.calls.append(messages)
            return '{"steps": [' if len(self.calls) == 1 else valid

    llm = _CorrectingLLM()
    outcome = assemble_phase(
        llm,
        make_store("the README explains the project"),
        phase_title="Architecture",
        phase_prompt="Review the repository.",
        project_id=PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(llm.calls) == 2
    assert "could not be validated" in llm.calls[1][-1]["content"]


def test_drops_an_ungrounded_step_but_keeps_the_phase() -> None:
    """A step with no source does not ship -- that is the phase's grounding rule."""
    store = make_store("the README explains the project and how to run it locally")
    llm = make_llm(
        payload(
            steps=[
                step("Read the README", ["c1"]),
                step("Invented step", ["nope"]),
            ]
        )
    )

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
    )

    assert outcome.status == "assembled"
    assert [s.title for s in outcome.steps] == ["Read the README"]
    assert outcome.steps_dropped == 1


def test_resolves_dependency_edges_between_grounded_items() -> None:
    """blocked_by keys survive only when they name an earlier kept item."""
    store = make_store(
        "the README explains the project and how to run it locally",
        "a build requires the environment to be configured first",
    )
    readme = step("Read the README", ["c1"])
    readme["key"] = "s1"
    setup = step("Set up the environment", ["c2"])
    setup["key"] = "s2"
    setup["blocked_by"] = ["s1", "missing"]
    ghost = step("Ungrounded step", ["nope"])
    ghost["key"] = "s3"
    later = step("Depends on a dropped step", ["c1"])
    later["key"] = "s4"
    later["blocked_by"] = ["s3", "s2"]
    selfie = step("Self-referencing step", ["c1"])
    selfie["key"] = "s5"
    selfie["blocked_by"] = ["s5"]

    outcome = assemble_phase(
        make_llm(
            payload(
                steps=[readme, setup, ghost, later, selfie],
                questions=[question(key="q1", blocked_by=["s1", "unknown"])],
            )
        ),
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
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
    store = make_store(
        "the README explains the project and how to run it locally",
        "the build only works once the README's environment setup is done",
        "coding conventions live in CONTRIBUTING.md",
    )
    readme = step("Read the README", ["c1"])
    readme["key"] = "s1"
    build = step("Run the build", ["c2"])
    build["key"] = "s2"
    build["blocked_by"] = ["s1"]
    conventions = step("Follow the conventions", ["c3"])
    conventions["key"] = "s3"
    conventions["blocked_by"] = ["s2"]

    outcome = assemble_phase(
        make_llm(
            payload(
                steps=[readme, build, conventions],
                questions=[
                    question(key="q1", blocked_by=["s1"]),
                    question(key="q2", blocked_by=["s3"]),
                ],
            )
        ),
        store,
        phase_title="Project Setup",
        phase_prompt=(
            "Guide a new member from reading the README to following the conventions."
        ),
        project_id=PROJECT,
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
    store = make_store("project security conventions are documented in CONTRIBUTING.md")
    llm = make_llm(
        payload(
            steps=[step("Follow the conventions", ["c1"])],
            questions=[
                question(),
                # All-correct MULTIPLE_CHOICE discriminates nothing: dropped.
                question(
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
        project_id=PROJECT,
    )

    assert outcome.status == "assembled"
    assert len(outcome.check_questions) == 1
    assert outcome.questions_dropped == 1


def test_skipped_when_the_corpus_is_empty() -> None:
    store = StubVectorStore()
    llm = make_llm({})

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
    )

    assert outcome.status == "skipped"
    assert outcome.steps == []


def test_skipped_when_every_step_is_ungrounded() -> None:
    store = make_store("the README explains the project")
    llm = make_llm(payload(steps=[step("Invented step", ["missing"])]))

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
    )

    assert outcome.status == "skipped"
    assert outcome.steps == []
    assert outcome.steps_dropped == 1


def test_unchanged_when_the_corpus_fingerprint_matches() -> None:
    """Idempotency: an unchanged corpus short-circuits retrieval and generation."""
    store = make_store("the README explains the project and how to run it locally")
    first = assemble_phase(
        make_llm(payload(steps=[step("Read the README", ["c1"])])),
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
    )
    assert first.status == "assembled"
    fingerprint = first.provenance.corpus_fingerprint
    assert fingerprint is not None

    second = assemble_phase(
        make_llm(json.dumps({"steps": []})),
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        last_fingerprint=fingerprint,
    )

    assert second.status == "unchanged"


def test_stream_emits_item_per_grounded_step_then_terminal_done() -> None:
    store = make_store("the README explains the project and how to run it locally")
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))

    events, outcome = collect(
        stream_phase(
            llm,
            store,
            phase_title="Project Overview",
            phase_prompt="Generate an overview.",
            project_id=PROJECT,
        )
    )

    assert outcome.status == "assembled"
    types = [e["type"] for e in events]
    assert types[0] == "stage"
    assert "item" in types
    assert types[-1] == "done"
    item = next(e for e in events if e["type"] == "item")
    assert item["item"]["title"] == "Read the README"  # type: ignore[typeddict-item]


def test_phase_query_and_prompt_with_industry() -> None:
    from onboarding.phase import _build_prompt, _phase_query

    query = _phase_query(
        title="Architecture",
        description="Core services",
        prompt="Explain backend structure",
        industry="Fintech / Banking",
    )
    expected_query = (
        "Industry: Fintech / Banking Architecture Core services "
        "Explain backend structure"
    )
    assert query == expected_query

    store = make_store("evidence text")
    messages = _build_prompt(
        title="Architecture",
        description="Core services",
        prompt="Explain backend structure",
        chunks=store.query(
            EMBED,
            top_k=1,
            min_score=0.0,
            filters=RetrievalFilters(project_id=PROJECT),
        ),
        industry="Fintech / Banking",
    )
    assert len(messages) == 2
    assert "Project industry/domain: Fintech / Banking" in messages[1]["content"]


def test_phase_query_and_prompt_backward_compat_when_industry_none() -> None:
    from onboarding.phase import _build_prompt, _phase_query

    query = _phase_query(
        title="Architecture",
        description="Core services",
        prompt="Explain backend structure",
        industry=None,
    )
    assert query == "Architecture Core services Explain backend structure"

    store = make_store("evidence text")
    messages = _build_prompt(
        title="Architecture",
        description="Core services",
        prompt="Explain backend structure",
        chunks=store.query(
            EMBED,
            top_k=1,
            min_score=0.0,
            filters=RetrievalFilters(project_id=PROJECT),
        ),
        industry=None,
    )
    assert len(messages) == 2
    assert "Project industry/domain:" not in messages[1]["content"]


def test_assemble_phase_with_industry() -> None:
    store = make_store("the README explains the project and how to run it locally")
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))

    outcome = assemble_phase(
        llm,
        store,
        phase_title="Project Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        industry="Healthcare",
    )

    assert outcome.status == "assembled"
    assert outcome.steps[0].title == "Read the README"


def test_fingerprint_includes_industry() -> None:
    store = make_store("the README explains the project and how to run it locally")
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))

    res_none = assemble_phase(
        llm,
        store,
        phase_title="Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        industry=None,
    )
    res_blank = assemble_phase(
        llm,
        store,
        phase_title="Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        industry="   ",
    )
    res_fintech = assemble_phase(
        llm,
        store,
        phase_title="Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        industry="Fintech",
    )
    res_healthcare = assemble_phase(
        llm,
        store,
        phase_title="Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        industry="Healthcare",
    )

    fp_none = res_none.provenance.corpus_fingerprint
    fp_blank = res_blank.provenance.corpus_fingerprint
    fp_fintech = res_fintech.provenance.corpus_fingerprint
    fp_healthcare = res_healthcare.provenance.corpus_fingerprint

    assert fp_none is not None
    assert fp_none == fp_blank
    assert fp_fintech != fp_none
    assert fp_healthcare != fp_none
    assert fp_fintech != fp_healthcare


def test_changing_industry_invalidates_fingerprint_cache() -> None:
    store = make_store("the README explains the project and how to run it locally")
    llm = make_llm(payload(steps=[step("Read the README", ["c1"])]))

    first = assemble_phase(
        llm,
        store,
        phase_title="Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        industry="Fintech",
    )
    assert first.status == "assembled"
    fp_fintech = first.provenance.corpus_fingerprint
    assert fp_fintech is not None

    # Same industry + same last_fingerprint -> unchanged
    same_industry = assemble_phase(
        make_llm(json.dumps({"steps": []})),
        store,
        phase_title="Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        last_fingerprint=fp_fintech,
        industry="Fintech",
    )
    assert same_industry.status == "unchanged"

    # Changed industry + old last_fingerprint -> re-assembles (not unchanged)
    changed_industry = assemble_phase(
        llm,
        store,
        phase_title="Overview",
        phase_prompt="Generate an overview.",
        project_id=PROJECT,
        last_fingerprint=fp_fintech,
        industry="Healthcare",
    )
    assert changed_industry.status == "assembled"
    assert changed_industry.provenance.corpus_fingerprint != fp_fintech
