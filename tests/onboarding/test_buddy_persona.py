"""The mentor's persona is assembled per hire, not fixed.

The through-line: **the persona must never describe a capability this hire does not
have.** A mentor told about a tool it was not given will offer the hire something
impossible, and a mentor told to celebrate merges will say so to a Scrum Master.
"""

from onboarding.buddy_persona import build_persona
from onboarding.vocabulary import DEFAULT_VOCABULARY, Vocabulary

_ALL_TOOLS = (
    "search_docs",
    "get_arrival_steps",
    "get_learning_plan",
    "get_module",
    "get_my_metrics",
    "get_my_competencies",
    "get_suggested_tasks",
    "submit_verification",
    "claim_goal",
    "flag_to_pm",
)


def test_full_toolset_persona_mentions_every_mounted_tool() -> None:
    persona = build_persona(_ALL_TOOLS)

    for tool in _ALL_TOOLS:
        assert f"`{tool}`" in persona


def test_a_tool_that_is_not_mounted_is_never_mentioned() -> None:
    mounted = [t for t in _ALL_TOOLS if t != "get_module"]

    persona = build_persona(mounted)

    # Not softened to "if available" -- absent, so the model cannot try to call it.
    assert "`get_module`" not in persona
    assert "`get_learning_plan`" in persona


def test_without_an_escalation_tool_the_persona_offers_no_escalation() -> None:
    persona = build_persona([t for t in _ALL_TOOLS if t != "flag_to_pm"])

    assert "flag_to_pm" not in persona
    # The honesty instruction survives; only the offer of a route out goes.
    assert "rather than inventing an answer" in persona


def test_hire_state_tools_are_listed_only_when_mounted() -> None:
    persona = build_persona(["search_docs", "get_my_metrics"])

    assert "`get_my_metrics`" in persona
    assert "`get_my_competencies`" not in persona
    assert "`get_suggested_tasks`" not in persona


def test_no_hire_state_tools_means_no_hire_state_clause() -> None:
    persona = build_persona(["search_docs"])

    assert "hire-state tools" not in persona


def test_arrival_is_raised_before_anything_is_suggested() -> None:
    """The failure this initiative exists for: somebody who cannot clone the
    repository being handed a good first issue, and reading as calm rather than
    blocked because the stall detector watches contributions."""
    persona = build_persona(_ALL_TOOLS)

    assert "Before suggesting anything to work on" in persona
    assert persona.index("get_arrival_steps") < persona.index("get_suggested_tasks")


def test_arrival_steps_are_never_described_as_a_gate() -> None:
    """Ordering, not blocking. The model's own words are the one place the gate
    could come back with no code saying so."""
    persona = build_persona(_ALL_TOOLS)

    assert "never a reason they may not do something" in persona
    assert "not tell them to finish setup first" in persona


def test_the_arrival_clause_is_absent_without_the_tool() -> None:
    """A project that has authored no arrival list gets the persona that existed
    before A2 -- the backend only mounts the tool when a step actually applies."""
    persona = build_persona([t for t in _ALL_TOOLS if t != "get_arrival_steps"])

    assert "arrival" not in persona.lower()
    assert "Before suggesting anything to work on" not in persona


def test_default_vocabulary_is_the_engineering_wording() -> None:
    persona = build_persona(_ALL_TOOLS, DEFAULT_VOCABULARY)

    assert "Celebrate the changes and milestones" in persona


def test_a_tracks_vocabulary_replaces_the_engineering_wording() -> None:
    delivery = Vocabulary(
        contribution_noun="ceremony",
        contribution_noun_plural="ceremonies",
        contribution_verb_past="facilitated",
    )

    persona = build_persona(_ALL_TOOLS, delivery)

    assert "Celebrate the ceremonies and milestones" in persona
    assert "changes" not in persona


def test_the_persona_never_presumes_the_hire_writes_code() -> None:
    persona = build_persona(
        [t for t in _ALL_TOOLS if t != "get_my_metrics"],
        Vocabulary("plan", "plans", "published"),
    )

    # The regression this whole slice exists for: a hire whose work is never a pull
    # request must not meet a mentor whose standing instructions are about merging.
    lowered = persona.lower()
    assert "pull request" not in lowered
    assert "merge" not in lowered
    # The arrival clause is fixed text no track can reach, so engineering nouns in
    # it are invisible to the vocabulary that exists to keep them out. Writing
    # "somebody who cannot clone the repository" was the obvious first draft.
    assert "clone" not in lowered
    assert "repository" not in lowered
    assert "commit" not in lowered


def test_the_grounding_rule_survives_every_toolset() -> None:
    for mounted in ([], ["search_docs"], _ALL_TOOLS):
        persona = build_persona(mounted)

        # Non-negotiable regardless of what is mounted: grounding and the
        # test/fixture caveat are safety rules, not capabilities.
        assert "Ground every claim" in persona
        assert "test, fixture, or sample-data files" in persona
