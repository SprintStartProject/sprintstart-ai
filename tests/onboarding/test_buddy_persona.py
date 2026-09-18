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
    "get_my_metrics",
    "get_my_competencies",
    "get_suggested_tasks",
    "claim_goal",
    "flag_to_pm",
)


def test_full_toolset_persona_mentions_every_mounted_tool() -> None:
    persona = build_persona(_ALL_TOOLS)

    for tool in _ALL_TOOLS:
        assert f"`{tool}`" in persona


def test_a_tool_that_is_not_mounted_is_never_mentioned() -> None:
    mounted = [t for t in _ALL_TOOLS if t != "get_my_competencies"]

    persona = build_persona(mounted)

    # Not softened to "if available" -- absent, so the model cannot try to call it.
    assert "`get_my_competencies`" not in persona
    assert "`get_my_metrics`" in persona


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

    # A hire whose work is never a pull request must not meet a mentor whose standing
    # instructions are about merging.
    lowered = persona.lower()
    assert "pull request" not in lowered
    assert "merge" not in lowered
    # The arrival clause is fixed text the vocabulary cannot reach, so engineering
    # nouns in it are invisible to the thing that exists to keep them out. Writing
    # "somebody who cannot clone the repository" was the obvious first draft.
    assert "clone" not in lowered
    assert "repository" not in lowered
    assert "commit" not in lowered


def test_the_assessment_is_offered_never_administered() -> None:
    """The whole design of the feature, in the one place a model could undo it: a
    mentor told to place a hire will interview them."""
    persona = build_persona([*_ALL_TOOLS, "get_competencies_to_assess"])

    assert "Offer -- once, lightly --" in persona
    assert "If they would rather not, drop it" in persona
    assert "in this conversation" in persona


def test_a_placement_is_never_recorded_without_asking() -> None:
    persona = build_persona([*_ALL_TOOLS, "get_competencies_to_assess"])

    assert "Only after actually asking" in persona
    assert "their answer wins" in persona
    # A weak prior that accepted work outranks -- the rule the ledger enforces, said
    # in the words the hire will hear it in.
    assert "never call it a score, a result, or final" in persona


def test_a_hire_with_nothing_unplaced_is_never_offered_an_assessment() -> None:
    """The backend drops the read tool once everything has something behind it, so
    the offer stops rather than repeating every visit."""
    persona = build_persona(_ALL_TOOLS)

    assert "get_competencies_to_assess" not in persona
    assert "record_assessment" not in persona


def test_the_grounding_rule_survives_every_toolset() -> None:
    for mounted in ([], ["search_docs"], _ALL_TOOLS):
        persona = build_persona(mounted)

        # Non-negotiable regardless of what is mounted: grounding and the
        # test/fixture caveat are safety rules, not capabilities.
        assert "Ground every claim" in persona
        assert "test, fixture, or sample-data files" in persona


# --- Modes -----------------------------------------------------------------
#
# Two flags pick which persona is assembled. The through-line above still holds
# inside each one: what is not mounted is never mentioned. What these add is that a
# mode is not a softening of the default -- capabilities off must not sound able to
# act, and team mode must not sound like it is talking to the person it describes.

_TEAM_TOOLS = (
    "search_docs",
    "get_team_attention",
    "find_member",
    "get_member_progress",
    "open_area",
)


def test_capabilities_off_says_it_is_answering_from_the_material() -> None:
    persona = build_persona(["search_docs"], capabilities_enabled=False)

    assert "`search_docs` and nothing else" in persona
    assert "do not offer to record, claim, flag or change anything" in persona


def test_capabilities_off_keeps_grounding_and_the_fixture_caveat() -> None:
    """Safety rules are not capabilities, so no mode drops them."""
    persona = build_persona(["search_docs"], capabilities_enabled=False)

    assert "Ground every claim" in persona
    assert "test, fixture, or sample-data files" in persona


def test_capabilities_off_never_offers_an_escalation_or_a_claim() -> None:
    # Nothing is mounted with capabilities off, so an offer here is one the hire
    # cannot take up -- the refusal that follows reads as a fault in the product.
    persona = build_persona(_ALL_TOOLS, capabilities_enabled=False)

    assert "flag_to_pm" not in persona
    assert "claim_goal" not in persona
    assert "get_arrival_steps" not in persona


def test_team_mode_addresses_the_manager_not_a_hire() -> None:
    persona = build_persona(_TEAM_TOOLS, team_mode=True)

    assert "manager of one project" in persona
    assert "Never greet them as a new hire" in persona
    assert "the mentor who guides a new hire" not in persona


def test_team_mode_drops_every_hire_directed_clause() -> None:
    """Dropped by mode, not only by mounting: a manager must never be asked to
    settle where *they* are starting from."""
    persona = build_persona([*_TEAM_TOOLS, *_ALL_TOOLS], team_mode=True)

    assert "get_arrival_steps" not in persona
    assert "claim_goal" not in persona
    assert "get_competencies_to_assess" not in persona
    assert "record_assessment" not in persona
    assert "hire-state tools" not in persona


def test_team_mode_states_situations_as_facts_about_the_situation() -> None:
    persona = build_persona(_TEAM_TOOLS, team_mode=True)

    assert "never as a judgment of the person" in persona
    assert "the reviewer's move" in persona
    assert "never call anybody slow or behind" in persona


def test_team_mode_never_claims_to_have_made_a_change() -> None:
    persona = build_persona(_TEAM_TOOLS, team_mode=True)

    assert "You never make a change yourself" in persona
    assert "confirms it outside this conversation" in persona


def test_the_proposal_rule_holds_before_any_action_is_mounted() -> None:
    """Actions arrive only once an area is opened, and the rule that stops the model
    announcing a change must be there on the hop before that."""
    persona = build_persona(["search_docs", "get_team_attention"], team_mode=True)

    assert "You never make a change yourself" in persona


def test_team_mode_explains_areas_only_when_open_area_is_mounted() -> None:
    with_areas = build_persona(_TEAM_TOOLS, team_mode=True)
    without = build_persona(
        [t for t in _TEAM_TOOLS if t != "open_area"], team_mode=True
    )

    assert "`open_area`" in with_areas
    assert "available on your *next* step" in with_areas
    assert "open_area" not in without


def test_team_mode_lists_only_the_team_tools_that_are_mounted() -> None:
    persona = build_persona(
        ["search_docs", "get_team_attention"],
        team_mode=True,
    )

    assert "`get_team_attention`" in persona
    assert "`find_member`" not in persona
    assert "`get_member_progress`" not in persona


def test_team_mode_without_team_tools_describes_none_of_them() -> None:
    persona = build_persona(["search_docs"], team_mode=True)

    assert "the team tools" not in persona
    # The identity and the safety rules are all that is left, and they are enough.
    assert "manager of one project" in persona
    assert "Ground every claim" in persona


def test_team_mode_offers_no_escalation_to_a_pm() -> None:
    """`flag_to_pm` raises a question *to* a manager; the reader here is one."""
    persona = build_persona([*_TEAM_TOOLS, "flag_to_pm"], team_mode=True)

    assert "flag_to_pm" not in persona


def test_team_mode_with_capabilities_off_is_search_only_and_still_a_manager() -> None:
    persona = build_persona(_TEAM_TOOLS, team_mode=True, capabilities_enabled=False)

    assert "manager of one project" in persona
    assert "`search_docs` and nothing else" in persona
    assert "never as a judgment of the person" in persona
    assert "get_team_attention" not in persona


def test_the_defaults_are_todays_behaviour() -> None:
    assert build_persona(_ALL_TOOLS) == build_persona(
        _ALL_TOOLS, capabilities_enabled=True, team_mode=False
    )
    assert build_persona(_ALL_TOOLS, DEFAULT_VOCABULARY) == build_persona(_ALL_TOOLS)
