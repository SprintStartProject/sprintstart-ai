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


def test_a_hire_who_asks_to_flag_something_is_offered_it() -> None:
    """'Last resort' is the mentor's bar for flagging unasked, never the hire's.

    Read as the only rule, it had the mentor refuse an explicit "flag this to my PM"
    as not being a PM matter.
    """
    persona = build_persona(_ALL_TOOLS)

    assert "When they ask you to flag" in persona
    assert "never decide for them that it is not a PM matter" in persona


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


def test_asking_what_to_work_on_is_routed_to_the_suggestion_tool() -> None:
    """The bug this clause exists for: the mentor answered the question without ever
    naming a task, because the hire-state list sorts tools by subject and "what
    should I work on" is not cleanly the hire's own progress."""
    persona = build_persona(_ALL_TOOLS)

    assert "When the hire asks what to work on" in persona
    assert "call `get_suggested_tasks` and present what it returns" in persona


def test_starter_work_is_never_answered_from_the_corpus() -> None:
    """Starter Work is a product noun as well as this hire's queue, and
    `search_docs` is described as covering how the product works -- so without this
    the model explains the feature, accurately, and names no task."""
    persona = build_persona(_ALL_TOOLS)

    assert "naming Starter Work" in persona
    assert "never answer this from `search_docs`" in persona
    assert "from the conversation summary" in persona


def test_an_empty_ranking_is_reported_rather_than_filled_in() -> None:
    persona = build_persona(_ALL_TOOLS)

    assert "nothing to suggest" in persona
    assert "A task you assembled yourself is not one anybody has agreed to" in persona


def test_the_arrival_read_does_not_stand_in_for_the_suggestion() -> None:
    """Arrival pauses the turn for the backend to run it, and the hop that resumes
    has already said something helpful. Both tools belong in the one reply."""
    persona = build_persona(_ALL_TOOLS)

    assert "does not stand in for that call" in persona
    assert "Both belong in the same reply" in persona


def test_without_the_arrival_tool_nothing_is_said_about_reading_it_first() -> None:
    persona = build_persona([t for t in _ALL_TOOLS if t != "get_arrival_steps"])

    assert "When the hire asks what to work on" in persona
    assert "does not stand in for that call" not in persona


def test_the_routing_rule_is_absent_without_the_suggestion_tool() -> None:
    """A role with no starter work mounted must not be told to call for it."""
    persona = build_persona([t for t in _ALL_TOOLS if t != "get_suggested_tasks"])

    assert "get_suggested_tasks" not in persona
    assert "When the hire asks what to work on" not in persona


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


_PATH_TOOLS = (
    "get_my_onboarding_path",
    "complete_step",
    "complete_task",
    "answer_question",
    "add_path_step",
    "request_skip",
)


def test_a_hire_without_a_path_meets_a_mentor_that_never_mentions_one() -> None:
    """The backend mounts the path tools only for a hire who has a path, so a mentor
    without them must not describe walking one."""
    persona = build_persona(_ALL_TOOLS)

    assert "onboarding path" not in persona
    for tool in _PATH_TOOLS:
        assert tool not in persona


def test_the_path_is_the_plan_and_the_blueprint_is_not_the_mentors() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    # The whole point of reading it: one plan, and it is the one a person wrote.
    assert "the path wins" in persona
    # The authority line. The mentor edits the hire's copy, never the curriculum.
    assert "You never edit the blueprint" in persona
    assert "*their copy*" in persona


def test_the_mentor_is_a_tutor_on_a_question_and_never_a_shortcut() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    # It is not told the answer, and the clause says so rather than only forbidding
    # it -- an honest "I do not have it" is the behaviour we want when asked.
    assert "not told which answer is correct" in persona
    assert "you do not have it" in persona


def test_completing_a_step_is_asked_for_never_announced() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "Only the hire knows whether they have actually done a step" in persona
    assert "Ask; do not announce" in persona


def test_a_done_checklist_on_an_open_step_is_raised_unprompted() -> None:
    """A hire who ticked every line but never finished the step is locked out of what
    comes next without knowing why -- and will not ask about the step doing it."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "READY TO CLOSE" in persona
    assert "if they say not yet, leave it" in persona


def test_a_wrong_answer_can_become_a_refresher_step_but_never_the_answer() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "refresher step" in persona
    assert "never the answer" in persona
    assert "not after every wrong answer" in persona


def test_phases_are_a_choice_not_a_queue() -> None:
    """A hire who finished phase 1 and picked phase 3 was sent back to phase 2."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "Phases are not a queue" in persona
    assert "never tell them to take the lowest number first" in persona


def test_the_path_is_described_as_a_graph_not_a_sequence() -> None:
    """It told a hire "after #6 comes #7" about items that did not depend on each
    other, because it read the page's numbering as an order."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "dependency graph" in persona
    assert "not the order they come in" in persona


def test_narrowing_the_options_is_hinting() -> None:
    """No answer stated, and the question given away all the same: "one option
    matches the title of #1 word for word"."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "Narrowing the options down is hinting too" in persona
    assert "never say how close a wrong answer was" in persona


def test_a_ready_step_is_answered_where_they_are_first_never_button_first() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "Never lead with the button" in persona
    assert "never lecture them about being sure" in persona


def test_an_added_step_is_placed_in_the_graph() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "never just at the end" in persona
    assert "`waits_on`" in persona
    assert "`unlocks`" in persona


def test_a_skip_is_requested_with_the_hires_reason_and_decided_by_the_pm() -> None:
    """The mentor files the request; it does not grant it, and it does not invent
    the reason. Without saying so it either promised a skip or sent a request that
    would sit."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "a request their PM decides" in persona
    assert "`request_skip`" in persona
    assert "Ask why first" in persona
    assert "Never promise it will be accepted" in persona
    # Finishing a step drops its pending skip, which the hire would never notice.
    assert "finishing it withdraws the request" in persona


def test_no_skip_clause_without_the_skip_action() -> None:
    persona = build_persona([*_ALL_TOOLS, "get_my_onboarding_path", "complete_step"])

    assert "request_skip" not in persona


def test_each_path_clause_is_gated_on_its_own_tool() -> None:
    read_only = build_persona([*_ALL_TOOLS, "get_my_onboarding_path"])

    # The read mounts the tutor framing; the actions stay unmentioned until they are
    # mounted, so the mentor never offers a proposal it cannot make.
    assert "onboarding path" in read_only
    assert "`complete_step`" not in read_only
    assert "`answer_question`" not in read_only
    assert "`add_path_step`" not in read_only


def test_a_question_about_the_path_is_never_answered_from_the_work_pool() -> None:
    """The two-systems problem, moved inside one conversation: "where am I?" has two
    plausible answers and only one of them is the plan."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "the path, first and always" in persona
    # The other direction too: progress questions belong to the metrics, not the path.
    assert "never say what comes next" in persona


def test_how_the_onboarding_is_going_is_the_paths_question() -> None:
    """The metrics are about work. Routing "how is my onboarding going" to them
    would make the first accepted contribution the end of onboarding again."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert 'how is my onboarding going" -> the path' in persona
    assert "never say how far along their onboarding is" in persona


def test_a_step_that_points_at_something_is_read_with_the_hire() -> None:
    """Explaining why a step exists was in the persona; going through what it points
    at was not, so a step reading "read issue 123" got pointed at rather than read."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "look it up with `search_docs` and go through it with them" in persona
    assert "not a step you can only point at" in persona


def test_no_material_clause_without_the_search_tool() -> None:
    persona = build_persona(
        [t for t in _ALL_TOOLS if t != "search_docs"] + list(_PATH_TOOLS)
    )

    assert "go through it with them" not in persona


def test_picking_up_work_is_never_gated_on_how_far_the_path_got() -> None:
    """A hire with issues in the pool asked whether there was something they could
    do and was told there was not. The routing clause had the suggested work coming
    "only when the path has nothing open", so a half-finished path read as a closed
    door -- a gate nobody designed, in the one place the hire cannot see it."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "Real work is open to them from day one" in persona
    assert "Never make it conditional on onboarding progress" in persona
    # The routing table has to agree with the clause, or the mentor holds both.
    assert "is there an issue I could pick up" in persona
    assert "never tell them they are not far enough along" in persona


def test_the_ambiguous_question_gets_both_answers_rather_than_one() -> None:
    """ "What should I work on" is genuinely two questions. Answering only the path
    hides the pool; answering only the pool hides the plan somebody wrote."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "ambiguous, so say both" in persona


def test_the_mentor_stays_with_a_task_the_hire_claimed() -> None:
    persona = build_persona([*_ALL_TOOLS, "open_orientation", *_PATH_TOOLS])

    assert "Once they have claimed something, stay with it" in persona
    assert "`open_orientation`" in persona


def test_no_follow_through_clause_without_the_packet() -> None:
    # Promising a packet that is not mounted is the button-that-never-appears defect.
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "`open_orientation`" not in persona
    assert "Once they have claimed something" not in persona


def test_the_path_comes_before_setup_and_setup_before_work() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert (
        persona.index("The hire has an onboarding path")
        < persona.index("`get_arrival_steps`")
        < persona.index("`get_suggested_tasks`")
    )


def test_no_ramp_towards_a_first_contribution_is_left_in_the_persona() -> None:
    """Task 0 and "doing real work is the point" were the old onboarding: a ramp
    that ended at a first accepted contribution. The path replaced it."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS, "open_orientation"]).lower()

    assert "task 0" not in persona
    assert "claim_task_zero" not in persona
    assert "the path to it" not in persona


def test_no_routing_rule_where_there_is_nothing_to_route_between() -> None:
    # A hire with a path but no state tools has one place an answer can come from, so a
    # rule about choosing would be noise.
    persona = build_persona(["search_docs", "get_my_onboarding_path"])

    assert "onboarding path" in persona
    assert "first and always" not in persona


def test_a_locked_item_is_never_agreed_to() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    # It said yes to a step the hire's own page refuses to open. Told what locked means,
    # the answer is what it is waiting on.
    assert "LOCKED" in persona
    assert "however directly they ask" in persona


def test_items_are_named_by_number_and_linked() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "linked, then its title" in persona
    assert "means that item" in persona


def test_a_refused_proposal_is_never_described_as_a_button() -> None:
    """It told a hire to click something that was never rendered. Mounted with any
    action at all, because the mistake is not specific to one."""
    for mounted in (["claim_goal"], ["add_path_step"], list(_PATH_TOOLS)):
        persona = build_persona(mounted)

        assert "NOT PROPOSED" in persona

    assert "NOT PROPOSED" not in build_persona(["search_docs", "get_my_metrics"])


def test_part_of_a_step_is_a_line_not_the_whole_step() -> None:
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS, "complete_task"])

    assert "`complete_task`" in persona
    # The product allows a finished step with open lines; the mentor must not invent a
    # rule the product does not have.
    assert "checklist has to be empty" in persona


def test_offering_is_described_as_a_tool_call_not_as_a_sentence() -> None:
    """The failure this exists for: hires were told to click a button that no tool call
    had produced, so there was nothing on screen."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "by calling its tool" in persona
    assert "Never describe a button instead of making one" in persona
    # And the arguments, because a call missing one is refused and also shows no button.
    assert "pass every argument the tool asks for" in persona


def test_an_item_is_written_as_a_linked_number() -> None:
    """A literal shape rather than a description of one: a small model copies a pattern
    far more reliably than it follows an instruction about formatting."""
    persona = build_persona([*_ALL_TOOLS, *_PATH_TOOLS])

    assert "[#3](/onboarding?step=" in persona
    assert "character for character" in persona


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

    assert "This turn you can only search" in persona
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
    assert "get_suggested_tasks" not in persona


def test_team_mode_addresses_the_manager_not_a_hire() -> None:
    persona = build_persona(_TEAM_TOOLS, team_mode=True)

    assert "manager of one project" in persona
    assert "Never greet them as a new hire" in persona
    assert "the tutor who guides a new hire" not in persona


def test_team_mode_drops_every_hire_directed_clause() -> None:
    """Dropped by mode, not only by mounting: a manager must never be asked to
    settle where *they* are starting from."""
    persona = build_persona([*_TEAM_TOOLS, *_ALL_TOOLS], team_mode=True)

    assert "get_arrival_steps" not in persona
    assert "claim_goal" not in persona
    assert "get_competencies_to_assess" not in persona
    assert "record_assessment" not in persona
    assert "hire-state tools" not in persona
    # The identity already forbids it; dropping the routing rule as well means a
    # backend that one day mounts the tool still cannot hand a manager a task list.
    assert "get_suggested_tasks" not in persona
    assert "what to work on" not in persona


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
    assert "This turn you can only search" in persona
    assert "never as a judgment of the person" in persona
    assert "get_team_attention" not in persona


def test_the_defaults_are_todays_behaviour() -> None:
    assert build_persona(_ALL_TOOLS) == build_persona(
        _ALL_TOOLS, capabilities_enabled=True, team_mode=False
    )
    assert build_persona(_ALL_TOOLS, DEFAULT_VOCABULARY) == build_persona(_ALL_TOOLS)


def test_search_only_names_grep_only_when_it_is_mounted() -> None:
    with_grep = build_persona(["search_docs", "grep"], capabilities_enabled=False)
    without = build_persona(["search_docs"], capabilities_enabled=False)
    team = build_persona(
        ["search_docs", "grep"], capabilities_enabled=False, team_mode=True
    )

    assert "`grep` for an exact name" in with_grep
    assert "`grep` for an exact name" in team
    assert "grep" not in without
