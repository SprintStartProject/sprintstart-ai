"""The mentor's system prompt, assembled per hire rather than fixed.

Tools are mounted per hire, so a persona naming every tool would instruct
the model to call things it has not been given -- and would describe pull
requests to a Scrum Master. Each clause below declares the tools it depends on
and is dropped when they are absent.

What counts as a unit of work differs by role, so the words come from the
caller. The skeleton is fixed and only slots are filled: the caller contributes
three nouns, **never prose** (see :mod:`buddy.vocabulary`).

Any fixed clause must be checked against that vocabulary too -- wording like
"clone the repository" puts engineering work in front of a role that has none.

Two flags from the backend choose *which* persona is assembled, and both arrive on
every hop because the persona is rebuilt on every hop:

- ``capabilities_enabled=False`` is the hire asking the corpus rather than the
  mentor. No backend tool is mounted, so the tool-gated clauses fall away on their
  own -- but an identity that still offers to act makes the resulting refusal read
  as a bug, so this mode says plainly what it can and cannot do.
- ``team_mode=True`` is a different reader altogether: the manager of one project,
  asking about that project's team. The hire-directed clauses are dropped by mode
  rather than only by mounting, because a manager must never be told what *they*
  should work on next.
"""

from collections.abc import Collection

from buddy.persona_clauses import (
    APP_GUIDE_CLAUSE,
    ARRIVAL_CLAUSE,
    ASSESS_CLAUSE,
    CLAIM_CLAUSE,
    CLAIMED_CLAUSE,
    DAY_ONE_CLAUSE,
    FIXTURE_CLAUSE,
    FLAG_ON_REQUEST_CLAUSE,
    GROUNDING_CLAUSE,
    IDENTITY,
    NO_BUTTON_CLAUSE,
    PATH_CLAUSE,
    PATH_COMPLETE_CLAUSE,
    PATH_MATERIAL_CLAUSE,
    PATH_QUESTION_CLAUSE,
    PATH_REFERENCE_CLAUSE,
    PATH_ROUTING_CLAUSE,
    PATH_SKIP_CLAUSE,
    PATH_STEP_CLAUSE,
    PATH_TASK_CLAUSE,
    SUGGEST_AFTER_ARRIVAL_CLAUSE,
    SUGGEST_CLAUSE,
    search_only_clauses,
)
from buddy.team_persona import team_persona
from buddy.vocabulary import DEFAULT_VOCABULARY, Vocabulary

_ACTION_TOOLS = (
    "flag_to_pm",
    "open_orientation",
    "claim_goal",
    "request_attestation",
    "set_github_login",
    "record_assessment",
    "complete_step",
    "complete_task",
    "answer_question",
    "add_path_step",
    "request_skip",
)

_STATE_TOOLS = (
    "get_arrival_steps",
    "get_my_metrics",
    "get_my_competencies",
    "get_suggested_tasks",
)

# In the order they are spoken: the question rule, then completing, ticking, adding
# and skipping, each mounted with the one tool it describes.
_PATH_ACTION_CLAUSES = (
    ("answer_question", PATH_QUESTION_CLAUSE),
    ("complete_step", PATH_COMPLETE_CLAUSE),
    ("complete_task", PATH_TASK_CLAUSE),
    ("add_path_step", PATH_STEP_CLAUSE),
    ("request_skip", PATH_SKIP_CLAUSE),
)


def build_persona(
    tool_names: Collection[str],
    vocabulary: Vocabulary = DEFAULT_VOCABULARY,
    *,
    capabilities_enabled: bool = True,
    team_mode: bool = False,
) -> str:
    """Assembles the persona for one reader of the buddy.

    Args:
        tool_names: Every tool mounted for this reader, backend and local. A clause
            whose tools are absent is omitted rather than softened, so the persona
            never mentions a capability this reader does not have.
        vocabulary: What one unit of this hire's accepted work is called. Defaults
            to the engineering wording when a caller supplies nothing. Unused in
            team mode, whose prose is about people rather than about their work.
        capabilities_enabled: False when the reader asked the corpus rather than the
            mentor. Only the searches (``search_docs``, plus ``grep`` when it is in
            ``tool_names``) are mounted, and the persona says so instead of
            offering what it cannot do.
        team_mode: True when the reader is a project's manager asking about that
            project's team, rather than a hire asking about their own onboarding.

    Returns:
        The system prompt, without any conversation summary appended.
    """
    available = set(tool_names)
    if team_mode:
        return team_persona(available, capabilities_enabled)
    if not capabilities_enabled:
        return _search_only_persona(available)
    parts = [IDENTITY]
    parts.extend(_path_clauses(available))
    parts.extend(_tool_routing_clauses(available))
    parts.extend(_grounding_clauses(available))
    parts.extend(_work_clauses(available))
    parts.append(
        f"- Celebrate the {vocabulary.contribution_noun_plural} and milestones the "
        "metrics report."
    )
    return "".join(parts)


def _search_only_persona(available: set[str]) -> str:
    """Assembles the persona for a hire who asked the corpus rather than the mentor.

    Nothing past the searches is mounted, so every other clause would drop anyway.
    Said outright, because a mentor who still sounds able to act turns its own refusal
    into something that reads like a fault.
    """
    return "".join(
        [
            IDENTITY,
            *search_only_clauses(available),
            GROUNDING_CLAUSE + ".\n",
            FIXTURE_CLAUSE,
        ]
    )


def _path_clauses(available: set[str]) -> list[str]:
    """The onboarding-path clauses, each gated on the tool it depends on.

    First, because the path is the onboarding: a mentor never told about it answers
    "what should I do next" out of the work pool while the hire is looking at a page
    that says something else. A hire with no path meets a mentor that never mentions
    one.
    """
    parts: list[str] = []
    if "get_my_onboarding_path" in available:
        parts.append(PATH_CLAUSE)
        parts.append(PATH_REFERENCE_CLAUSE)
        if "search_docs" in available:
            parts.append(PATH_MATERIAL_CLAUSE)
        # Only where the ambiguity exists. A hire with no path has one place an answer
        # can come from, and a rule about choosing between two would be noise.
        if any(name in available for name in _STATE_TOOLS):
            parts.append(PATH_ROUTING_CLAUSE)
    for tool, clause in _PATH_ACTION_CLAUSES:
        if tool in available:
            parts.append(clause)
    return parts


def _tool_routing_clauses(available: set[str]) -> list[str]:
    """Which tool answers which kind of question, from arrival through suggestions.

    Setup is not part of the path, but it comes before any work: what has to be true
    before somebody can work comes before what they should work on. The backend mounts
    the arrival tool only when a step actually applies, so a project with no arrival
    list gets no arrival clause.
    """
    parts: list[str] = []
    if "get_arrival_steps" in available:
        parts.append(ARRIVAL_CLAUSE)
    if any(name in available for name in _ACTION_TOOLS):
        parts.append(NO_BUTTON_CLAUSE)

    state_tools = [name for name in _STATE_TOOLS if name in available]
    if state_tools:
        rendered = ", ".join(f"`{name}`" for name in state_tools)
        parts.append(
            "- Use `search_docs` for anything about how this codebase, product, or "
            f"process works, and the hire-state tools ({rendered}) for the hire's "
            "own progress.\n"
        )
    # Straight after the routing rule above, which it narrows: "how this product
    # works" reads as including the app, and it does not.
    if "get_app_guide" in available:
        parts.append(APP_GUIDE_CLAUSE)

    # After the list above, which it narrows: that clause sorts tools by subject, and
    # "what should I work on" belongs to no subject cleanly enough to be routed by it.
    if "get_suggested_tasks" in available:
        parts.append(SUGGEST_CLAUSE)
        parts.append(DAY_ONE_CLAUSE)
        if "get_arrival_steps" in available:
            parts.append(SUGGEST_AFTER_ARRIVAL_CLAUSE)
    return parts


def _grounding_clauses(available: set[str]) -> list[str]:
    """Grounding and fixture caution; escalation is offered only if it can be kept."""
    escalation = (
        "; offer `flag_to_pm` as the last resort.\n" + FLAG_ON_REQUEST_CLAUSE
        if "flag_to_pm" in available
        else ".\n"
    )
    return [GROUNDING_CLAUSE + escalation, FIXTURE_CLAUSE]


def _work_clauses(available: set[str]) -> list[str]:
    """Claiming a task, staying with it, and the competency assessment."""
    parts: list[str] = []
    if "claim_goal" in available:
        parts.append(CLAIM_CLAUSE)
        # Only where the follow-through exists: the packet is the task they claimed,
        # so a mentor without it would promise a door it cannot open.
        if "open_orientation" in available:
            parts.append(CLAIMED_CLAUSE)
    # Gated on the *read*, not on `record_assessment`. The backend mounts
    # `get_competencies_to_assess` only while something is still unplaced, so a hire
    # who has been placed on everything meets a mentor with nothing to offer them --
    # which is what stops the offer being made every visit forever.
    if "get_competencies_to_assess" in available:
        parts.append(ASSESS_CLAUSE)
    return parts
