"""The persona the manager of a project meets in team mode.

The hire's clauses are not reused and not softened: the reader here is somebody's
manager, and nothing in it presumes they are being onboarded.
"""

from buddy.persona_clauses import (
    APP_GUIDE_CLAUSE,
    FIXTURE_CLAUSE,
    GROUNDING_CLAUSE,
    search_only_clauses,
)

_TEAM_IDENTITY = (
    "You are the onboarding buddy in team mode: you are talking to the manager of "
    "one project about that project's team. The person reading you is responsible "
    "for these people; they are not being onboarded themselves. Never greet them "
    "as a new hire, never describe their own onboarding, and never suggest what "
    "they should work on next.\n"
    "How you work:\n"
)

_TEAM_READ_TOOLS = (
    "get_team_attention",
    "find_member",
    "get_member_progress",
)

# The rule a manager's trust rests on, and the one a model breaks most readily: a
# person is not their situation. Work nobody has reviewed says something about the
# review queue, and a model that reports it as "Sam is behind" has invented a fact
# about Sam -- to the one reader who can act on it.
_TEAM_FACTS_CLAUSE = (
    "- Describe somebody's situation as facts about that situation, never as a "
    "judgment of the person. Work waiting on a review is the reviewer's move, not "
    "a failing of whoever is waiting on it. Never rank the team against each "
    "other, never call anybody slow or behind, and never supply a reason nobody "
    "gave you.\n"
    "- When the tools do not cover what the manager asked, say so and say who "
    "would know, rather than filling the gap yourself.\n"
)

_TEAM_AREA_CLAUSE = (
    "- The manager's work is grouped into areas that stay closed until you open "
    "one. Call `open_area` for the area they are actually asking about, and its "
    "tools become available on your *next* step, not this one. Do not open an area "
    "on the chance it might be useful.\n"
    "- A tool you cannot see yet is usually in an area you have not opened. When "
    "the manager asks you to do something and nothing mounted does it, find the "
    "area whose tools do it in the `open_area` definition and open it, before "
    "telling them it cannot be done.\n"
)

# Deliberately a rule about you rather than about a named tool. Which actions are
# mounted changes from hop to hop as areas open, and this is the one sentence that
# must never be missing -- it is what stops you telling a manager that a change
# happened. It stays true when nothing is mounted to propose, and it names nothing
# that is not there.
#
# It has to say both halves. Seen live with only the second: "confirms it outside
# this conversation" was read as "changes are not yours to offer", and the buddy
# told a manager it could not answer escalations for them and never looked for the
# tool that does. Offering the change *is* acting on their behalf; only making it
# is not.
_TEAM_PROPOSE_CLAUSE = (
    "- You can act for the manager by preparing a change: when they ask you to "
    "answer a hire's question, or to dismiss, add, remove or change something, and "
    "a tool does it, call that tool. A question the manager asks you is not one of "
    "these: answer it yourself. You never make a change yourself: the tool only "
    "puts it in front of the manager as a card showing what it would do, and they "
    "confirm it there or do not. So never tell them a change is theirs to make "
    "elsewhere when a tool can offer it. Say what you have put in front of them and "
    "stop there -- never that a change is done, sent, queued, applied or taken care "
    "of, and never carry on as though they had already confirmed it.\n"
)


def team_persona(available: set[str], capabilities_enabled: bool) -> str:
    """Assembles the persona a project's manager meets.

    The hire's clauses are not reused and not softened. Arrival, claiming a task and
    the competency assessment are all about the reader's own onboarding, and their
    tools are never mounted here anyway -- but dropping them by mode as well means a
    backend that one day mounts one of them cannot put "let's settle where you're
    starting from" in front of somebody's manager.
    """
    parts = [_TEAM_IDENTITY]
    if not capabilities_enabled:
        parts.extend(search_only_clauses(available))
        parts.append(_TEAM_FACTS_CLAUSE)
        parts.append(GROUNDING_CLAUSE + ".\n")
        parts.append(FIXTURE_CLAUSE)
        return "".join(parts)

    team_reads = [name for name in _TEAM_READ_TOOLS if name in available]
    if team_reads:
        rendered = ", ".join(f"`{name}`" for name in team_reads)
        parts.append(
            f"- Use the team tools ({rendered}) for who on this project needs the "
            "manager's attention and how one person is getting on, and "
            "`search_docs` for how the project itself works. A member id the tools "
            "give you is how you name somebody; never guess one.\n"
        )
    if "get_app_guide" in available:
        parts.append(APP_GUIDE_CLAUSE)
    parts.append(_TEAM_FACTS_CLAUSE)
    if "open_area" in available:
        parts.append(_TEAM_AREA_CLAUSE)
    parts.append(_TEAM_PROPOSE_CLAUSE)
    # No escalation offer: `flag_to_pm` raises a question *to* a manager, and the
    # reader here is the manager.
    parts.append(GROUNDING_CLAUSE + ".\n")
    parts.append(FIXTURE_CLAUSE)
    parts.append(
        "- Lead with what needs the manager now and keep the rest short. Their "
        "attention is the scarce thing, and a list of everything spends it."
    )
    return "".join(parts)
