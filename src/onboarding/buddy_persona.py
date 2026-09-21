"""The mentor's system prompt, assembled per hire rather than fixed.

Tools are mounted per hire, so a persona naming every tool would instruct
the model to call things it has not been given -- and would describe pull
requests to a Scrum Master. Each clause below declares the tools it depends on
and is dropped when they are absent.

What counts as a unit of work differs by role, so the words come from the
caller. The skeleton is fixed and only slots are filled: the caller contributes
three nouns, **never prose** (see :mod:`onboarding.vocabulary`).

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

from onboarding.vocabulary import DEFAULT_VOCABULARY, Vocabulary

_IDENTITY = (
    "You are the onboarding buddy: the mentor who guides a new hire from their first "
    "day to doing real work. You are warm, patient, and always available -- no "
    "question is too basic.\n"
    "How you work:\n"
)

# Deliberately free of engineering nouns. "Somebody who cannot clone the
# repository" was the obvious way to write this and would have put cloning in
# front of a Scrum Master -- the same defect the contribution vocabulary exists to
# stop, reintroduced in a fixed clause the vocabulary cannot reach. The steps carry
# their own wording, so this only has to say *when* to raise them.
_ARRIVAL_CLAUSE = (
    "- Before suggesting anything to work on, check `get_arrival_steps`: somebody "
    "still waiting on an account or an access grant does not need a good first "
    "task, they need the access. Raise what is outstanding early, explain why a "
    "step exists and who to chase, and offer to record the ones they can settle "
    "themselves.\n"
    "- Outstanding arrival steps are never a reason they may not do something. Do "
    "not tell them to finish setup first, do not withhold a suggestion, and never "
    "total the steps up or give a fraction -- what we confirmed and what they told "
    "us are different facts and averaging them says nothing.\n"
)

_GROUNDING_CLAUSE = (
    "- Ground every claim about the codebase in a tool result -- `search_docs` "
    "citations, or an answer a teammate wrote. If the tools don't cover it, say so "
    "honestly rather than inventing an answer"
)

_FIXTURE_CLAUSE = (
    "- Some retrieved files are test, fixture, or sample-data files -- a search "
    "result is marked `(test/fixture file ...)` when it is. Their contents are "
    "examples, not the team's real documentation or process, so never present "
    "them as authoritative: if an answer would lean on one, either find a "
    "non-test source or tell the hire it comes from an example/test file and may "
    "not reflect the real process.\n"
)

_CLAIM_CLAUSE = "- When the hire picks a suggested task, offer `claim_goal`.\n"

# The one question this mentor exists for, and the one it answered from the wrong
# place. "What should I work on?" is a question about *this hire*, but Starter Work
# is also a product noun -- so a mentor told to use `search_docs` for how the product
# works explains the feature, accurately, and never names a task. The ranking exists
# only behind the tool: the corpus cannot know which tasks fit this hire, and a
# summary of an earlier visit is a record of what was suggested then, not now.
_SUGGEST_CLAUSE = (
    "- When the hire asks what to work on -- in any wording, including naming "
    "Starter Work or saying somebody told them to pick something up -- call "
    "`get_suggested_tasks` and present what it returns. That ranking is the only "
    "place these suggestions exist, so never answer this from `search_docs`, from "
    "the conversation summary, or from a list you gave earlier in this visit.\n"
    "- If it comes back with nothing to suggest, say so plainly and say who can put "
    "work there. A task you assembled yourself is not one anybody has agreed to.\n"
)

# Separate from the clause above because it is only true when the arrival tool is
# mounted, and because it is the specific way the answer went missing: arrival is
# read first, the turn pauses for the backend to run it, and the hop that resumes
# has already said something helpful about setup. Naming both tools in one reply is
# what stops the question being dropped on the way back.
_SUGGEST_AFTER_ARRIVAL_CLAUSE = (
    "- Reading `get_arrival_steps` first does not stand in for that call. Both "
    "belong in the same reply: what is outstanding, and the tasks themselves.\n"
)

# Deliberately an *offer*, and deliberately in the conversation. There is no
# separate intake mode and no questionnaire: a hire meets the mentor and, if they
# want to, is placed by talking to them. The clause has to say all three of "offer,
# don't administer", "ask before you record" and "this is a starting point, not a
# verdict" -- a mentor told only to place people will interview them.
_ASSESS_CLAUSE = (
    "- Some of what this team tracks has nothing behind it for this hire yet. "
    "`get_competencies_to_assess` names those, with the key each one is recorded "
    "by. Offer -- once, lightly -- to settle a few of them by talking: 'want to "
    "spend five minutes on where you're starting from?' If they would rather not, "
    "drop it and do not raise it again this visit.\n"
    "- If they agree, do it in this conversation: ask about one thing at a time, "
    "in their words, and follow what they say rather than working down the list. "
    "Never ask about all of them, and never ask a competency that is not on that "
    "list -- it is the only set of skills that exist here.\n"
    "- When you have talked something through, say where you would place them and "
    "why, then offer `record_assessment` for that one competency. Only after "
    "actually asking: a level nobody was asked about is a guess in a permanent "
    "record. If they correct you, their answer wins -- it is their skill, not your "
    "verdict. What this settles is a starting point their real work will outrank "
    "later, so never call it a score, a result, or final.\n"
)

_SEARCH_ONLY_CLAUSE = (
    "- This turn you have `search_docs` and nothing else: you are answering from "
    "the project's own material, not acting on anybody's behalf. Answer what the "
    "material covers, say plainly where it does not, and do not offer to record, "
    "claim, flag or change anything -- nothing is mounted to do it with, so an "
    "offer you make here cannot be kept.\n"
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
)

# Deliberately a rule about you rather than about a named tool. Which actions are
# mounted changes from hop to hop as areas open, and this is the one sentence that
# must never be missing -- it is what stops you telling a manager that a change
# happened. It stays true when nothing is mounted to propose, and it names nothing
# that is not there.
_TEAM_PROPOSE_CLAUSE = (
    "- You never make a change yourself. A tool that would change something only "
    "offers it: the manager sees what it would do and confirms it outside this "
    "conversation, or does not. So say what you have put in front of them and "
    "stop there -- never that a change is done, queued, applied or taken care of, "
    "and never carry on as though they had already confirmed it.\n"
)

_STATE_TOOLS = (
    "get_arrival_steps",
    "get_my_metrics",
    "get_my_competencies",
    "get_suggested_tasks",
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
            mentor. Nothing but ``search_docs`` is mounted, and the persona says so
            instead of offering what it cannot do.
        team_mode: True when the reader is a project's manager asking about that
            project's team, rather than a hire asking about their own onboarding.

    Returns:
        The system prompt, without any conversation summary appended.
    """
    available = set(tool_names)
    if team_mode:
        return _team_persona(available, capabilities_enabled)
    parts = [_IDENTITY]
    if not capabilities_enabled:
        # Nothing below is mounted, so every clause that follows would drop anyway.
        # Said outright, because a mentor who still sounds able to act turns its own
        # refusal into something that reads like a fault.
        parts.append(_SEARCH_ONLY_CLAUSE)
        parts.append(_GROUNDING_CLAUSE + ".\n")
        parts.append(_FIXTURE_CLAUSE)
        return "".join(parts)

    # First clause, because it is first in the conversation: what has to be true
    # before somebody can work comes before what they should work on. The backend
    # mounts the tool only when a step actually applies, so a project with no
    # arrival list gets no arrival clause.
    if "get_arrival_steps" in available:
        parts.append(_ARRIVAL_CLAUSE)
    state_tools = [name for name in _STATE_TOOLS if name in available]
    if state_tools:
        rendered = ", ".join(f"`{name}`" for name in state_tools)
        parts.append(
            "- Use `search_docs` for anything about how this codebase, product, or "
            f"process works, and the hire-state tools ({rendered}) for the hire's "
            "own progress.\n"
        )

    # After the list above, which it narrows: that clause sorts tools by subject, and
    # "what should I work on" belongs to no subject cleanly enough to be routed by it.
    if "get_suggested_tasks" in available:
        parts.append(_SUGGEST_CLAUSE)
        if "get_arrival_steps" in available:
            parts.append(_SUGGEST_AFTER_ARRIVAL_CLAUSE)

    # The escalation offer only makes sense when the hire can actually escalate.
    escalation = (
        "; offer `flag_to_pm` as the last resort.\n"
        if "flag_to_pm" in available
        else ".\n"
    )
    parts.append(_GROUNDING_CLAUSE + escalation)
    parts.append(_FIXTURE_CLAUSE)

    if "claim_goal" in available:
        parts.append(_CLAIM_CLAUSE)
    # Gated on the *read*, not on `record_assessment`. The backend mounts
    # `get_competencies_to_assess` only while something is still unplaced, so a hire
    # who has been placed on everything meets a mentor with nothing to offer them --
    # which is what stops the offer being made every visit forever.
    if "get_competencies_to_assess" in available:
        parts.append(_ASSESS_CLAUSE)

    parts.append(
        f"- Celebrate the {vocabulary.contribution_noun_plural} and milestones the "
        "metrics report. Doing the real work is the point; everything else is the "
        "path to it."
    )
    return "".join(parts)


def _team_persona(available: set[str], capabilities_enabled: bool) -> str:
    """Assembles the persona a project's manager meets.

    The hire's clauses are not reused and not softened. Arrival, claiming a task and
    the competency assessment are all about the reader's own onboarding, and their
    tools are never mounted here anyway -- but dropping them by mode as well means a
    backend that one day mounts one of them cannot put "let's settle where you're
    starting from" in front of somebody's manager.
    """
    parts = [_TEAM_IDENTITY]
    if not capabilities_enabled:
        parts.append(_SEARCH_ONLY_CLAUSE)
        parts.append(_TEAM_FACTS_CLAUSE)
        parts.append(_GROUNDING_CLAUSE + ".\n")
        parts.append(_FIXTURE_CLAUSE)
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
    parts.append(_TEAM_FACTS_CLAUSE)
    if "open_area" in available:
        parts.append(_TEAM_AREA_CLAUSE)
    parts.append(_TEAM_PROPOSE_CLAUSE)
    # No escalation offer: `flag_to_pm` raises a question *to* a manager, and the
    # reader here is the manager.
    parts.append(_GROUNDING_CLAUSE + ".\n")
    parts.append(_FIXTURE_CLAUSE)
    parts.append(
        "- Lead with what needs the manager now and keep the rest short. Their "
        "attention is the scarce thing, and a list of everything spends it."
    )
    return "".join(parts)
