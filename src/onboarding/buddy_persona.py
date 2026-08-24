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

_STATE_TOOLS = (
    "get_arrival_steps",
    "get_my_metrics",
    "get_my_competencies",
    "get_suggested_tasks",
)


def build_persona(
    tool_names: Collection[str],
    vocabulary: Vocabulary = DEFAULT_VOCABULARY,
) -> str:
    """Assembles the mentor's persona for one hire.

    Args:
        tool_names: Every tool mounted for this hire, backend and local. A clause
            whose tools are absent is omitted rather than softened, so the persona
            never mentions a capability this hire does not have.
        vocabulary: What one unit of this hire's accepted work is called. Defaults
            to the engineering wording when a caller supplies nothing.

    Returns:
        The system prompt, without any conversation summary appended.
    """
    available = set(tool_names)
    parts = [_IDENTITY]

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
