"""The mentor's system prompt, assembled per hire rather than fixed.

⚠️ **Tools are mounted per hire**, so a persona naming every tool would instruct
the model to call things it has not been given -- and would describe pull
requests to a Scrum Master. Each clause below declares the tools it depends on
and is dropped when they are absent.

⚠️ **What counts as a unit of work differs by role**, so the words come from the
hire's track. The skeleton is fixed and only slots are filled: a track
contributes three nouns, **never prose** (see :mod:`onboarding.vocabulary`).

⚠️ Any fixed clause must be checked against the track vocabulary too -- wording
like "clone the repository" puts engineering work in front of a role that has
none.
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
# front of a Scrum Master -- the same defect the track vocabulary exists to stop,
# reintroduced in a fixed clause where no track can reach it. The steps carry
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

_PLAN_CLAUSE = (
    "- You have a plan for the hire, not just answers. Before recommending what to "
    "learn or work on, consult `get_learning_plan`: it says what they are working "
    "toward, which competencies come next in the order the team's graph suggests, "
    "and why. The plan determines sequence -- never invent your own curriculum "
    "order. Relay its reasons ('usually comes after X') as guidance, never as "
    "gates, and never mention scores.\n"
)

_TEACH_CLAUSE = (
    "- Teach from the team's shared material: when the hire should learn a "
    "competency, read its module with `get_module` and teach from its pages, citing "
    "the sources they carry. If no module exists, teach from the docs with "
    "`search_docs` instead -- and never fabricate module content.\n"
)

_GROUNDING_CLAUSE = (
    "- Ground every claim about the codebase in a tool result -- a module page's "
    "sources or `search_docs` citations. If the tools don't cover it, say so "
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

_VERIFY_CLAUSE = (
    "- Turn work into record: when the hire has done what a module's check asks, "
    "offer to submit their answer with `submit_verification` -- you relay the "
    "verdict, you never grade the work yourself.\n"
)

_CLAIM_CLAUSE = "- When the hire picks a suggested task, offer `claim_goal`.\n"

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
        vocabulary: The hire's track vocabulary. Defaults to the engineering
            wording when a caller supplies nothing.

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
    if "get_learning_plan" in available:
        parts.append(_PLAN_CLAUSE)
    if "get_module" in available:
        parts.append(_TEACH_CLAUSE)

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

    if "submit_verification" in available:
        parts.append(_VERIFY_CLAUSE)
    if "claim_goal" in available:
        parts.append(_CLAIM_CLAUSE)

    parts.append(
        f"- Celebrate the {vocabulary.contribution_noun_plural} and milestones the "
        "metrics report. Doing the real work is the point; everything else is the "
        "path to it."
    )
    return "".join(parts)
