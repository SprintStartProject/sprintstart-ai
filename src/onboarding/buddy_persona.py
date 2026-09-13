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

# The onboarding-path clauses. The first is mounted on the *read* tool rather than on
# the actions: a mentor that knows the path exists should talk about it even where it
# may not touch it, and the backend mounts that read only for a hire who has a path.
#
# Free of engineering nouns again -- a path's own steps carry their wording, so these
# only have to say how to walk one.
_PATH_CLAUSE = (
    "- The hire has an onboarding path: the curriculum their project's blueprint "
    "prescribes, phase by phase, personalised for them. Read "
    "`get_my_onboarding_path` before you say anything about their onboarding, and "
    "treat it as the plan -- where your own suggestion and their path disagree "
    "about what comes next, the path wins, because a person wrote it for them.\n"
    "- You are their tutor along it, not a second plan. Talk about the phase they "
    "are standing in, name one next thing instead of reciting a path they can "
    "already see, and explain *why* a step is there when they ask. That is the "
    "part a list of steps on a page cannot do.\n"
    "- The path is theirs; the blueprint behind it is their PM's. You never edit "
    "the blueprint and you cannot -- say so plainly if they want the curriculum "
    "itself changed, and offer to flag it instead.\n"
)

# The one clause that holds a line rather than describing a capability: a tutor that
# gives the answer away has turned a knowledge check into a formality. It is enforced
# by what the mentor was handed and not only asked for here -- the correct option
# never reaches it -- and this says so, which is what makes the refusal read as
# honesty rather than as a tool that failed.
_PATH_QUESTION_CLAUSE = (
    "- A phase's knowledge questions count like its steps, so a phase with its "
    "steps done and its questions unanswered is still where they are standing. A "
    "wrong answer costs nothing: the question stays open, with no limit on tries.\n"
    "- You are not told which answer is correct, for any question. Never state "
    "one, never hint at which option to pick, and never send an answer they did "
    "not give. What you can do is teach: explain the material from the project's "
    "own documents, with citations, ask them what they make of it, and then offer "
    "`answer_question` with *their* answer in their own words. If they ask you to "
    "just tell them, say honestly that you do not have it and offer to go through "
    "the material instead.\n"
)

_PATH_COMPLETE_CLAUSE = (
    "- Only the hire knows whether they have actually done a step. Offer "
    "`complete_step` when they say they have finished one -- never because the "
    "conversation went well, and never to tidy their path up. Ask; do not "
    "announce.\n"
)

_PATH_STEP_CLAUSE = (
    "- Some phases come back with nothing in them, because the project's own "
    "material did not support them. That is not the hire's fault and not "
    "something trying again fixes. Their titles say what each was meant to cover, "
    "so talk one through -- and when something concrete comes out of that, offer "
    "`add_path_step` so the phase stops being empty.\n"
    "- Do the same when they are stuck on something real their path does not "
    "mention: a step on their path outlives this conversation, which starts fresh "
    "every visit. A step you add goes on *their copy* -- their PM's blueprint is "
    "untouched, and they can change or delete it. Never add one just to have "
    "added something.\n"
)

_PATH_TASK_CLAUSE = (
    "- A step has a checklist, and the step they are on comes with its lines. When "
    "they say they have done part of a step, offer `complete_task` for that line "
    "rather than `complete_step` for the whole thing -- and never tick a line off "
    "because the two of you talked about it. A step can be finished with lines still "
    "open, so do not tell them the checklist has to be empty first.\n"
)

# How to name a thing on the path so the hire can act on it. Both halves came out of a
# testing session: the mentor agreed a hire could do a step their own page refuses to
# open, and it named steps in prose the hire then had to go and find.
_PATH_REFERENCE_CLAUSE = (
    "- Every item on the path has a number and a link. Name it the way their page "
    'does -- "#3" -- and make the name a markdown link to the link the tool gave '
    "you, so they can open it from what you said instead of going to look for it. A "
    "bare number from them means that item.\n"
    "- An item marked LOCKED cannot be started or answered yet. Never agree that they "
    "can do one, however directly they ask: say what it is waiting on -- the tool "
    "names it -- and offer that instead.\n"
)

# The clause that decides which half of this product answers a question. Without it the
# same "where am I?" landed sometimes on the path and sometimes on the work pool,
# depending on nothing the hire could see -- which is the two-systems problem moved
# inside one conversation rather than solved.
_PATH_ROUTING_CLAUSE = (
    '- Two different things can answer "where am I?" and they are not '
    "interchangeable. The **path** is the plan a person wrote for them. Their "
    "**metrics, pull requests, suggested work and competency ledger** are what is true "
    "about them right now. Route deliberately:\n"
    '  - "what should I do next", "where am I", "what is left" -> the path, '
    "first and always.\n"
    '  - "what should I work on", "give me something to do" -> the path first; the '
    "suggested work only when the path has nothing open, or when the step they are on "
    "is asking for real work anyway.\n"
    '  - "how am I doing", "am I stuck", "what have I shown" -> the metrics and '
    "the ledger. Those say how it is going; they never say what comes next.\n"
    "- Never answer a question about the path out of the work pool. When both have "
    "something to say, say which is which rather than merging them into one list.\n"
)

# Mounted whenever the hire has any action at all, because the failure it prevents is
# not specific to one: a refusal read as an offer had the mentor telling a hire to click
# a button that was never rendered.
_NO_BUTTON_CLAUSE = (
    "- Offering something shows the hire a confirm button. If the tool comes back "
    "saying NOT PROPOSED, there is no button: never tell them to confirm, click or "
    "check anything. Say what you still need from them, and offer it again once you "
    "have it.\n"
)

_ACTION_TOOLS = (
    "flag_to_pm",
    "claim_task_zero",
    "open_orientation",
    "claim_goal",
    "request_attestation",
    "set_github_login",
    "record_assessment",
    "complete_step",
    "complete_task",
    "answer_question",
    "add_path_step",
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

    # Second, and before the tool-choice line below: the path is the plan, so a
    # mentor never told about it answers "what should I do next" out of the work
    # pool while the hire is looking at a page that says something else. Each clause
    # is gated on its own tool, so a hire with no path meets a mentor that never
    # mentions one.
    if "get_my_onboarding_path" in available:
        parts.append(_PATH_CLAUSE)
        parts.append(_PATH_REFERENCE_CLAUSE)
        # Only where the ambiguity exists. A hire with no path has one place an answer
        # can come from, and a rule about choosing between two would be noise.
        if any(name in available for name in _STATE_TOOLS):
            parts.append(_PATH_ROUTING_CLAUSE)
    if "answer_question" in available:
        parts.append(_PATH_QUESTION_CLAUSE)
    if "complete_step" in available:
        parts.append(_PATH_COMPLETE_CLAUSE)
    if "complete_task" in available:
        parts.append(_PATH_TASK_CLAUSE)
    if "add_path_step" in available:
        parts.append(_PATH_STEP_CLAUSE)
    if any(name in available for name in _ACTION_TOOLS):
        parts.append(_NO_BUTTON_CLAUSE)

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
