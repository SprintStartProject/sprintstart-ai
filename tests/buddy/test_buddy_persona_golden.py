"""The assembled persona is pinned byte for byte, across modes and toolsets.

The persona is a prompt, so a changed word is a changed product. The readable golden
covers the named toolsets; the digest golden covers every pair of tools and a seeded
sample of larger subsets, so a clause that moves or loses its gate shows up as a
changed line. After a deliberate change rerun with ``UPDATE_PROMPT_GOLDENS=1`` and
review the diff.
"""

import hashlib
import itertools
import os
import random
from pathlib import Path

from buddy.persona import build_persona
from buddy.vocabulary import Vocabulary
from llm.base import Message
from tests.support.golden import assert_matches_golden

_ALL_TOOLS = (
    "search_docs",
    "grep",
    "get_app_guide",
    "get_arrival_steps",
    "get_my_metrics",
    "get_my_competencies",
    "get_suggested_tasks",
    "get_competencies_to_assess",
    "get_my_onboarding_path",
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
    "get_team_attention",
    "find_member",
    "get_member_progress",
    "open_area",
)
_DELIVERY = Vocabulary("ceremony", "ceremonies", "run")
_DIGESTS = Path(__file__).parent / "persona_digests.txt"


def _hire_tools() -> list[str]:
    return [
        t
        for t in _ALL_TOOLS
        if t
        not in ("get_team_attention", "find_member", "get_member_progress", "open_area")
    ]


def _named_scenarios() -> dict[str, str]:
    everything = list(_ALL_TOOLS)
    return {
        "hire-full": build_persona(_hire_tools()),
        "hire-full-delivery-vocabulary": build_persona(_hire_tools(), _DELIVERY),
        "hire-no-tools": build_persona([]),
        "hire-search-only-mode": build_persona(everything, capabilities_enabled=False),
        "hire-search-only-mode-no-grep": build_persona(
            ["search_docs"], capabilities_enabled=False
        ),
        "hire-no-path": build_persona(
            [t for t in _hire_tools() if t != "get_my_onboarding_path"]
        ),
        "team-full": build_persona(everything, team_mode=True),
        "team-no-tools": build_persona([], team_mode=True),
        "team-search-only-mode": build_persona(
            everything, capabilities_enabled=False, team_mode=True
        ),
        "team-search-only-mode-no-grep": build_persona(
            ["search_docs"], capabilities_enabled=False, team_mode=True
        ),
        **{f"hire-only-{tool}": build_persona([tool]) for tool in _ALL_TOOLS},
        **{
            f"team-only-{tool}": build_persona([tool], team_mode=True)
            for tool in _ALL_TOOLS
        },
    }


def _digest(persona: str) -> str:
    return hashlib.sha256(persona.encode("utf-8")).hexdigest()[:8]


def _combinations() -> list[tuple[str, ...]]:
    pairs = list(itertools.combinations(_ALL_TOOLS, 2))
    rng = random.Random(20261023)
    sampled = [
        tuple(rng.sample(_ALL_TOOLS, rng.randint(3, len(_ALL_TOOLS))))
        for _ in range(300)
    ]
    return [*pairs, *sampled]


def _digest_lines() -> str:
    lines = []
    for index, combo in enumerate(_combinations()):
        for mode, kwargs in (
            ("hire", {}),
            ("team", {"team_mode": True}),
            ("hire-search", {"capabilities_enabled": False}),
            ("team-search", {"team_mode": True, "capabilities_enabled": False}),
        ):
            persona = build_persona(combo, **kwargs)
            lines.append(f"{mode} {index} {_digest(persona)}\n")
    return "".join(lines)


def test_named_toolsets_match_the_golden() -> None:
    messages: list[Message] = [
        {"role": name, "content": persona}
        for name, persona in _named_scenarios().items()
    ]
    assert_matches_golden("buddy_persona", messages)


def test_every_pair_and_sampled_subset_matches_the_digests() -> None:
    actual = _digest_lines()
    if os.environ.get("UPDATE_PROMPT_GOLDENS") == "1":
        _DIGESTS.write_text(actual, encoding="utf-8", newline="\n")
    assert actual == _DIGESTS.read_text(encoding="utf-8")
