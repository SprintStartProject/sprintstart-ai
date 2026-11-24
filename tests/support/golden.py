"""Golden copies of the prompts the onboarding pipelines send to the model.

A prompt is behaviour: the model does what it is told, so a changed word is a
changed product. Comparing against a checked-in copy turns every such change into
a diff someone reviews. After a deliberate change, rerun the tests with
``UPDATE_PROMPT_GOLDENS=1`` and commit the rewritten files.
"""

import os
from pathlib import Path

from llm.base import Message

_DIR = Path(__file__).parent / "prompts"


def render(messages: list[Message]) -> str:
    """The conversation as plain text, one ``--- role ---`` header per message."""
    return "".join(f"--- {m['role']} ---\n{m['content']}\n" for m in messages)


def assert_matches_golden(name: str, messages: list[Message]) -> None:
    path = _DIR / f"{name}.txt"
    actual = render(messages)
    if os.environ.get("UPDATE_PROMPT_GOLDENS") == "1":
        path.parent.mkdir(exist_ok=True)
        path.write_text(actual, encoding="utf-8", newline="\n")
    expected = path.read_text(encoding="utf-8")
    assert actual == expected, f"prompt differs from {path.name}"
