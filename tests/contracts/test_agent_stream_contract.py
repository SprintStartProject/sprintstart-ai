"""Producer side of the agent stream contract.

The backend's consumer test decodes the same examples, so a change here that the
backend cannot read fails one side or the other.
"""

import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator

from api.stream_contract import render
from tests.conftest import parse_sse_events
from tests.stubs.app_client import scripted_client
from tests.stubs.llm import ScriptedLLMClient

CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"
SCHEMA_FILE = CONTRACTS / "agent-stream-events.schema.json"
EXAMPLES = sorted((CONTRACTS / "examples" / "agent-stream").glob("*.json"))
STREAM_URL = "/api/v1/onboarding/buddy/agent/stream"


def _validator() -> Draft202012Validator:
    return Draft202012Validator(json.loads(SCHEMA_FILE.read_text("utf-8")))


def _errors(event: dict[str, Any]) -> list[str]:
    return [error.message for error in _validator().iter_errors(event)]


def test_committed_schema_matches_the_code() -> None:
    assert SCHEMA_FILE.read_text("utf-8") == render(), (
        "contracts/agent-stream-events.schema.json is stale; regenerate it from src/ "
        "with `uv run python -m api.stream_contract "
        "../contracts/agent-stream-events.schema.json` and update the backend's copy"
    )


@pytest.mark.parametrize("example", EXAMPLES, ids=[e.stem for e in EXAMPLES])
def test_every_example_follows_the_schema(example: Path) -> None:
    assert _errors(json.loads(example.read_text("utf-8"))) == []


def test_every_event_type_has_an_example() -> None:
    schema = json.loads(SCHEMA_FILE.read_text("utf-8"))
    types = {branch["properties"]["type"]["const"] for branch in schema["oneOf"]}
    examples = {json.loads(e.read_text("utf-8"))["type"] for e in EXAMPLES}

    assert types == examples


def test_a_streamed_turn_follows_the_schema() -> None:
    llm = ScriptedLLMClient(
        turns=[[("grep", {"patterns": ["login"]})]],
        answer="In auth.py.",
        reasoning="grep for it",
    )
    for client in scripted_client(llm):
        response = client.post(
            STREAM_URL,
            json={
                "messages": [{"role": "user", "content": "where is login?"}],
                "capabilities_enabled": False,
                "project_ids": ["p1"],
            },
        )

    events = parse_sse_events(response.text)
    assert {e["type"] for e in events} >= {"reasoning", "tool_use", "token", "result"}
    assert {e["type"]: _errors(e) for e in events if _errors(e)} == {}
