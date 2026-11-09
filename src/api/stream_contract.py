"""The contract of ``/onboarding/buddy/agent/stream`` as JSON Schema.

The backend consumes these events; it keeps a copy of the generated schema under
``contracts/`` and decodes the shared examples in its own contract test. Regenerate
the committed file after changing an event (run from ``src/``):

    uv run python -m api.stream_contract ../contracts/agent-stream-events.schema.json
"""

import json
import sys
from pathlib import Path
from typing import Any

from api.schemas import BuddyAgentResponse

SCHEMA_ID = "https://sprintstart.dev/contracts/agent-stream-events.schema.json"

JsonSchema = dict[str, Any]


def _event(event_type: str, properties: JsonSchema, required: list[str]) -> JsonSchema:
    return {
        "type": "object",
        "properties": {"type": {"const": event_type}, **properties},
        "required": ["type", *required],
        "additionalProperties": False,
    }


def _fragment(event_type: str) -> JsonSchema:
    return _event(event_type, {"content": {"type": "string"}}, ["content"])


def _result_event() -> JsonSchema:
    response = BuddyAgentResponse.model_json_schema(ref_template="#/$defs/{model}")
    properties: JsonSchema = {"type": {"const": "result"}, **response["properties"]}
    return {
        "type": "object",
        "properties": properties,
        "required": ["type", *response.get("required", [])],
        "additionalProperties": False,
    }


def build_schema() -> JsonSchema:
    """The full event schema: one branch per event type, discriminated by ``type``."""
    response_defs = BuddyAgentResponse.model_json_schema(ref_template="#/$defs/{model}")
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": SCHEMA_ID,
        "title": "Buddy agent stream event (sprintstart-ai -> sprintstart-backend)",
        "oneOf": [
            _fragment("reasoning"),
            _fragment("token"),
            _event(
                "tool_use",
                {"name": {"type": "string"}, "arguments": {"type": "object"}},
                ["name", "arguments"],
            ),
            _event("error", {"message": {"type": "string"}}, ["message"]),
            _result_event(),
        ],
        "$defs": response_defs.get("$defs", {}),
    }


def render() -> str:
    """The schema exactly as committed: sorted keys, two-space indent, final newline."""
    return json.dumps(build_schema(), indent=2, sort_keys=True) + "\n"


if __name__ == "__main__":
    Path(sys.argv[1]).write_text(render(), encoding="utf-8", newline="\n")
