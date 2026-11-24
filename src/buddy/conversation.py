"""The message list a turn carries: the persona on top, then what was said.

The persona is rebuilt on every hop and the summary of older turns rides its system
message, so a resume needs nothing re-sent beyond the window."""

from collections.abc import Collection

from buddy.persona import build_persona
from buddy.query_fence import QUERY_FENCE_NOTE
from buddy.vocabulary import Vocabulary
from llm.base import (
    ChatResult,
    Message,
)

# How the summary enters the model's context: appended to the persona in the system
# message, so it rides the running conversation the caller carries between turns.
_SUMMARY_HEADER = "\n\nConversation so far (compressed memory of earlier turns):\n"


def _persona_prompt(
    summary: str | None,
    tool_names: Collection[str],
    vocabulary: Vocabulary,
    capabilities_enabled: bool,
    team_mode: bool,
) -> str:
    persona = build_persona(
        tool_names,
        vocabulary,
        capabilities_enabled=capabilities_enabled,
        team_mode=team_mode,
    )
    # Every mode and every hop: the fence is applied to every user message, so the
    # rule for reading it has to be in front of the model whenever one is.
    persona = persona.rstrip("\n") + "\n" + QUERY_FENCE_NOTE
    if not summary:
        return persona
    return persona + _SUMMARY_HEADER + summary


def _extract_summary(system_content: str) -> str | None:
    if _SUMMARY_HEADER in system_content:
        return system_content.split(_SUMMARY_HEADER, 1)[1]
    return None


def ensure_persona(
    messages: list[Message],
    summary: str | None,
    tool_names: Collection[str],
    vocabulary: Vocabulary,
    capabilities_enabled: bool,
    team_mode: bool,
) -> list[Message]:
    # The system message a resume carries is replaced, not kept: only its summary
    # survives. So both modes have to arrive on every hop -- a hop that dropped one
    # would rebuild the default persona mid-turn and answer a manager as a hire.
    effective_summary = summary
    rest = messages
    if messages and messages[0]["role"] == "system":
        if effective_summary is None:
            effective_summary = _extract_summary(messages[0].get("content") or "")
        rest = messages[1:]
    persona_msg = Message(
        role="system",
        content=_persona_prompt(
            effective_summary,
            tool_names,
            vocabulary,
            capabilities_enabled,
            team_mode,
        ),
    )
    return [persona_msg, *rest]


def assistant_message(result: ChatResult) -> Message:
    msg = Message(role="assistant", content=result.text)
    if result.tool_calls:
        msg["tool_calls"] = result.tool_calls
    # Kept for the next hop: a reasoning provider continues a tool-using thought
    # from these, and one with extended thinking rejects the turn without them.
    # They also go out on the wire message (``api.routes.buddy._from_message``),
    # so the hop after a backend tool gets them back from the caller.
    if result.reasoning:
        msg["reasoning"] = result.reasoning
    if result.reasoning_details:
        msg["reasoning_details"] = result.reasoning_details
    return msg


def tool_result_message(call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=call_id)
