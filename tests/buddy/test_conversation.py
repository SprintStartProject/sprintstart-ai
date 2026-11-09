from buddy.conversation import (
    assistant_message,
    ensure_persona,
    tool_result_message,
)
from buddy.persona import build_persona
from buddy.query_fence import QUERY_FENCE_NOTE
from buddy.vocabulary import DEFAULT_VOCABULARY
from llm.base import ChatResult, Message, ToolCall

_HEADER = "\n\nConversation so far (compressed memory of earlier turns):\n"
_TOOLS = {"search_docs"}


def _persona(*, team_mode: bool = False, capabilities: bool = True) -> str:
    persona = build_persona(
        _TOOLS,
        DEFAULT_VOCABULARY,
        capabilities_enabled=capabilities,
        team_mode=team_mode,
    )
    return persona.rstrip("\n") + "\n" + QUERY_FENCE_NOTE


def _ensure(
    messages: list[Message],
    summary: str | None = None,
    *,
    team_mode: bool = False,
    capabilities: bool = True,
) -> list[Message]:
    return ensure_persona(
        messages, summary, _TOOLS, DEFAULT_VOCABULARY, capabilities, team_mode
    )


def test_the_persona_is_the_first_message_and_ends_with_the_fence_note() -> None:
    user = Message(role="user", content="hi")

    [persona, rest] = _ensure([user])

    assert persona == {"role": "system", "content": _persona()}
    assert persona["content"].endswith(QUERY_FENCE_NOTE)
    assert rest == user


def test_a_summary_rides_the_persona_after_a_header() -> None:
    [persona] = _ensure([], "earlier we covered auth")

    assert persona["content"] == _persona() + _HEADER + "earlier we covered auth"


def test_no_summary_adds_no_header() -> None:
    assert _HEADER not in _ensure([], None)[0]["content"]
    assert _HEADER not in _ensure([], "")[0]["content"]


def test_a_carried_system_message_is_replaced_but_its_summary_survives() -> None:
    carried = Message(role="system", content="stale persona" + _HEADER + "kept notes")
    user = Message(role="user", content="next")

    [persona, rest] = _ensure([carried, user])

    assert persona["content"] == _persona() + _HEADER + "kept notes"
    assert rest == user


def test_an_explicit_summary_beats_the_carried_one() -> None:
    carried = Message(role="system", content="stale" + _HEADER + "old")

    [persona] = _ensure([carried], "new")

    assert persona["content"].endswith(_HEADER + "new")
    assert "old" not in persona["content"]


def test_a_carried_system_message_without_a_summary_adds_none() -> None:
    [persona] = _ensure([Message(role="system", content="just a persona")])

    assert persona["content"] == _persona()


def test_a_carried_system_message_with_no_content_is_dropped_cleanly() -> None:
    carried: Message = {"role": "system", "content": ""}

    [persona] = _ensure([carried])

    assert persona["content"] == _persona()


def test_only_a_leading_system_message_is_replaced() -> None:
    later = Message(role="system", content="notice" + _HEADER + "not a summary")
    user = Message(role="user", content="q")

    result = _ensure([user, later])

    assert result[1:] == [user, later]
    assert _HEADER not in result[0]["content"]


def test_the_mode_flags_change_the_persona() -> None:
    assert _ensure([], team_mode=True)[0]["content"] == _persona(team_mode=True)
    assert _ensure([], capabilities=False)[0]["content"] == _persona(capabilities=False)
    assert _persona(team_mode=True) != _persona()


def test_an_assistant_message_carries_calls_and_reasoning_when_present() -> None:
    call = ToolCall(id="c1", name="search_docs", arguments={"q": "x"})
    details: list[dict[str, object]] = [{"type": "reasoning.text", "text": "t"}]

    full = assistant_message(
        ChatResult(
            text="thinking",
            tool_calls=[call],
            reasoning="because",
            reasoning_details=details,
        )
    )
    bare = assistant_message(ChatResult(text="plain", tool_calls=[]))

    assert full == {
        "role": "assistant",
        "content": "thinking",
        "tool_calls": [call],
        "reasoning": "because",
        "reasoning_details": details,
    }
    assert bare == {"role": "assistant", "content": "plain"}


def test_a_tool_result_message_names_the_call_it_answers() -> None:
    assert tool_result_message("c1", "found") == {
        "role": "tool",
        "content": "found",
        "tool_call_id": "c1",
    }
