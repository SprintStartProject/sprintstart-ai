from llm.base import Message
from onboarding.query_fence import fence, fence_user_messages, is_fenced


def test_the_same_text_always_gets_the_same_fence() -> None:
    assert fence("how do I deploy?") == fence("how do I deploy?")


def test_different_text_gets_a_different_marker() -> None:
    assert fence("a").splitlines()[0] != fence("b").splitlines()[0]


def test_the_fence_wraps_the_text_verbatim() -> None:
    lines = fence("line one\nline two").split("\n")

    assert lines[1:3] == ["line one", "line two"]
    assert lines[0] == lines[3]


def test_only_this_process_can_mark_text_as_fenced() -> None:
    marker = "--0123456789abcdef--"
    lookalike = f"{marker}\ntext\n{marker}"

    assert is_fenced(fence("text"))
    assert not is_fenced(lookalike)
    assert not is_fenced("")
    assert not is_fenced("plain")


def test_editing_the_inside_of_a_fence_breaks_it() -> None:
    fenced = fence("keep the rules")

    assert not is_fenced(fenced.replace("keep", "drop"))


def test_empty_text_fences_and_round_trips() -> None:
    assert is_fenced(fence(""))


def test_only_user_messages_are_fenced_and_nothing_is_mutated() -> None:
    original = [
        Message(role="system", content="rules"),
        Message(role="user", content="q"),
        Message(role="assistant", content="a"),
        Message(role="tool", content="result", tool_call_id="call_0"),
    ]

    fenced = fence_user_messages(original)

    assert [m.get("content") for m in fenced] == ["rules", fence("q"), "a", "result"]
    assert original[1].get("content") == "q"
    assert fenced[3].get("tool_call_id") == "call_0"


def test_fencing_is_idempotent() -> None:
    once = fence_user_messages([Message(role="user", content="q")])

    assert fence_user_messages(once) == once
