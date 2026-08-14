"""Tests for recovering tool calls a model leaked as text markup."""

from llm.tool_call_recovery import recover_tool_calls

# The exact shape a hire saw leak into a buddy answer: DeepSeek's tool-call markup
# (fullwidth-pipe DSML delimiters) that OpenRouter did not lift into tool_calls.
_LEAKED = (
    '<｜｜DSML｜｜tool_calls> <｜｜DSML｜｜invoke name="search_docs"> '
    '<｜｜DSML｜｜parameter name="query" string="true">'
    "open pull request waiting review 1514 hours"
    "</｜｜DSML｜｜parameter> </｜｜DSML｜｜invoke> </｜｜DSML｜｜tool_calls>"
)


def test_recovers_a_leaked_search_docs_call():
    calls, text = recover_tool_calls(_LEAKED)

    assert len(calls) == 1
    assert calls[0].name == "search_docs"
    assert calls[0].arguments == {
        "query": "open pull request waiting review 1514 hours"
    }
    # The markup is stripped; nothing of it is left as visible text.
    assert text == ""


def test_keeps_prose_before_the_leaked_block():
    content = "Let me check that for you.\n\n" + _LEAKED
    calls, text = recover_tool_calls(content)

    assert len(calls) == 1
    assert text == "Let me check that for you."


def test_recovers_multiple_invokes():
    content = (
        '<invoke name="get_my_metrics"></invoke>'
        '<invoke name="search_docs">'
        '<parameter name="query">code review SLA</parameter></invoke>'
    )
    calls, _ = recover_tool_calls(content)

    assert [call.name for call in calls] == ["get_my_metrics", "search_docs"]
    assert calls[1].arguments == {"query": "code review SLA"}


def test_plain_text_answer_is_left_untouched():
    content = "You have two open pull requests: #128 and #142."
    calls, text = recover_tool_calls(content)

    assert calls == []
    assert text == content


def test_prose_mentioning_the_word_invoke_is_not_a_false_positive():
    # No markup corroborates it, so it must not be parsed as a call.
    content = "To invoke the build you run ./gradlew build."
    calls, text = recover_tool_calls(content)

    assert calls == []
    assert text == content


def test_coerces_non_string_parameter_values():
    content = (
        '<invoke name="submit_verification">'
        '<parameter name="passed">true</parameter>'
        '<parameter name="count">3</parameter></invoke>'
    )
    calls, _ = recover_tool_calls(content)

    assert calls[0].arguments == {"passed": True, "count": 3}


# ── guard_stream ─────────────────────────────────────────────────────────────
# The visible answer is streamed, and that phase has no tool loop -- so leaked
# markup there cannot be run, only kept off the hire's screen.

from llm.tool_call_recovery import guard_stream  # noqa: E402

# Verbatim from a tutor's session on 2026-08-03: a warm greeting, then two
# search_docs calls the model wrote as text because it had no way to call them.
_LEAKED_ANSWER = (
    "Welcome back! Ready to get your hands dirty? Issue #165 is still the "
    "perfect starter.\n"
    '<｜｜DSML｜｜tool_calls> <｜｜DSML｜｜invoke name="search_docs"> '
    '<｜｜DSML｜｜parameter name="query" string="true">GithubUser entity fields'
    "</｜｜DSML｜｜parameter> </｜｜DSML｜｜invoke> "
    '<｜｜DSML｜｜invoke name="search_docs"> '
    '<｜｜DSML｜｜parameter name="query" string="true">personal access token PAT'
    "</｜｜DSML｜｜parameter> </｜｜DSML｜｜invoke> </｜｜DSML｜｜tool_calls>"
)


def _by_char(text: str) -> list[str]:
    """The worst case a real stream produces: one character per chunk."""
    return list(text)


def test_the_answer_survives_and_the_markup_does_not() -> None:
    out = "".join(guard_stream(_by_char(_LEAKED_ANSWER)))

    assert out.startswith("Welcome back!")
    assert "Issue #165 is still the perfect starter." in out
    for leaked in ("DSML", "tool_calls", "invoke", "search_docs", "<"):
        assert leaked not in out


def test_the_opening_delimiters_go_too() -> None:
    """Cutting at the marker rather than the tag would leave a stray '<｜｜'."""
    out = "".join(guard_stream(_by_char("hello <｜｜DSML｜｜tool_calls> junk")))

    assert out == "hello"


def test_ordinary_prose_passes_through_unchanged() -> None:
    prose = (
        "You can use `List<String>` here, and 5 < 7 is true. Nothing about this "
        "should be withheld or cut, however long it runs on for."
    )

    assert "".join(guard_stream(_by_char(prose))) == prose


def test_nothing_is_held_back_at_the_end_of_a_short_answer() -> None:
    """A whole answer shorter than the hold-back must still be delivered."""
    assert "".join(guard_stream(["ok"])) == "ok"


def test_an_empty_stream_yields_nothing() -> None:
    assert list(guard_stream([])) == []
    assert list(guard_stream(["", ""])) == []


def test_markup_split_across_chunks_is_still_caught() -> None:
    """The whole point: no single chunk contains a complete marker."""
    chunks = ["Here you go.", " <｜｜DS", "ML｜｜to", "ol_calls>", ' invoke name="x"']

    assert "".join(guard_stream(chunks)) == "Here you go."


def test_an_answer_that_is_only_markup_yields_nothing() -> None:
    assert "".join(guard_stream(_by_char("<｜｜DSML｜｜tool_calls>"))) == ""


def test_output_is_not_delayed_to_the_end_of_the_stream() -> None:
    """A hold-back is not a buffer: text has to keep flowing as it arrives."""
    long_answer = "word " * 200
    emitted = list(
        guard_stream([long_answer[i : i + 5] for i in range(0, len(long_answer), 5)])
    )

    assert len(emitted) > 1, (
        "the guard buffered the whole answer instead of streaming it"
    )
    assert "".join(emitted) == long_answer
