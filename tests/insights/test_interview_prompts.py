import json

import pytest

from insights.interview_components import Component, RiskFile, Trigger
from insights.interview_prompts import (
    Coverage,
    DraftQuestion,
    Excerpt,
    QuestionType,
    generate_prompt,
    judge_prompt,
    parse_questions,
    parse_verdicts,
)

_FILE = RiskFile(
    repository="github:acme/app",
    path="src/auth/Login.kt",
    level="ORPHANED",
    score=0.8,
    kind="CODE",
    reasons=("Nobody who knows it is left.",),
)
_COMPONENT = Component("github:acme/app", "src/auth", (_FILE,))


def _questions(*entries: dict[str, object]) -> str:
    return json.dumps({"questions": list(entries)})


def test_the_generation_prompt_carries_findings_files_reasons_and_excerpts() -> None:
    prompt = generate_prompt(_COMPONENT, [Excerpt("auth.md", "Login uses OAuth.")], 3)
    user = prompt[1]["content"]

    assert "github:acme/app/src/auth" in user
    assert "orphaned: nobody who stays knows these files" in user
    assert "- src/auth/Login.kt (code): Nobody who knows it is left." in user
    assert "[auth.md]\nLogin uses OAuth." in user
    assert "at most 3 questions" in user


def test_the_generation_prompt_says_when_there_are_no_excerpts() -> None:
    assert "(none)" in generate_prompt(_COMPONENT, [], 3)[1]["content"]


def test_questions_keep_only_the_findings_and_files_of_the_component() -> None:
    raw = "Here: " + _questions(
        {
            "text": " Why does Login retry three times? ",
            "type": "rationale",
            "triggers": ["orphaned", "hotspot"],
            "files": ["src/auth/Login.kt", "src/other.kt"],
        }
    )

    [draft] = parse_questions(raw, _COMPONENT, 5)

    assert draft == DraftQuestion(
        "Why does Login retry three times?",
        QuestionType.RATIONALE,
        (Trigger.ORPHANED,),
        ("src/auth/Login.kt",),
    )


def test_a_question_without_known_findings_falls_back_to_concentration() -> None:
    raw = _questions({"text": "How is it deployed?", "type": "operational"})

    assert parse_questions(raw, _COMPONENT, 5)[0].triggers == (Trigger.CONCENTRATED,)


def test_blank_questions_are_dropped_and_the_count_is_capped() -> None:
    raw = _questions(
        {"text": " ", "type": "gotcha"},
        {"text": "A?", "type": "gotcha"},
        {"text": "B?", "type": "handover"},
    )

    assert [d.text for d in parse_questions(raw, _COMPONENT, 1)] == ["A?"]


@pytest.mark.parametrize(
    "raw", ["no json here", '{"questions": [{"text": "A?", "type": "opinion"}]}']
)
def test_a_malformed_generation_answer_is_an_error(raw: str) -> None:
    with pytest.raises(ValueError):
        parse_questions(raw, _COMPONENT, 5)


def test_the_judge_sees_each_question_with_its_own_excerpts() -> None:
    drafts = [
        DraftQuestion("A?", QuestionType.GOTCHA, (Trigger.HOTSPOT,), ()),
        DraftQuestion("B?", QuestionType.GOTCHA, (Trigger.HOTSPOT,), ()),
    ]

    user = judge_prompt(drafts, [[Excerpt("a.md", "about A")], []])[1]["content"]

    assert "Question 0: A?\nExcerpts:\n[a.md] about A" in user
    assert "Question 1: B?\nExcerpts:\n(none)" in user


def test_verdicts_fill_in_open_for_skipped_questions_and_ignore_bad_indexes() -> None:
    raw = json.dumps(
        {
            "verdicts": [
                {
                    "index": 1,
                    "coverage": "answered",
                    "answer": "It is X.",
                    "sources": ["a.md"],
                },
                {"index": 7, "coverage": "answered", "answer": "?"},
                {"index": 2, "coverage": "open", "answer": "ignored"},
            ]
        }
    )

    verdicts = parse_verdicts(raw, 3)

    assert [v.coverage for v in verdicts] == [
        Coverage.OPEN,
        Coverage.ANSWERED,
        Coverage.OPEN,
    ]
    assert verdicts[1].answer == "It is X."
    assert verdicts[1].sources == ("a.md",)
    assert verdicts[2].answer is None


def test_a_malformed_judge_answer_is_an_error() -> None:
    with pytest.raises(ValueError):
        parse_verdicts('{"verdicts": [{"index": 0, "coverage": "maybe"}]}', 1)
