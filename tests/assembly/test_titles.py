from assembly.titles import generate_title
from llm.base import Message
from tests.stubs.llm import StubLLMClient
from tests.support.golden import assert_matches_golden


class _FixedLLM(StubLLMClient):
    def __init__(self, reply: str) -> None:
        super().__init__(generate_response=reply)
        self.calls: list[list[Message]] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.calls.append(messages)
        return super().generate(messages, temperature=temperature)


def test_title_prompt_matches_golden() -> None:
    llm = _FixedLLM("A title")

    generate_title(llm, "How do REST and GraphQL differ?", 40)

    assert_matches_golden("title_prompt", llm.calls[0])


def test_the_title_is_the_trimmed_answer() -> None:
    assert generate_title(_FixedLLM("  REST vs GraphQL \n"), "p", 50) == (
        "REST vs GraphQL"
    )


def test_an_answer_exactly_at_the_limit_is_kept() -> None:
    assert generate_title(_FixedLLM("abcdef"), "p", 6) == "abcdef"


def test_a_longer_answer_is_cut_at_the_limit_and_trimmed() -> None:
    assert generate_title(_FixedLLM("hello world"), "p", 6) == "hello"
