import pytest

from corpus.source_role import classify_source_role, is_source_role


@pytest.mark.parametrize(
    "filename",
    [
        "test_agent.py",
        "test_ollama_client.py",
        "foo_test.go",
        "widget.test.ts",
        "widget.spec.js",
        "conftest.py",
        "json_large_sample.json",
        "text_small_sample.txt",
        "markdown_large_sample.md",
        "users_fixture.json",
    ],
)
def test_classify_test_material(filename: str) -> None:
    assert classify_source_role(filename) == "test"


@pytest.mark.parametrize(
    "filename",
    [
        "anthropic_client.py",
        "chunker.py",
        "README.md",
        "dev-setup.md",
        "team-roles.md",
        "pyproject.toml",
        "latest.py",  # ends in "test" but not a test file
        "contest.py",  # contains "test" but not a test file
    ],
)
def test_classify_primary_material(filename: str) -> None:
    assert classify_source_role(filename) == "primary"


def test_is_source_role() -> None:
    assert is_source_role("primary")
    assert is_source_role("test")
    assert not is_source_role("bogus")
