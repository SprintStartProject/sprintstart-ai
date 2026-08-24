"""Tests for the language-agnostic test/fixture source classifier."""

import pytest

from rag.source_kind import is_test_chunk, is_test_source, repo_path_from_url


@pytest.mark.parametrize(
    "path",
    [
        # The case that started this: a fixture under a tests/ tree.
        "tests/rag/demo-corpus/process.md",
        "tests/insights/test_faq.py",
        # Test directories, various ecosystems and depths.
        "test/models/user_model.rb",
        "src/__tests__/app.js",
        "app/__mocks__/api.ts",
        "backend/src/testData/sample.json",
        "e2e/checkout.spec.ts",
        "src/components/__snapshots__/Button.snap",
        "web/fixtures/users.yaml",
        # Conventional filenames sitting next to the code they exercise.
        "internal/server/server_test.go",
        "pkg/util/util_test.py",
        "src/components/Button.test.tsx",
        "src/components/Button.spec.js",
        "src/main/java/com/acme/UserServiceTest.java",
        "src/test/kotlin/com/acme/UserServiceSpec.kt",
        "it/com/acme/CheckoutIT.java",
        "spec/models/user_spec.rb",
        "conftest.py",
        "tests/conftest.py",
        # Windows-style separators still classify.
        "tests\\api\\test_client.py",
    ],
)
def test_recognises_test_and_fixture_files(path: str):
    assert is_test_source(path) is True


@pytest.mark.parametrize(
    "path",
    [
        # Real source and docs.
        "src/onboarding/checks.py",
        "AGENTS.md",
        "docs/process.md",
        "README.md",
        "src/main/java/com/acme/UserService.java",
        "internal/server/server.go",
        # Data/config that merely contains "spec"/"test" — must not false-positive.
        "openapi_spec.yaml",
        "config/manifest.json",
        "docs/testing-guide.md",
        "src/latest.js",
        # Commit artifacts the ingester produces.
        "commit-7766d14af83f8df82de1e7667a629e1914626bbd.md",
        "",
        None,
    ],
)
def test_leaves_real_sources_alone(path: str | None):
    assert is_test_source(path) is False


# --- recovering the path a chunk lost -------------------------------------------
#
# The classifier above was always right; it was never shown a path. A retrieved
# chunk carries only its basename, and `process.md` contains no `tests/` segment --
# which is why the fixture that started this went unmarked while `test_chunker.py`
# was caught on its filename, making it look like the model ignored a label.


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "https://github.com/o/r/blob/7766d14/tests/rag/demo-corpus/process.md",
            "tests/rag/demo-corpus/process.md",
        ),
        ("https://github.com/o/r/blob/main/src/app.py", "src/app.py"),
        (
            "https://gitlab.com/o/r/-/raw/main/tests/fixtures/data.json",
            "tests/fixtures/data.json",
        ),
        ("https://github.com/o/r/tree/main/docs/guide.md", "docs/guide.md"),
        # Percent-encoding is undone, or a path with a space never matches.
        ("https://github.com/o/r/blob/main/docs/my%20guide.md", "docs/my guide.md"),
        # No forge marker: nothing is inferred rather than guessed at.
        ("https://wiki.example.com/spaces/TEAM/pages/12345/Process", None),
        ("https://example.com/tests/whatever", None),
        ("not a url at all", None),
        ("", None),
        (None, None),
    ],
)
def test_recovers_a_repository_path_only_when_the_url_really_is_one(
    url: str | None, expected: str | None
):
    assert repo_path_from_url(url) == expected


def test_the_fixture_that_started_this_is_recognised_from_its_url():
    # The whole bug in one assertion: basename says no, the URL says yes.
    assert is_test_source("process.md") is False
    assert (
        is_test_chunk(
            "process.md",
            "https://github.com/o/r/blob/7766d14/tests/rag/demo-corpus/process.md",
        )
        is True
    )


def test_a_real_document_stays_quotable():
    url = "https://github.com/o/r/blob/main/AGENTS.md"
    assert is_test_chunk("AGENTS.md", url) is False


def test_any_one_signal_is_enough():
    # Each of the three is blind somewhere else, so they are combined, not ranked.
    assert is_test_chunk("test_chunker.py", None, None) is True
    assert is_test_chunk("data.json", None, "test") is True
    assert is_test_chunk("data.json", None, "primary") is False


def test_a_chunk_with_no_url_falls_back_to_its_basename():
    assert is_test_chunk("process.md", None, None) is False
    assert is_test_chunk("conftest.py", None, None) is True
