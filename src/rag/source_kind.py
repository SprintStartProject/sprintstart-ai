"""Classify a source file as test/fixture material by its path — language-agnostic.

Quoting a fixture as authoritative is the worst thing this service does: a
"Code Review SLA" that lives in a ``tests/.../demo-corpus/`` fixture is example
data, and a new hire has no way to tell.

Recognition keys only on naming conventions — a test directory in the path, or a
conventional test/spec suffix — so it works for any repo or language with no
configuration. ⚠️ Filename patterns are tied to known code extensions so data and
config files (``openapi_spec.yaml``, ``manifest.json``) are never misread as
tests.

⚠️ **A retrieved chunk does not carry its path.** Chroma metadata holds only the
*basename* (``process.md``), which contains no ``tests/`` segment — so
:func:`is_test_source` on a chunk's ``filename`` answers False for exactly the
fixtures this exists to catch, while still catching ``test_chunker.py`` on its
filename pattern. The path *is* recoverable from the chunk's ``source_url``,
which is what :func:`is_test_chunk` does — **use that for chunks.**
"""

import re
from urllib.parse import unquote, urlparse

# Path *directory* components that mark everything beneath them as test/fixture
# material, across ecosystems (pytest, jest, go, maven/gradle, rspec, ...).
_TEST_DIR_SEGMENTS = frozenset(
    {
        "test",
        "tests",
        "__tests__",
        "__mocks__",
        "__snapshots__",
        "snapshots",
        "testdata",
        "test-data",
        "test_data",
        "fixture",
        "fixtures",
        "mocks",
        "e2e",
        "testing",
    }
)

# Extensions that carry executable test code, so a test/spec suffix on one is a
# real test and not a coincidentally-named document or data file.
_CODE_EXT = (
    r"(?:py|rb|go|js|jsx|ts|tsx|mjs|cjs|java|kt|kts|scala|groovy|cs|swift|php|"
    r"rs|dart|ex|exs|clj|c|cc|cpp|h|hpp|m|mm)"
)

# File *names* that are tests by convention even when they sit next to the code
# they exercise (Go's ``_test.go``, co-located ``foo.test.ts``, xUnit classes).
_TEST_FILENAME_RE = re.compile(
    r"^conftest\.py$"
    r"|^test_.+\.(?:py|rb)$"
    rf"|.+_test\.{_CODE_EXT}$"
    r"|.+_spec\.rb$"
    r"|.+\.(?:test|spec)\.(?:js|jsx|ts|tsx|mjs|cjs)$"
    r"|.+(?:Test|Tests|Spec|IT)\.(?:java|kt|kts|scala|groovy|cs|swift|php)$"
)


def is_test_source(path: str | None) -> bool:
    """True when ``path`` names a test, fixture, mock or sample-data file.

    Recognises a test directory anywhere in the path, or a conventional test/spec
    filename suffix. Case-insensitive on directory names; filename suffixes keep
    their conventional casing (``FooTest.java`` is a test, ``footest.md`` is not).
    """
    if not path:
        return False
    parts = [segment for segment in path.replace("\\", "/").split("/") if segment]
    if not parts:
        return False
    if any(segment.lower() in _TEST_DIR_SEGMENTS for segment in parts[:-1]):
        return True
    return _TEST_FILENAME_RE.match(parts[-1]) is not None


# URL segments after which a forge serves a *repository-relative* path:
# `.../blob/<ref>/src/main.py`, GitLab's `.../-/raw/<ref>/...`, and so on. Only a
# path recovered this way is trusted, because only then is it certainly a file in
# a repository. ⚠️ An arbitrary URL that happens to contain "/tests/" is left
# alone: a false positive means a real document goes unquoted, which is the same
# failure this module exists to prevent, pointing the other way.
_PATH_MARKERS = frozenset({"blob", "raw", "tree", "blame", "src"})


def repo_path_from_url(source_url: str | None) -> str | None:
    """The repository-relative path a forge URL points at, or None if it isn't one.

    ``https://github.com/o/r/blob/<sha>/tests/rag/demo-corpus/process.md`` yields
    ``tests/rag/demo-corpus/process.md``. Anything without a recognisable
    ``blob``/``raw``/``tree`` marker — a Confluence page, a bare link — yields
    None, so nothing infers a directory structure from a URL that has none.
    """
    if not source_url:
        return None
    try:
        parts = [p for p in unquote(urlparse(source_url).path).split("/") if p]
    except ValueError:
        return None
    for index, segment in enumerate(parts):
        # The marker is followed by a ref (branch or sha), then the path itself.
        if segment in _PATH_MARKERS and len(parts) > index + 2:
            return "/".join(parts[index + 2 :])
    return None


def is_test_chunk(
    filename: str | None,
    source_url: str | None = None,
    source_role: str | None = None,
) -> bool:
    """True when a retrieved chunk is test, fixture, mock or sample material.

    Checks everything that could know, cheapest first, because each source is
    blind in a different place:

    * ``source_role`` is decided at ingest from a **basename**, so it is right
      about ``test_chunker.py`` and wrong about ``process.md``;
    * the chunk's ``filename`` is that same basename, with the same blind spot;
    * ``source_url`` is the only one carrying the directory the file lives in,
      which is how a fixture under ``tests/`` is finally recognisable.

    Any of the three saying "test" is enough. They do not disagree so much as
    each see part of it.
    """
    if source_role == "test":
        return True
    if is_test_source(repo_path_from_url(source_url)):
        return True
    return is_test_source(filename)


__all__ = ["is_test_chunk", "is_test_source", "repo_path_from_url"]
