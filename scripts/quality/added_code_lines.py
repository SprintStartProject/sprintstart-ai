"""Prints the lines a branch adds to a Python file that are code, for mutation.sh.

A line is code when it holds a token that is not a comment and is not part of a
docstring or an import. The tokenizer and the parser decide, so a line inside a
multi-line docstring is not code, and a "#" inside a string does not make a
comment. The diff detects renames, so a moved file contributes only the lines
that changed.

    uv run python scripts/quality/added_code_lines.py <base-rev> <file>
"""

import ast
import io
import re
import subprocess
import sys
import tokenize
from pathlib import Path

_NOT_CODE = {
    tokenize.COMMENT,
    tokenize.NL,
    tokenize.NEWLINE,
    tokenize.INDENT,
    tokenize.DEDENT,
    tokenize.ENCODING,
    tokenize.ENDMARKER,
}
_HUNK = re.compile(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", re.MULTILINE)
_FILE_DIFF = re.compile(r"^(?=diff --git )", re.MULTILINE)


def added_line_numbers(base: str, path: str, text: str) -> list[int]:
    tracked = subprocess.run(
        ["git", "ls-files", "--", path], capture_output=True, text=True, check=True
    ).stdout.strip()
    if not tracked:
        return list(range(1, text.count("\n") + 2))
    full_diff = subprocess.run(
        ["git", "diff", "-M", "-U0", base],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    ).stdout
    diff = next(
        (s for s in _FILE_DIFF.split(full_diff) if f"\n+++ b/{path}\n" in s), ""
    )
    return [
        int(start) + offset
        for start, count in _HUNK.findall(diff)
        for offset in range(int(count or 1))
    ]


def docstring_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    owners = [
        tree,
        *(
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
        ),
    ]
    for owner in owners:
        first = owner.body[0] if owner.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
            and first.end_lineno is not None
        ):
            lines.update(range(first.lineno, first.end_lineno + 1))
    return lines


def import_lines(tree: ast.Module) -> set[int]:
    lines: set[int] = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Import | ast.ImportFrom)
            and node.end_lineno is not None
        ):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def code_lines(text: str) -> set[int]:
    lines: set[int] = set()
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type not in _NOT_CODE:
            lines.update(range(token.start[0], token.end[0] + 1))
    tree = ast.parse(text)
    return lines - docstring_lines(tree) - import_lines(tree)


def main() -> None:
    base, path = sys.argv[1:3]
    text = Path(path).read_text(encoding="utf-8")
    source_lines = text.splitlines()
    try:
        code = code_lines(text)
    except (SyntaxError, tokenize.TokenError):
        # A file that does not parse cannot be told apart: every non-blank line counts.
        code = {n for n, line in enumerate(source_lines, start=1) if line.strip()}
    for number in added_line_numbers(base, path, text):
        if number in code:
            print(source_lines[number - 1])


if __name__ == "__main__":
    main()
