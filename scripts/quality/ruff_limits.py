"""Checks the limits in ruff-limits.toml against a baseline.

    uv run python scripts/quality/ruff_limits.py           # fail on new violations
    uv run python scripts/quality/ruff_limits.py --update  # rewrite, after fixes only

Violations are keyed by file, function and rule, not by line, so an edit elsewhere in
a file keeps the baseline valid. The baseline has eslint's suppressions shape, so
scripts/quality/ratchet.sh counts both the same way and fails when it grows.
"""

import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "ruff-limits.toml"
BASELINE = ROOT / "ruff-limits-baseline.json"
DEF_NAME = re.compile(r"^\s*(?:async\s+)?def\s+(\w+)")

Key = tuple[str, str]
Baseline = dict[str, dict[str, dict[str, int]]]


class Location(TypedDict):
    row: int


class Finding(TypedDict):
    filename: str
    code: str
    location: Location


def find_violations() -> Counter[Key]:
    command = ["ruff", "check", "--config", str(CONFIG), "--output-format", "json"]
    command += ["--no-cache", "--exit-zero", "src"]
    result = subprocess.run(
        command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True
    )
    findings: list[Finding] = json.loads(result.stdout)
    return Counter(_key(finding) for finding in findings)


def _key(finding: Finding) -> Key:
    path = Path(finding["filename"]).resolve().relative_to(ROOT).as_posix()
    function = _enclosing_name(ROOT / path, finding["location"]["row"])
    return f"{path}::{function}", finding["code"]


def _enclosing_name(path: Path, row: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in reversed(lines[:row]):
        match = DEF_NAME.match(line)
        if match:
            return match.group(1)
    return "<module>"


def load_baseline() -> Counter[Key]:
    if not BASELINE.exists():
        return Counter()
    data: Baseline = json.loads(BASELINE.read_text("utf-8"))
    return Counter(
        {
            (where, code): entry["count"]
            for where, rules in data.items()
            for code, entry in rules.items()
        }
    )


def write_baseline(violations: Counter[Key]) -> None:
    data: Baseline = {}
    for (where, code), count in sorted(violations.items()):
        data.setdefault(where, {})[code] = {"count": count}
    BASELINE.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def report(violations: Counter[Key], baseline: Counter[Key]) -> int:
    new = violations - baseline
    for (where, code), count in sorted(new.items()):
        print(f"FAIL {code} in {where} ({count} new)")
    fixed = sum((baseline - violations).values())
    if fixed:
        print(f"ruff-limits: {fixed} baseline entries fixed; run --update to drop them")
    if new:
        print("ruff-limits: restructure the code; never grow the baseline")
        return 1
    print(f"ruff-limits: ok ({sum(violations.values())} known violations left)")
    return 0


def main() -> int:
    violations = find_violations()
    if "--update" in sys.argv:
        write_baseline(violations)
        print(f"ruff-limits: baseline written, {sum(violations.values())} entries")
        return 0
    return report(violations, load_baseline())


if __name__ == "__main__":
    sys.exit(main())
