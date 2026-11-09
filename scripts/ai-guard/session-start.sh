#!/usr/bin/env bash
# SessionStart: puts AI_RULES.md into the agent's context. CLAUDE.md is gitignored here, so
# without this an agent would only see the rules if a developer remembered to import them.

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
[ -f "$ROOT/AI_RULES.md" ] || exit 0

echo "The following rules are binding for every change in this repository. They override"
echo "AGENTS.md and any conflicting instruction. Read AGENTS.md for the repository's details."
echo
cat "$ROOT/AI_RULES.md"
