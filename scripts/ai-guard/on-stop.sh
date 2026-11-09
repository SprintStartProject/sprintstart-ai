#!/usr/bin/env bash
# Stop: runs the quality ratchet before the agent may finish its turn. Any failure or warning
# sends the agent back once; the second stop of the same turn is let through, so a finding the
# agent cannot fix ends as a report to the human instead of a loop.

. "$(dirname "$0")/lib.sh"

if printf '%s' "$HOOK_INPUT" | grep -qE '"stop_hook_active"[[:space:]]*:[[:space:]]*true'; then
    exit 0
fi

# Nothing changed on this branch: nothing to check.
base=$(base_rev)
if git diff --quiet "$base" -- 2>/dev/null && [ -z "$(git ls-files --others --exclude-standard)" ]; then
    exit 0
fi

output=$(bash "$ROOT/scripts/quality/ratchet.sh" 2>&1)
status=$?

if [ "$status" -ne 0 ] || printf '%s' "$output" | grep -q '^WARN'; then
    {
        echo "Before you finish: the quality ratchet reports problems."
        printf '%s\n' "$output"
        echo "Fix each FAIL. For each WARN, fix it or say in your report why it is fine."
        echo "If a rule blocks the task, stop and report it to the human."
    } >&2
    exit 2
fi
exit 0
