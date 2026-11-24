#!/usr/bin/env bash
# PreToolUse (Edit|MultiEdit|Write|NotebookEdit|Bash): protected config and bypassed git hooks.
# - An edit to protected configuration is put to the human to decide ("ask").
# - Bypassing git hooks or rewriting shared history is refused ("deny").

. "$(dirname "$0")/lib.sh"

decide() {
    local decision=$1 reason=$2
    reason=${reason//\\/\\\\}
    reason=${reason//\"/\\\"}
    printf '{"hookSpecificOutput":{"hookEventName":"PreToolUse","permissionDecision":"%s","permissionDecisionReason":"%s"}}\n' \
        "$decision" "$reason"
    exit 0
}

tool=$(json_field tool_name)

if [ "$tool" = "Bash" ]; then
    cmd=$(json_field command)
    if printf '%s' "$cmd" | grep -qE -- '--no-verify|git[^|;&]*push[^|;&]*(-f\b|--force)'; then
        decide deny "Bypassing git hooks or force-pushing is not allowed for agents."
    fi
    for path in "${PROTECTED[@]}" "${LINT_BASELINES[@]}"; do
        if printf '%s' "$cmd" | grep -qF -- "$path" \
            && printf '%s' "$cmd" | grep -qE -- '(sed|perl)[^|;&]*-i|>|\btee\b|\bmv\b|\bcp\b|\brm\b|truncate|git[^|;&]*(checkout|restore)'; then
            decide ask "This command may change $path, which decides what is checked. Only approve if you, not the agent, decided this change."
        fi
    done
    exit 0
fi

rel=$(repo_relative "$(json_field file_path)")
[ -z "$rel" ] && rel=$(repo_relative "$(json_field notebook_path)")
[ -z "$rel" ] && exit 0

if starts_with_any "$rel" "${PROTECTED[@]}"; then
    decide ask "$rel decides what is checked. Only approve if you, not the agent, decided this change; the PR then needs the quality-exception label."
fi
if starts_with_any "$rel" "${LINT_BASELINES[@]}"; then
    decide ask "$rel is a lint baseline and may only shrink. Approve only if this edit removes entries."
fi
exit 0
