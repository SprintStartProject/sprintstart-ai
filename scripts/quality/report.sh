#!/usr/bin/env bash
# Quality report: the repository's quality numbers as one Markdown table, for
# the sprint review and to see whether the baselines shrink.
#
#   bash scripts/quality/report.sh > quality-report.md
#
# Reads what the other scripts measure; coverage and mutation appear when their reports exist (run the
# tests with coverage, or scripts/quality/mutation.sh, first). Identical in all SprintStart repos.

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
. "$ROOT/scripts/quality/ratchet.conf"

count() {
    local regex=$1
    shift
    [ -z "$regex" ] && { echo "–"; return; }
    git grep -c -E "$regex" -- "$@" 2>/dev/null | awk -F: '{ s += $NF } END { print s + 0 }'
}

entries() {
    case "$1" in
        *.xml) grep -c '<ID>' "$1" ;;
        *known-violations.json) grep -c '"from"' "$1" ;;
        *.json) grep -oE '"count"[[:space:]]*:[[:space:]]*[0-9]+' "$1" | grep -oE '[0-9]+$' \
                    | awk '{ s += $1 } END { print s + 0 }' ;;
        *.txt) grep -cvE '^[[:space:]]*(#|$)' "$1" ;;
        *.importlinter) grep -cE '^[[:space:]]+[A-Za-z_.]+ -> [A-Za-z_.*]+' "$1" ;;
    esac
}

oversized() {
    git ls-files -- "${SIZE_PATHS[@]}" | while read -r f; do
        [ "$(wc -l < "$f")" -gt "$FILE_LIMIT" ] && echo "$f"
    done | wc -l
}

metric() {
    grep -E "^$1=" quality-metrics.properties 2>/dev/null | cut -d= -f2
}

mutation_line() {
    read -r killed total < <(mutation_counts 2>/dev/null || echo "0 0")
    if [ "${total:-0}" -gt 0 ]; then
        echo "$((100 * killed / total))% ($killed of $total mutants killed, last run)"
    else
        echo "not run"
    fi
}

echo "# Quality report: $(basename "$ROOT")"
echo
echo "$(git rev-parse --abbrev-ref HEAD) at $(git rev-parse --short HEAD), $(date +%Y-%m-%d)"
echo
echo "| Measure | Value |"
echo "| --- | --- |"
echo "| Suppressions in code | $(count "$SUPPRESSION_RE" "${SOURCE_PATHS[@]}" "${TEST_PATHS[@]}") |"
echo "| Skipped or focused tests | $(count "$SKIP_RE" "${TEST_PATHS[@]}") |"
[ ${#RAWID_PATHS[@]} -gt 0 ] && echo "| Raw-typed ids in signatures (grep) | $(count "$RAWID_RE" "${RAWID_PATHS[@]}") |"
echo "| Hidden now()/random/new-id calls | $(count "${HIDDEN_DEPS_RE:-}" "${SOURCE_PATHS[@]}") |"
echo "| Files over $FILE_LIMIT lines | $(oversized) |"
for file in "${LINT_BASELINES[@]}"; do
    [ -f "$file" ] && echo "| Known violations: \`$(basename "$file")\` | $(entries "$file") |"
done
echo "| Line coverage | $(metric coverage.lines || true)% |"
echo "| Mutation score | $(mutation_line) |"
echo "| Duplicated lines (jscpd) | $(metric duplication.lines || true) |"
[ -n "${DEADCODE_CMD:-}" ] && echo "| Dead-code findings | $(metric deadcode.findings || true) |"
echo
echo "Every number here may only go down (coverage and mutation score: up)."
