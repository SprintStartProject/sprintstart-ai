#!/usr/bin/env bash
# Quality ratchet: compares the working tree with a base revision
# and fails when the code got worse on a measure that may only improve.
#
#   bash scripts/quality/ratchet.sh [base-rev]
#
# base-rev defaults to the merge-base with origin/dev. Patterns and paths come from
# ratchet.conf next to this script; this file is identical in all SprintStart repos.
# Set QUALITY_EXCEPTION=1 to downgrade protected-config failures to warnings; CI does
# that only when a human other than the author labelled the PR `quality-exception`.

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
# shellcheck source=ratchet.conf
. "$ROOT/scripts/quality/ratchet.conf"

resolve_base() {
    if [ -n "${1:-}" ]; then
        git merge-base HEAD "$1" 2>/dev/null || git rev-parse "$1"
        return
    fi
    local ref
    for ref in origin/dev origin/main; do
        if git rev-parse -q --verify "$ref" >/dev/null; then
            git merge-base HEAD "$ref"
            return
        fi
    done
    git rev-parse HEAD
}

BASE=$(resolve_base "${1:-}") || { echo "ratchet: cannot resolve base revision" >&2; exit 2; }
failures=0
warnings=0

fail() { echo "FAIL $*"; failures=$((failures + 1)); }
warn() { echo "WARN $*"; warnings=$((warnings + 1)); }

# Sum of matching lines for an extended regex: in BASE when $1 is a revision, else in the
# working tree including untracked (but not ignored) files.
count_matches() {
    local where=$1 regex=$2
    shift 2
    if [ "$where" = "worktree" ]; then
        git grep --untracked -c -E "$regex" -- "$@" 2>/dev/null
    else
        git grep -c -E "$regex" "$where" -- "$@" 2>/dev/null
    fi | awk -F: '{ s += $NF } END { print s + 0 }'
}

# Files that differ from BASE, plus new untracked files, limited to the given pathspecs.
changed_files() {
    { git diff --name-only --diff-filter=d "$BASE" -- "$@" 2>/dev/null
      git ls-files --others --exclude-standard -- "$@"; } | sort -u
}

# Added lines matching a regex, as "file: line", for showing where a count went up.
added_lines() {
    local regex=$1
    shift
    # awk only labels each added line with its file; grep does the matching, because awk's
    # regex dialect has no \b.
    git diff -U0 "$BASE" -- "$@" 2>/dev/null | awk '
        /^\+\+\+ b\// { file = substr($0, 7); next }
        /^\+/ { print "    " file ": " substr($0, 2) }' | grep -E -- "$regex"
    git ls-files --others --exclude-standard -- "$@" | while read -r f; do
        grep -nE "$regex" "$f" 2>/dev/null | sed "s|^|    $f:|"
    done
}

ratchet_count() {
    local label=$1 regex=$2
    shift 2
    [ -z "$regex" ] && return
    local before after
    before=$(count_matches "$BASE" "$regex" "$@")
    after=$(count_matches worktree "$regex" "$@")
    if [ "$after" -gt "$before" ]; then
        fail "$label: $before -> $after (may not rise)"
        added_lines "$regex" "$@" | head -20
    fi
}

check_protected() {
    [ ${#PROTECTED[@]} -eq 0 ] && return
    local touched
    touched=$(changed_files "${PROTECTED[@]}")
    [ -z "$touched" ] && return
    if [ "${QUALITY_EXCEPTION:-0}" = "1" ]; then
        warn "Protected config changed under a quality-exception:"
    else
        fail "Protected config changed (needs a human decision and the quality-exception label):"
    fi
    echo "$touched" | sed 's/^/    /'
}

baseline_entries() {
    local where=$1 file=$2 content
    if [ "$where" = "worktree" ]; then
        [ -f "$file" ] || { echo 0; return; }
        content=$(cat "$file")
    else
        content=$(git show "$where:$file" 2>/dev/null) || { echo 0; return; }
    fi
    case "$file" in
        *.xml) printf '%s' "$content" | grep -c '<ID>' ;;
        *known-violations.json) printf '%s' "$content" | grep -c '"from"' ;;
        *.json) printf '%s' "$content" | grep -oE '"count"[[:space:]]*:[[:space:]]*[0-9]+' \
                    | grep -oE '[0-9]+$' | awk '{ s += $1 } END { print s + 0 }' ;;
        *.txt) printf '%s\n' "$content" | grep -cvE '^[[:space:]]*(#|$)' ;;
        *.importlinter) printf '%s\n' "$content" | grep -cE '^[[:space:]]+[A-Za-z_.]+ -> [A-Za-z_.*]+' ;;
        *) echo 0 ;;
    esac
}

check_baselines() {
    local file before after
    for file in "${LINT_BASELINES[@]}"; do
        # A baseline the base branch does not have yet is being introduced; only later changes are held to it.
        git cat-file -e "$BASE:$file" 2>/dev/null || continue
        before=$(baseline_entries "$BASE" "$file")
        after=$(baseline_entries worktree "$file")
        if [ "$after" -gt "$before" ]; then
            fail "Lint baseline $file grew: $before -> $after entries (may only shrink)"
        fi
    done
}

# The path a file had at BASE: its own, or the one it was moved from, so a move keeps its history.
base_path() {
    local renamed
    renamed=$(git diff -M --name-status --diff-filter=R "$BASE" 2>/dev/null | awk -F'\t' -v f="$1" '$3 == f { print $2; exit }')
    echo "${renamed:-$1}"
}

# A file's length on stdin: its lines, without those SIZE_IGNORE_RE matches.
counted_lines() {
    if [ -n "${SIZE_IGNORE_RE:-}" ]; then
        grep -cvE -- "$SIZE_IGNORE_RE"
    else
        wc -l
    fi
}

check_file_length() {
    local f before after
    while read -r f; do
        [ -z "$f" ] && continue
        after=$(counted_lines < "$f")
        [ "$after" -le "$FILE_LIMIT" ] && continue
        before=$(git show "$BASE:$(base_path "$f")" 2>/dev/null | counted_lines)
        if [ "$after" -gt "$before" ]; then
            fail "$f has $after lines${SIZE_IGNORE_RE:+ besides /$SIZE_IGNORE_RE/} (limit $FILE_LIMIT, was $before; may not grow past the limit)"
        fi
    done < <(changed_files "${SIZE_PATHS[@]}")
}

check_history_comments() {
    [ -z "$HISTORY_RE" ] && return
    local hits
    hits=$(added_lines "$HISTORY_RE" "${SOURCE_PATHS[@]}" | head -15)
    [ -z "$hits" ] && return
    warn "Comments that narrate history (move them to the commit message, PR or ADR):"
    echo "$hits"
}

check_rule_citations() {
    [ -z "${RULE_CITATION_RE:-}" ] && return
    local hits
    hits=$(added_lines "$RULE_CITATION_RE" "${SOURCE_PATHS[@]}" "${TEST_PATHS[@]}" | head -15)
    [ -z "$hits" ] && return
    fail "Comments that cite the coding rules (delete the citation):"
    echo "$hits"
}

echo "Quality ratchet against $(git rev-parse --short "$BASE")"
check_protected
ratchet_count "suppressions" "$SUPPRESSION_RE" "${SOURCE_PATHS[@]}" "${TEST_PATHS[@]}"
ratchet_count "skipped or focused tests" "$SKIP_RE" "${TEST_PATHS[@]}"
[ ${#RAWID_PATHS[@]} -gt 0 ] && ratchet_count "raw-typed ids in signatures" "$RAWID_RE" "${RAWID_PATHS[@]}"
ratchet_count "hidden now()/random/new-id calls" "${HIDDEN_DEPS_RE:-}" "${SOURCE_PATHS[@]}"
check_baselines
check_file_length
check_history_comments
check_rule_citations

if [ "$failures" -gt 0 ]; then
    echo "ratchet: $failures failure(s), $warnings warning(s)"
    exit 1
fi
echo "ratchet: ok ($warnings warning(s))"
