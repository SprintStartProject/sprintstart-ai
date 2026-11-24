#!/usr/bin/env bash
# Mutation gate: mutates the source files a branch changed and fails when the tests
# kill fewer than MUTATION_MIN percent of the mutants. A test that stays green while the code under it
# is broken does not test it — this is the check that catches that.
#
#   bash scripts/quality/mutation.sh [base-rev]   # the files a branch adds code to; fails below MUTATION_MIN
#   bash scripts/quality/mutation.sh --all        # every source file; reports the score, never fails
#
# Identical in all SprintStart repos. ratchet.conf provides MUTATION_FILES (pathspecs of mutable
# source files), MUTATION_MIN, run_mutation <files...> and mutation_counts (prints "killed total"), and
# either COMMENT_LINE_RE or its own added_code_lines.
#
# A changed file is mutated only when the branch adds code to it. A file whose diff only removes lines
# or touches comments adds no behaviour a test could miss, so a comment sweep or a deletion is not
# judged on tests it never touched.

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
. "$ROOT/scripts/quality/ratchet.conf"

# added_code_lines <base> <file>: the lines the branch adds to <file> that are code, i.e. neither blank
# nor matching COMMENT_LINE_RE. A repository whose comments need more than a line pattern (Python
# docstrings) defines its own in ratchet.conf.
if ! declare -F added_code_lines >/dev/null; then
    added_code_lines() {
        if git ls-files --error-unmatch -- "$2" >/dev/null 2>&1; then
            git diff -U0 "$1" -- "$2" | grep -E '^\+' | grep -vE '^\+\+\+ ' | cut -c2-
        else
            cat -- "$2"
        fi | grep -vE "${COMMENT_LINE_RE:-^[[:space:]]*$}" | grep -vE '^[[:space:]]*$'
    }
fi

all=0
if [ "${1:-}" = "--all" ]; then
    all=1
    shift
fi

base_ref=${1:-}
if [ -z "$base_ref" ]; then
    for ref in origin/dev origin/main; do
        git rev-parse -q --verify "$ref" >/dev/null && { base_ref=$ref; break; }
    done
fi
BASE=$(git merge-base HEAD "$base_ref")

if [ "$all" = 1 ]; then
    mapfile -t files < <(git ls-files -- "${MUTATION_FILES[@]}")
else
    mapfile -t files < <({ git diff --name-only --diff-filter=d "$BASE" -- "${MUTATION_FILES[@]}" 2>/dev/null
        git ls-files --others --exclude-standard -- "${MUTATION_FILES[@]}"; } | sort -u)
fi

if [ "$all" = 0 ]; then
    gated=()
    for file in "${files[@]}"; do
        if [ -n "$(added_code_lines "$BASE" "$file")" ]; then
            gated+=("$file")
        else
            echo "mutation: skipped $file (no added code)"
        fi
    done
    files=("${gated[@]}")
fi

if [ ${#files[@]} -eq 0 ]; then
    echo "mutation: no source files with added code against $base_ref"
    exit 0
fi

echo "mutation: ${#files[@]} file(s)"
[ "$all" = 1 ] || printf '    %s\n' "${files[@]}"
run_mutation "${files[@]}" || { echo "mutation: the mutation run itself failed"; exit 2; }

read -r killed total < <(mutation_counts)
if [ "${total:-0}" -eq 0 ]; then
    echo "mutation: no mutants were generated"
    exit 0
fi

score=$((100 * killed / total))
echo "mutation: $killed of $total mutants killed = $score% (minimum $MUTATION_MIN%)"
if [ "$all" = 0 ] && [ "$score" -lt "$MUTATION_MIN" ]; then
    echo "mutation: the tests do not notice enough broken code; strengthen their assertions"
    exit 1
fi
