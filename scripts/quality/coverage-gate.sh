#!/usr/bin/env bash
# Coverage gate: run after the tests have written their coverage report.
#   1. Lines changed against the base branch must be covered to at least DIFF_COVERAGE_MIN percent.
#   2. Total line coverage may not drop below the value recorded on the base branch.
#
#   bash scripts/quality/coverage-gate.sh [base-rev]
#
# Identical in all SprintStart repos; report path and format come from ratchet.conf.

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
. "$ROOT/scripts/quality/ratchet.conf"
DIFF_COVER_VERSION=10.6.0

if [ ! -f "$COVERAGE_REPORT" ]; then
    echo "coverage-gate: $COVERAGE_REPORT is missing — run the tests with coverage first"
    exit 2
fi

base_ref=${1:-}
if [ -z "$base_ref" ]; then
    for ref in origin/dev origin/main; do
        git rev-parse -q --verify "$ref" >/dev/null && { base_ref=$ref; break; }
    done
fi

diff_cover() {
    if command -v diff-cover >/dev/null; then
        diff-cover "$@"
    elif command -v uvx >/dev/null; then
        uvx --quiet "diff-cover==$DIFF_COVER_VERSION" "$@"
    else
        pipx run --quiet "diff-cover==$DIFF_COVER_VERSION" "$@"
    fi
}

status=0
echo "Diff coverage against $base_ref (minimum $DIFF_COVERAGE_MIN%)"
# COVERAGE_SRC_ROOTS is unquoted on purpose: it may hold several roots.
diff_cover "$COVERAGE_REPORT" --compare-branch="$base_ref" --fail-under="$DIFF_COVERAGE_MIN" \
    ${COVERAGE_SRC_ROOTS:+--src-roots $COVERAGE_SRC_ROOTS} || status=1

bash "$ROOT/scripts/quality/metrics.sh" --only coverage.lines "$base_ref" || status=1
exit $status
