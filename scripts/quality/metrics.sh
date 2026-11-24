#!/usr/bin/env bash
# Quality metrics ratchet: measures the repository and compares each metric with
# the value recorded in quality-metrics.properties on the base branch. A metric may only improve.
#
#   bash scripts/quality/metrics.sh [--update] [--only <metric>] [base-rev]
#
# Metrics: coverage.lines (higher is better; needs the coverage report from the test run),
# duplication.lines and deadcode.findings (lower is better). --update writes the measured values,
# which is how an improvement is recorded; CI rejects a recorded value that is worse than the base's.
# Identical in all SprintStart repos; the commands come from ratchet.conf.

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
. "$ROOT/scripts/quality/ratchet.conf"
METRICS_FILE=quality-metrics.properties
JSCPD_VERSION=5.4.0

update=0
only=""
base_arg=""
while [ $# -gt 0 ]; do
    case "$1" in
        --update) update=1 ;;
        --only) only=$2; shift ;;
        *) base_arg=$1 ;;
    esac
    shift
done

resolve_base() {
    local ref
    for ref in ${base_arg:-origin/dev origin/main}; do
        git rev-parse -q --verify "$ref" >/dev/null && { git merge-base HEAD "$ref"; return; }
    done
    git rev-parse HEAD
}
BASE=$(resolve_base)

measure_duplication() {
    local out
    out=$(mktemp -d)
    npx -y "jscpd@$JSCPD_VERSION" "${DUPLICATION_PATHS[@]}" --min-tokens 70 --reporters json \
        --output "$out" --silent --format "$DUPLICATION_FORMATS"         ${DUPLICATION_IGNORE:+--ignore "$DUPLICATION_IGNORE"} \
        ${DUPLICATION_IGNORE_PATTERN:+--ignore-pattern "$DUPLICATION_IGNORE_PATTERN"} >/dev/null 2>&1
    grep -oE '"duplicatedLines"[[:space:]]*:[[:space:]]*[0-9]+' "$out/jscpd-report.json" 2>/dev/null \
        | tail -1 | grep -oE '[0-9]+$'
    rm -rf "$out"
}

measure_deadcode() {
    [ -z "${DEADCODE_CMD:-}" ] && return
    bash -c "$DEADCODE_CMD"
}

measure_coverage() {
    [ -f "$COVERAGE_REPORT" ] || return
    case "$COVERAGE_FORMAT" in
        jacoco) grep -oE '<counter type="LINE" missed="[0-9]+" covered="[0-9]+"/>' "$COVERAGE_REPORT" | tail -1 \
                    | grep -oE '[0-9]+' | paste -sd' ' | awk '{ printf "%.1f", 100 * $2 / ($1 + $2) }' ;;
        cobertura) grep -m1 -oE 'line-rate="[0-9.]+"' "$COVERAGE_REPORT" | grep -oE '[0-9.]+' \
                    | awk '{ printf "%.1f", 100 * $1 }' ;;
        lcov) awk -F: '/^LF:/ { f += $2 } /^LH:/ { h += $2 } END { if (f) printf "%.1f", 100 * h / f }' "$COVERAGE_REPORT" ;;
    esac
}

recorded() {
    git show "$BASE:$METRICS_FILE" 2>/dev/null | grep -E "^$1=" | cut -d= -f2
}

# Exit status 1 when $2 (measured) is worse than $3 (recorded) for direction $1.
worse() {
    awk -v dir="$1" -v now="$2" -v was="$3" 'BEGIN { exit !((dir == "up" && now < was) || (dir == "down" && now > was)) }'
}

write_value() {
    touch "$METRICS_FILE"
    grep -vE "^$1=" "$METRICS_FILE" > "$METRICS_FILE.tmp"
    echo "$1=$2" >> "$METRICS_FILE.tmp"
    sort "$METRICS_FILE.tmp" > "$METRICS_FILE" && rm "$METRICS_FILE.tmp"
}

failures=0
check() {
    local name=$1 direction=$2 value=$3 was
    [ -n "$only" ] && [ "$only" != "$name" ] && return
    if [ -z "$value" ]; then
        echo "SKIP $name: not measured here"
        return
    fi
    was=$(recorded "$name")
    if [ "$update" = 1 ]; then
        write_value "$name" "$value"
        echo "SET  $name=$value (base: ${was:-none})"
    elif [ -z "$was" ]; then
        echo "NEW  $name=$value (no value recorded on the base branch yet)"
    elif worse "$direction" "$value" "$was"; then
        echo "FAIL $name: $was -> $value (may only get better)"
        failures=$((failures + 1))
    else
        echo "OK   $name: $was -> $value"
    fi
}

echo "Quality metrics against $(git rev-parse --short "$BASE")"
[ -z "$only" ] || [ "$only" = coverage.lines ] && check coverage.lines up "$(measure_coverage)"
[ -z "$only" ] || [ "$only" = duplication.lines ] && check duplication.lines down "$(measure_duplication)"
[ -z "$only" ] || [ "$only" = deadcode.findings ] && check deadcode.findings down "$(measure_deadcode)"

# A recorded value must not be worse than the base's either: --update cannot be used to lower the bar.
if [ "$update" = 0 ] && [ -f "$METRICS_FILE" ]; then
    while IFS='=' read -r name value; do
        was=$(recorded "$name")
        direction=down
        [ "$name" = coverage.lines ] && direction=up
        if [ -n "$was" ] && worse "$direction" "$value" "$was"; then
            echo "FAIL $METRICS_FILE lowers the bar for $name: $was -> $value"
            failures=$((failures + 1))
        fi
    done < "$METRICS_FILE"
fi

[ "$failures" -eq 0 ] || exit 1
