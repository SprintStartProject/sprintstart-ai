#!/usr/bin/env bash
# Applies .github/rulesets/*.json to a repository on GitHub: creates each ruleset, or
# updates the one with the same name. Needs admin rights and changes how everybody merges — run it only
# after the team agreed. Existing rulesets are left alone; GitHub applies the strictest rule of all.
#
#   bash scripts/quality/apply-rulesets.sh                          # show what would change
#   bash scripts/quality/apply-rulesets.sh --apply                  # change it
#   bash scripts/quality/apply-rulesets.sh --repo owner/name --solo --apply
#
# --repo defaults to the `origin` remote. --solo is for a repository only one person works in (a
# personal fork): every check stays required, but no second approval or code-owner review is asked for,
# because nobody could give it. Identical in all SprintStart repos.

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2

apply=0
solo=0
repo=$(git remote get-url origin | sed -E 's#^.*github\.com[:/]##; s#\.git$##')
while [ $# -gt 0 ]; do
    case "$1" in
        --apply) apply=1 ;;
        --solo) solo=1 ;;
        --repo) repo=$2; shift ;;
        *) echo "unknown option $1"; exit 2 ;;
    esac
    shift
done

# The solo variant of a ruleset: same checks, no approvals that a single person cannot give.
solo_variant() {
    sed -E 's/"required_approving_review_count": [0-9]+/"required_approving_review_count": 0/;
            s/"require_code_owner_review": true/"require_code_owner_review": false/;
            s/"require_last_push_approval": true/"require_last_push_approval": false/;
            s/"dismiss_stale_reviews_on_push": true/"dismiss_stale_reviews_on_push": false/' "$1"
}

for file in .github/rulesets/*.json; do
    name=$(grep -m1 -oE '"name"[[:space:]]*:[[:space:]]*"[^"]+"' "$file" | sed -E 's/.*"([^"]+)"$/\1/')
    body=$(if [ "$solo" = 1 ]; then solo_variant "$file"; else cat "$file"; fi)
    id=$(gh api "repos/$repo/rulesets" --jq ".[] | select(.name == \"$name\") | .id")
    if [ "$apply" = 0 ]; then
        echo "$repo: would ${id:+update}${id:-create} ruleset '$name'$([ "$solo" = 1 ] && echo ' (solo)')"
    elif [ -n "$id" ]; then
        printf '%s' "$body" | gh api -X PUT "repos/$repo/rulesets/$id" --input - >/dev/null && echo "$repo: updated '$name'"
    else
        printf '%s' "$body" | gh api -X POST "repos/$repo/rulesets" --input - >/dev/null && echo "$repo: created '$name'"
    fi
done
