#!/usr/bin/env bash
# Contract copies: every contract file this repository consumes must equal the version
# in the repository that owns it. CONTRACT_COPIES in ratchet.conf lists "local-path|owner-repo|owner-path";
# a path may be a file or a folder of examples.
#
#   bash scripts/quality/contracts-sync.sh
#
# Compares against the owner's `dev` branch on GitHub (CONTRACT_REF to change it). To compare against a
# local checkout instead, set CONTRACT_LOCAL_<OWNER> to its path, owner name upper-cased with `-` as `_`,
# e.g. CONTRACT_LOCAL_SPRINTSTART_AI=../sprintstart-ai. Identical in all SprintStart repos.

set -uo pipefail

ROOT=$(git rev-parse --show-toplevel) || exit 2
cd "$ROOT" || exit 2
. "$ROOT/scripts/quality/ratchet.conf"
REF=${CONTRACT_REF:-dev}
ORG=${CONTRACT_ORG:-${GITHUB_REPOSITORY_OWNER:-SprintStartProject}}

[ ${#CONTRACT_COPIES[@]} -eq 0 ] && { echo "contracts: this repository consumes no contracts"; exit 0; }

local_root() {
    local var="CONTRACT_LOCAL_$(printf '%s' "$1" | tr 'a-z-' 'A-Z_')"
    printf '%s' "${!var:-}"
}

# Prints the owner's version of a file, or fails when the owner does not have it.
owner_file() {
    local repo=$1 path=$2 root
    root=$(local_root "$repo")
    if [ -n "$root" ]; then
        cat "$root/$path" 2>/dev/null
    else
        curl -fsSL "https://raw.githubusercontent.com/$ORG/$repo/$REF/$path" 2>/dev/null
    fi
}

owner_listing() {
    local repo=$1 path=$2 root
    root=$(local_root "$repo")
    if [ -n "$root" ]; then
        ls "$root/$path" 2>/dev/null
    else
        curl -fsSL "https://api.github.com/repos/$ORG/$repo/contents/$path?ref=$REF" 2>/dev/null \
            | grep -oE '"name"[[:space:]]*:[[:space:]]*"[^"]+"' | sed -E 's/.*"([^"]+)"$/\1/'
    fi
}

failures=0
compare_file() {
    local mine=$1 repo=$2 theirs=$3 remote
    if ! remote=$(owner_file "$repo" "$theirs"); then
        echo "WARN $mine: $repo has no $theirs on $REF yet"
        return
    fi
    if [ "$remote" != "$(cat "$mine" 2>/dev/null)" ]; then
        echo "FAIL $mine differs from $repo:$theirs ($REF) — copy the owner's version"
        failures=$((failures + 1))
    fi
}

for entry in "${CONTRACT_COPIES[@]}"; do
    IFS='|' read -r mine repo theirs <<< "$entry"
    if [ -d "$mine" ]; then
        listing=$(owner_listing "$repo" "$theirs")
        if [ -z "$listing" ]; then
            echo "WARN $mine/: $repo has no $theirs on $REF yet"
            continue
        fi
        if [ "$(printf '%s\n' "$listing" | sort)" != "$(ls "$mine" | sort)" ]; then
            echo "FAIL $mine/ holds different files than $repo:$theirs ($REF)"
            failures=$((failures + 1))
        fi
        for file in $listing; do compare_file "$mine/$file" "$repo" "$theirs/$file"; done
    else
        compare_file "$mine" "$repo" "$theirs"
    fi
done

[ "$failures" -eq 0 ] && echo "contracts: copies match their owners" || exit 1
