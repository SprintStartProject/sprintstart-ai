#!/usr/bin/env bash
# Shared helpers for the Claude Code hooks in this folder.
# No jq: the hooks must run on any machine that has git and bash.

ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || exit 0
# shellcheck source=../quality/ratchet.conf
. "$ROOT/scripts/quality/ratchet.conf"

HOOK_INPUT=$(cat)

# First string value of a top-level or nested JSON key, unescaped enough for paths and commands.
json_field() {
    printf '%s' "$HOOK_INPUT" \
        | grep -oE "\"$1\"[[:space:]]*:[[:space:]]*\"(\\\\.|[^\"\\\\])*\"" \
        | head -1 \
        | sed -E "s/^\"$1\"[[:space:]]*:[[:space:]]*\"//; s/\"$//; s/\\\\\\\\/\\\\/g; s/\\\\\"/\"/g"
}

# Repository-relative path with forward slashes, or empty when outside the repository.
repo_relative() {
    local path=${1//\\//}
    local root_lc=${ROOT,,} path_lc=${path,,}
    case "$path_lc" in
        "$root_lc"/*) printf '%s' "${path:${#ROOT}+1}" ;;
        /*|?:/*) ;;
        *) printf '%s' "$path" ;;
    esac
}

starts_with_any() {
    local rel=$1 prefix
    shift
    for prefix in "$@"; do
        case "$prefix" in ':(exclude)'*) continue ;; esac
        prefix=${prefix#:(glob)}
        # Unquoted on purpose: the pathspec may hold a glob such as tsconfig*.json.
        [[ $rel == $prefix || $rel == $prefix/* ]] && return 0
    done
    return 1
}

base_rev() {
    local ref
    for ref in origin/dev origin/main; do
        git rev-parse -q --verify "$ref" >/dev/null && { git merge-base HEAD "$ref"; return; }
    done
    git rev-parse HEAD
}

# Matches of a regex in a file now versus at the base revision.
count_in_file() {
    local rel=$1 regex=$2 where=$3
    if [ "$where" = "worktree" ]; then
        grep -cE -- "$regex" "$ROOT/$rel" 2>/dev/null || true
    else
        git show "$where:$rel" 2>/dev/null | grep -cE -- "$regex" || true
    fi
}
