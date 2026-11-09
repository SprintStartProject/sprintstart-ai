# ADR (proposed) 003: `onboarding` splits into `buddy` and `assembly`

**Status:** Proposed (2026-10-22), awaiting confirmation.

## Context

`onboarding` is the largest package (4,829 lines) and holds two things that do not share code. The
Buddy files (`buddy_agent`, `buddy_open`, `buddy_compact`, `buddy_persona`, `query_fence`,
`vocabulary`, 1,800 lines) are a conversation with tools. The assembly files (`assembly`, `phase`,
`orientation`, `diagram`, `starter_work`, their models, `corpus`, `citations`, `progress`,
`similarity`, 3,029 lines) are staged, deterministic generation of grounded content. No file in one
group imports a file in the other (as-is §2.5).

The name `onboarding` also hides that the layers contract treats it as one feature, so it cannot
express that the two are independent.

Drivers: Q6 understandability, Q5 changeability.

## Options

1. **Keep one package** and document the two clusters.
2. **Split into `buddy` and `assembly`.** Pure moves; the independence is then enforced by an
   `independence` contract.
3. **Split further**, one package per assembly job. The four jobs share `assembly.py`, citations,
   progress and similarity, so this would create a fifth package for the shared part and four tiny
   ones.

## Decision

Option 2. Stage C moves the files; the contract then names `buddy`, `assembly`, `insights`,
`projects` and `skills` as independent features.

## Consequences

- A change to the Buddy cannot silently import assembly internals, and the reverse.
- Option 3 stays open; if one job grows its own models and prompts it can leave `assembly` later.
- Import paths change throughout `api` and the tests. The moves are mechanical.
