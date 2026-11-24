# AI service architecture

The AI service was built feature by feature, without an architecture worked out up front. This
folder reconstructs that architecture after the fact and sets a target for it, in four steps. Each
step builds on the one before it. It follows the same structure as the backend's
`docs/architecture`, so the two read side by side.

| Step | Document | Question it answers |
|---|---|---|
| 1 | [Drivers](01-drivers.md) | What does the architecture have to achieve, and within which limits? |
| 2 | [As-is architecture](02-as-is.md) | What did we actually build? Recovered from the code, not from memory. |
| 3 | [Target architecture](03-target.md) | Which structure serves the drivers, and which rules keep it in place? |
| 4 | [Migration](04-migration.md) | How do we get from step 2 to step 3 in small, verifiable steps? |

Decisions behind the target are recorded as ADRs in [`../adrs-proposed/`](../adrs-proposed/). The
accepted ADRs of the project (ADR-001 to ADR-020) live in the
[Wiki](https://github.com/SprintStartProject/Wiki/tree/main/docs/source/explanations/adrs). The
decision that splits responsibility between this service and the backend is the backend's
`adr-proposed-005-stateless-ai-backend-executes-tools.md`; this folder builds on it.

## How the numbers were produced

The as-is numbers come from the source tree at the commit named in each document:

- **Package dependencies:** an AST pass over `src/` that collects every `import` / `from … import`
  between our own packages, and `lint-imports` (Import Linter) with the contract in `.importlinter`.
- **Function length:** an AST pass; length is `end_lineno - lineno + 1` of every `def`.
- **Raw ids:** parameters named `*_id` and annotated `str` or `int`.
- **Duplication and coverage:** `scripts/quality/metrics.sh`.

The scripts are not part of the build. Once the target is in place, the
[enforcement rules](03-target.md#enforcement) keep these numbers current instead.
