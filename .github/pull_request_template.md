## What and why

<!-- One or two sentences. Link the issue. -->

## Architecture decision

<!-- Link the accepted ADR in the Wiki, or write "No trigger applies, because …". -->

## How it was tested

<!-- Tests added (they fail without this change). For a user-facing journey: the entry in
sprintstart-frontend/docs/testing/e2e-journeys.md and its spec, or the manual end-to-end
check that was actually performed. -->

## Checklist

- [ ] **AI-assisted:** an AI agent wrote part of this diff, and I have read every line of it
- [ ] No new function, file or class over a limit; baseline entries I touched are removed
- [ ] No suppression added, no check loosened, no test skipped or weakened
- [ ] New ids are value types; closed sets are enums/sealed types; no nullable-as-mode
- [ ] Each new function has one responsibility; entry points are thin
- [ ] I searched for existing code (where: …); nothing was copied and adapted
- [ ] No swallowed exception
- [ ] No comments narrating history or justifying a rule violation
- [ ] One concern, ≤ 400 changed lines excluding tests and generated code
- [ ] `bash scripts/quality/ratchet.sh` passes locally

<!-- A reviewer other than the author approves. If an AI agent wrote this code, the person who
ran the agent does not count as the reviewer. -->
