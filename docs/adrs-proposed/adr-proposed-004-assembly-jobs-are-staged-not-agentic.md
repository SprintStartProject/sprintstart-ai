# ADR (proposed) 004: Assembly jobs are staged and deterministic, not agentic

**Status:** Proposed (2026-10-22), awaiting confirmation. Recorded after the fact: it describes a
decision the code made without an ADR.

## Context

The service generates four kinds of grounded content: phase content, task orientation, subject
diagrams and starter-work candidates. The Buddy shows that the model can be given tools and left to
loop. For these jobs that freedom was not wanted. The content ends up in front of a hire as the
project's own material, so every claim must trace to retrieved evidence, and a failed run must leave
no half-written result.

Drivers: Q2 groundedness, Q3 testability without a model, Q4 robustness.

## Options

1. **An agent per job:** give the model the retrieval tools and let it produce the result. Flexible,
   but groundedness would depend on the model's discipline and the run could not be replayed.
2. **A staged pipeline:** a corpus fingerprint gate, retrieval, one generation, one correction round,
   resolution against the evidence, and progress events. The code retrieves; the
   model writes only from what was retrieved.

## Decision

Option 2. The shared stages live in `assembly.py`; each job composes them with its own query,
prompt, parser and resolver. Anything the evidence does not support is dropped.

## Consequences

- The jobs are testable with a scripted model, and prompt changes show up as golden diffs.
- Results can be cached by the backend against the corpus fingerprint, which the gate computes.
- The cost is that a job cannot decide to search again. When a job needs that, it is no longer an
  assembly job and belongs with the Buddy.
- Each job is still one long `stream_*` function. Migration stage D reduces them to a declared
  sequence of the shared stages.
