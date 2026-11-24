# ADR (proposed) 002: Routes adapt, services orchestrate

**Status:** Proposed (2026-10-22), awaiting confirmation.

## Context

Seven route handlers are longer than 45 lines, and the two largest (`_ingest_one`, 239 lines, and
`ingest`, 156 lines) are the service's two ingest pipelines. Each implements parse → classify →
chunk → embed → store → revoke memberships inside a route module, with almost the same imports and
no shared code (as-is §2.4). The pipelines cannot be tested without building a request, a change
has to be made twice, and the ingest flow is invisible to anyone looking at `ingestion`.

Drivers: Q6 understandability, Q3 testability, Q5 changeability.

## Options

1. **Leave the logic in the routes** and extract helpers inside each route module. Removes the long
   functions, but the two pipelines stay two.
2. **A shared `ingestion.pipeline`** that both endpoints call. The routes validate, call it and map
   the result.
3. **A job queue** for ingestion. Solves a different problem (long-running work) and adds
   infrastructure for one instance.

## Decision

Option 2. `api.routes` may not import parsers, chunkers or the Chroma store (a `forbidden` import
contract), and only `api` may import `fastapi`. The same rule moves grading, the membership rewrite,
summary generation and title generation into the package that owns them.

## Consequences

- One ingest pipeline, testable without HTTP, with small steps that can be read one at a time.
- The route modules become uniform: validate, call, frame. The Buddy stream framing stays in `api`
  because it is HTTP/SSE, but the event mapping moves to the Buddy.
- The two ingest endpoints keep their URLs and wire models (migration stage B).
