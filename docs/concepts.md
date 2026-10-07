# Concepts

How the service keeps its answers grounded, scoped and cheap.

<details open>
<summary><b>Project separation</b></summary>

Artifacts belong to one or more projects, and every retrieval-backed feature is scoped to them.

- **Ingest** records membership via `projectIds`, written into chunk metadata as a delimited `project_ids` string plus one `project:<id>` marker key that Chroma can filter on server-side.
- **Retrieval** applies the project filter to both halves of hybrid search (Chroma `where` clause and the BM25/in-memory path in `rag/filters.py`), and to the corpus-scanning `grep` and `fetch_file` tools.
- **Requests** must name the project. `projectId` is required on chat, onboarding, blueprint generation and the insights endpoints. The backend authorizes the caller and passes it through.
- **Blueprints** are project-qualified: `project:<projectId>|global` and `project:<projectId>|area:<name>`. Anything scoped to another project, or unqualified, is ignored.

Chunks ingested before `projectIds` was sent are unreachable. Re-run a full `POST /ingest/sync` to backfill them.

</details>

<details open>
<summary><b>Hybrid retrieval</b></summary>

Retrieval fuses BM25 and vector search with reciprocal rank fusion (`rag/hybrid.py`). The BM25 index is cached in memory behind a lock, since a turn's tool calls run concurrently.

</details>

<details open>
<summary><b>Context-aware chunking</b></summary>

`POST /ingest` can optionally let an LLM choose chunk boundaries (`semantic_boundaries`) and prepend a short situating context block to chunks that need one (`contextualize`, in the spirit of Anthropic's Contextual Retrieval). Both flags default to `false`, which skips the LLM entirely and uses the plain character chunker.

When enabled, it still falls back to the plain chunker if the content exceeds `CONTEXT_AWARE_CHUNKING_MAX_CHARS`, the LLM is unreachable, or its response fails validation. No ingest request fails because of it. Only text and PDF are affected (PDFs are processed per page), and `/ingest/sync` always uses the plain chunker.

Resulting chunks carry `has_context_block`, `context_block_range`, `has_overlap` and `overlap_range` metadata.

</details>

<details open>
<summary><b>AI-proposed blueprints</b></summary>

Paths are assembled from versioned blueprints with a `source` of `authored` (human) or `generated` (drafted from the corpus). The backend owns persistence, versioning and rollback. This service only returns generated data.

Generation is **grounded** (every step cites a chunk), **idempotent** (an unchanged corpus fingerprint is a no-op) and protects human-owned invariants (it can never remove or downgrade a `required` or `invariant: true` step).

A refresh is O(number of scopes): one hybrid retrieval and one LLM call per scope, so *global + N areas* costs N+1 of each. It is a schedulable batch job and never runs on the latency-sensitive `/onboarding/path` request path.

</details>
