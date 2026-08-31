# sprintstart-ai

The AI and RAG pipeline service for [SprintStart](https://sprintstart.readthedocs.io/en/latest/), an AI-assisted onboarding and knowledge-retrieval platform for software development teams.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- [Ollama](https://ollama.com/) running locally with the required models pulled:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text

# Optional — required only for image captioning
ollama pull llava:7b        # lightweight, good for development
# ollama pull qwen2-vl:7b  # recommended for production (better on diagrams/text-heavy images)
```

## Getting Started

### Local

```bash
# 1. Install dependencies
uv sync

# 2. Configure environment
cp .env.example .env
# Edit .env and fill in the values

# 3. Run the service
uv run python -m src.main
```

The service runs on port `8000`. Interactive docs are available at `/docs`.

### Docker

```bash
# 1. Configure environment
cp .env.example .env
# Edit .env and fill in the values

# 2. Start the service
docker-compose up --build
```

The service runs on port `8000`.

> `OLLAMA_BASE_URL` is automatically overridden to `http://host.docker.internal:11434` inside the container, so no manual change is needed.

## Environment Variables

| Variable | Example value | Description |
|---|---|---|
| `LLM_BACKEND` | `ollama` | LLM backend used for chat/generation: `ollama`, `openai`/`litellm`, or `anthropic`. |
| `ANTHROPIC_THINKING_BUDGET_TOKENS` | `1024` | Reasoning-token budget for streamed Anthropic chat answers. Must be at least 1024 and lower than `ANTHROPIC_MAX_TOKENS`; set `0` to disable. |
| `OPENAI_MAX_TOKENS` | `4096` | Optional output-token limit for streamed OpenAI-compatible chat answers. Must be greater than the reasoning budget when both are configured. |
| `OPENAI_REASONING_MAX_TOKENS` | `1024` | Opt-in reasoning-token budget for streamed OpenAI-compatible chat answers (including OpenRouter). Set `0` or leave empty to disable. |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Base URL of the Ollama instance. Use `http://host.docker.internal:11434` when running via Docker with Ollama on the host. |
| `OLLAMA_MODEL` | `gemma4:e4b` | Chat model to use for generation. |
| `OLLAMA_EMBED_MODEL` | `nomic-embed-text:latest` | Embedding model to use for ingestion and retrieval. |
| `OLLAMA_VISION_MODEL` | `llava:7b` (dev) / `qwen2-vl:7b` (prod) | Vision model for image captioning. Optional — if unset, image files are accepted but produce no chunks (`chunk_count=0`). |
| `CHROMA_PATH` | `./data/chroma` | Path for ChromaDB persistent storage. If unset, an in-memory store is used and data will not persist. |
| `AGENT_DEBUG` | `0` | When set to a truthy value, logs each agent's reasoning step (LLM text and tool calls) to stderr. Disabled by default and for `0`/`false`/`no`/`off`/empty. |
| `CHUNK_SIZE` | `512` | Maximum number of characters per chunk |
| `CHUNK_OVERLAP` | `64` | Number of characters reused between consecutive chunks to preserve context when splitting large chunks |
| `CONTEXT_AWARE_CHUNKING_MAX_CHARS` | `24000` | Character-count ceiling (a proxy for a token limit — there is no tokenizer in this project) for text/PDF content sent to the LLM-based context-aware chunker. Above this, ingestion falls back to the plain `chunk_text` strategy. See [Context-aware chunking](#context-aware-chunking). |
| `INGEST_CONCURRENCY` | `8` | Maximum number of artifacts embedded concurrently across a batch in `POST /api/v1/ingest/sync`. |
| `INGEST_MAX_CONTENT_LENGTH` | `500000` | Maximum character length for text/code artifacts. Payloads exceeding this are skipped (`chunk_count=0`). |
| `INGEST_MAX_BINARY_BYTES` | `10485760` | Maximum decoded byte size for binary artifacts (images and PDFs, 10 MB default). Payloads exceeding this are skipped (`chunk_count=0`). |

## Project separation

Artifacts belong to one or more **projects** (the backend's `artifact_projects` mapping), and every retrieval-backed feature is scoped to exactly one project. Concretely:

- **Ingest** carries the membership: `projectIds` on `POST /api/v1/ingest` and on each artifact of `POST /api/v1/ingest/sync`. It is written into the chunk metadata (as a delimited `project_ids` string plus one `project:<id>` marker key that Chroma can filter on server-side) and into the ingestion index.
- **Retrieval** takes a `project_id` filter that is enforced on both halves of hybrid retrieval — the Chroma `where` clause and the in-memory/BM25 path (`rag/filters.py`) — as well as in the corpus-scanning agent tools (`grep`, `fetch_file`).
- **Requests** must name the project: `projectId` is required on `/chat`, `/onboarding/path`, `/onboarding/path/yaml`, `/onboarding/blueprints/generate`, `/insights/knowledge-gaps/detect`, and `/insights/faq/group`. The backend authorizes the caller for that project (`ProjectAuthorization`) and passes it through.
- **Blueprints** are project-qualified: generated scopes are `project:<projectId>|global` and `project:<projectId>|area:<name>`, and the corpus fingerprint that makes generation idempotent covers only that project's chunks. Blueprints scoped to another project — or unqualified, i.e. generated before project separation — are ignored when a path is assembled.

Filtering is **fail-closed**: a chunk with no project association is invisible to every project rather than visible to all of them. Chunks ingested before `projectIds` was sent therefore stop being retrievable — re-run a full `POST /api/v1/ingest/sync` with `projectIds` populated to backfill them.

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Reports service health including LLM backend status. Returns `503` if Ollama is unreachable. |
| `POST` | `/api/v1/ingest` | Parses, chunks, and embeds a document and stores it in the vector store. Re-ingesting the same `artifact_id` replaces existing chunks. `projectIds` records which projects may retrieve it (see [Project separation](#project-separation)). Supports text files and images (send image content as base64; requires `OLLAMA_VISION_MODEL`). Text/PDF content can optionally be chunked using the LLM-based [context-aware chunker](#context-aware-chunking) (opt-in, off by default). |
| `POST` | `/api/v1/chat` | Retrieves relevant chunks from the given `projectId` and streams a generated answer as Server-Sent Events (SSE). |
| `POST` | `/api/v1/title` | Generates a short descriptive title from a user prompt. |
| `POST` | `/api/v1/onboarding/path` | Generates a personalized onboarding path for one `projectId` (SSE). Blueprints are passed in by the backend on each request — the service is stateless. |
| `POST` | `/api/v1/onboarding/blueprints/generate` | Batch job: generates `source: generated` blueprints from one project's corpus and returns them. The backend owns persistence — it passes its active blueprints in and stores the results. |

### Chat SSE stream

The `/api/v1/chat` endpoint streams newline-delimited JSON events:

| Event type | Description |
|---|---|
| `tool_use` | A capability the agent invoked, in order. Has `name` and `kind` (currently always `tool`; `agent` is reserved for a delegating capability) |
| `reasoning` | A live reasoning fragment in the `reasoning` field. It is never part of the final answer or persisted message content. Models without a reasoning channel simply omit these events. |
| `token` | A single token fragment of the answer |
| `citation` | A source chunk used to generate the answer |
| `done` | Signals the end of the stream |
| `error` | Emitted on failure instead of the above |

## Context-aware chunking

`POST /api/v1/ingest` can optionally chunk text and PDF content using an LLM instead of the plain character-length `chunk_text` strategy (`ingestion/chunker.py`). This is opt-in per request via two independently toggleable flags on `IngestRequest`, both `false` by default:

| Field | Default | Description |
|---|---|---|
| `semantic_boundaries` | `false` | Let the LLM choose chunk boundaries based on topic shifts / section boundaries instead of accumulating paragraphs up to `CHUNK_SIZE`. |
| `contextualize` | `false` | Let the LLM selectively prepend a short situating "context block" (Anthropic-style *Contextual Retrieval*) to chunks that would otherwise lack context; self-contained chunks are left untouched. |

Both flags share a single LLM call (see `ingestion/context_aware_chunker.py`). Leaving both at `false` (the default) skips the LLM call entirely and behaves exactly like the legacy chunker — existing callers are unaffected unless they explicitly opt in. When either is enabled, the strategy still always falls back to `chunk_text` — no request ever fails because of it — when:

- the content exceeds `CONTEXT_AWARE_CHUNKING_MAX_CHARS`,
- the LLM backend is unreachable, or
- the LLM response is missing, malformed, or fails validation (e.g. out-of-range boundaries).

Resulting chunks carry extra metadata so context/overlap portions of a chunk's content are identifiable: `has_context_block`, `context_block_range`, `has_overlap`, `overlap_range` (character ranges as `"start:end"` strings, empty when not applicable).

**Scope:** only text and PDF content are affected; code and image parsing are unchanged. For PDFs, the LLM is invoked **per page** (pages are already parsed independently), so semantic boundaries and context blocks are page-scoped, not whole-document-scoped. The batch endpoint `POST /api/v1/ingest/sync` (GitHub run sync) does not expose these flags and always uses the plain chunker — changing that would mean changing the batch wire contract with the backend, which is a separate decision.


## AI-proposed onboarding blueprints

Onboarding paths are assembled from versioned **blueprints** scoped by project and audience: `project:<projectId>|global` or `project:<projectId>|area:<name>`. Blueprints carry a `source`: `authored` (human-written) or `generated` (drafted by the AI from the ingested corpus). The backend owns blueprint persistence, versioning, and rollback — the AI service is stateless and only returns generated data.

The generation job analyzes the requested project's corpus and returns proposed blueprints for the backend to persist. Every generated step is **grounded** (cites an ingested chunk), the job is **idempotent** (a blueprint records the `corpus_fingerprint` it was drafted from, so an unchanged corpus is a no-op), and **human-owned invariants are protected** — it may not remove or downgrade a `required` or `invariant: true` step; such changes are re-injected and the outcome is escalated, never silently applied.

Blueprint management (trigger generation, list versions, rollback) is exposed through the backend API at `POST /api/v1/onboarding/blueprints/generate`, `GET /api/v1/onboarding/blueprints/{scope}/versions`, and `POST /api/v1/onboarding/blueprints/{scope}/rollback`.

### Cost / performance

A full refresh is **O(number of scopes)**, not O(corpus size) or per onboarding request. Per scope it performs:

- **1 hybrid retrieval** (one embedding call + a cached in-memory BM25 pass over the corpus), and
- **1 LLM `generate` call** drafting the blueprint from the top ~12 retrieved chunks.

So a refresh of *global + N areas* is `N+1` retrievals and `N+1` generate calls — typically single-digit LLM calls for a whole organization. Scopes whose corpus fingerprint is unchanged are skipped entirely (no LLM call). The job is batch and schedulable; the latency-sensitive `/onboarding/path` request path never invokes generation.

## Running Tests

```bash
uv run pytest
```

With coverage:

```bash
uv run pytest --cov=src --cov-report=term-missing
```
