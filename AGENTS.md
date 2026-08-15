# SprintStart AI Service

FastAPI service that ingests project artifacts, indexes them via RAG, and
answers questions / generates personalized onboarding paths.

## Setup

```bash
uv sync
cp .env.example .env   # fill in values
uv run python -m src.main   # dev server on :8000, docs at /docs
```

Docker: `docker-compose up --build` (overrides `OLLAMA_BASE_URL` to
`http://host.docker.internal:11434` automatically).

Production Docker CMD (Dockerfile): `uvicorn api.app:app --app-dir src`.

## Commands

| Purpose | Command |
|---|---|
| Run tests | `uv run pytest` |
| Tests + coverage | `uv run pytest --cov=src --cov-report=term-missing` |
| Integration tests (needs Ollama) | `uv run pytest -m integration` |
| Lint | `uv run ruff check .` |
| Format check | `uv run ruff format --check .` |
| Auto-format | `uv run ruff format .` |
| Type-check | `uv run pyright src/` |

Run **lint → format check → type-check → tests** before considering a change
done. CI enforces this order:
`gitleaks → quality (ruff lint + ruff format check + pyright) → pytest`.

## Branch protection

PRs to `main` must come from `dev` or `hotfix/*`. Push to `main`/`dev`
triggers Docker image publish to `ghcr.io/sprintstartproject/sprintstart-ai`.

## Architecture

`src/` layout:

- **`api/`** — FastAPI app (`app.py`), DI (`dependencies.py`), `schemas.py`,
  SSE helpers (`sse.py`), and route modules under `routes/`.
- **`agents/`** — `ChatAgent` (`chat_agent.py`) runs one flat tool loop over a
  single message list: it searches while the model asks, then streams the
  answer from that same conversation. Tool results carry the retrieved text,
  so the loop that searched is the loop that answers. Tools live in
  `agents/tools/`, registered via `ToolRegistry`. `ChatOrchestrator`
  (`orchestrator.py`) maps the run's events to SSE for `/api/v1/chat`.
- **`ingestion/`** — Per-filetype parsers (`text_parser`, `pdf_parser`,
  `code_parser`, `image_parser`) behind `parser.py`, then `chunker.py` and
  `metadata_store.py`.
- **`rag/`** — `retriever.py`, `hybrid.py` (BM25 + vector hybrid retrieval
  with RRF fusion), `citation.py`, `prompt.py`.
- **`llm/`** — `LLMClient` protocol (`base.py`) with implementations:
  `ollama_client.py`, `openai_client.py`, `anthropic_client.py`.
  `SplitLLMClient` lets chat and embeddings use different backends
  (`LLM_BACKEND` vs `EMBED_BACKEND`). All three take a request timeout
  (`LLM_TIMEOUT_SECONDS`, default 600). The Anthropic client sets
  prompt-cache breakpoints; caching matches on exact bytes, so read
  `anthropic_client._user_message` before changing message serialisation.
- **`store/`** — `VectorStore` protocol (`base.py`), `chroma_store.py`.
- **`onboarding/`** — Deterministic staged pipeline (not agentic):
  `select → filter → retrieve → synthesize → validate → emit`, yielding
  `StageProgress` markers. Blueprints are owned by the backend and passed in
  on each request — the service is stateless. Scopes are project-qualified
  (`project:<id>|global`, `project:<id>|area:<name>`), parsed by
  `onboarding/scope.py`.

## Project separation

Every retrieval-backed request is scoped to exactly one project, and the
scoping is **fail-closed** — a chunk with no project association is invisible
to all projects rather than visible to all of them. When touching retrieval,
ingest, or an endpoint, keep that property:

- `project_id` is required on `ChatRequest`, `OnboardingPathRequest`,
  `GenerateBlueprintsRequest`, `KnowledgeGapsRequest` and `FaqGroupRequest`
  (all inherit `ProjectScopedRequest` in `api/schemas.py`).
- `RetrievalFilters.project_id` is enforced in **both** halves of hybrid
  retrieval — `where_filter_for_chroma()` for the vector side and
  `matches_retrieval_filters()` for the BM25/in-memory side (`rag/filters.py`).
  Anything that scans `all_chunks_without_embeddings()` directly (the `grep`
  and `fetch_file` tools) must apply the filter itself.
- Chroma metadata cannot hold lists, so membership is stored twice: a
  delimited `project_ids` string (read back into `Chunk.project_ids`) and one
  `project:<id>` boolean marker key per project for the `where` clause. Chroma
  *merges* metadata on upsert, so `ChromaVectorStore.add` deletes the ids first
  — otherwise a stale marker would keep a moved artifact visible to its old
  project.
- Chunks ingested before the backend sent `projectIds` are unreachable; a full
  `POST /api/v1/ingest/sync` re-sync backfills them.

## Conventions

- `src` is on `pythonpath` for tests — import as `from agents.base import
  Agent`, **not** `from src.agents...`.
- Don't assume Ollama-only: check `llm/base.py`'s `LLMClient` protocol when
  touching LLM calls. Backend is configurable per deployment via
  `LLM_BACKEND` / `EMBED_BACKEND`.
- Chat is one agent, one loop. Give it a new `Tool` in `agents/tools/` rather
  than a sub-agent to delegate to: every tier costs a serialized round-trip
  ahead of the user's first token. Tool results must carry their chunk text —
  a count-only summary forces a second pass to write the answer.
- Tools must be read-only and thread-safe: a turn's tool calls run
  concurrently (`ChatAgent._run_tools`). Cache across calls only behind a lock,
  as `rag.hybrid.BM25IndexCache` does.
- `AGENT_DEBUG=1` logs each agent's reasoning (LLM text + tool calls) to
  stderr — useful when debugging agent behavior.
- The onboarding pipeline is intentionally deterministic/staged rather than
  agentic: a bad LLM output degrades to a blueprint-only path instead of
  breaking. Keep that property when modifying it.
- `data/` is gitignored — don't commit local ChromaDB state or test fixtures
  there.

## Tests

`tests/` mirrors `src/`. Key utilities in `conftest.py`:

- `llm_required` / `vision_required` markers skip tests when Ollama is
  unreachable or vision model unconfigured.
- `parse_sse_events()` parses SSE streams into `dict` lists.
- `clear_dependency_caches` (autouse fixture) resets `lru_cache` on
  `get_llm`, `get_store`, `get_ingestion_metadata_store` before each test.

Reuse the fakes in `tests/stubs/llm.py` (`StubLLMClient`, `ScriptedLLMClient`)
and `tests/stubs/store.py` (`StubVectorStore`) instead of hand-rolling mocks.