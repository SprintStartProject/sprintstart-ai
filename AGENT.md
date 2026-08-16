# AGENT.md

RAG + agent backend for SprintStart: a FastAPI service that ingests project
artifacts, indexes them, and answers questions / generates onboarding paths
over them.

> This doc describes the repo including `feature/98-...` (onboarding API +
> multi-backend LLM support), which adds `src/onboarding/` and isn't merged
> to `main` yet. If that branch changes significantly before merging, revisit
> this file.

## Setup

```bash
uv sync
cp .env.example .env   # fill in values — see .env.example for backend options
uv run python -m src.main   # serves on :8000, docs at /docs
```

Docker: `docker-compose up --build` (overrides `OLLAMA_BASE_URL` to
`http://host.docker.internal:11434` automatically).

## Commands

| Purpose | Command |
|---|---|
| Run tests | `uv run pytest` |
| Tests + coverage | `uv run pytest --cov=src --cov-report=term-missing` |
| Integration tests (needs Ollama running) | `uv run pytest -m integration` |
| Lint | `uv run ruff check .` |
| Type-check | `uv run pyright` |

Run lint + type-check + tests before considering a change done.

## Architecture

`src/` layout, by subpackage:

- **`api/`** — FastAPI app (`app.py`), DI (`dependencies.py`), `schemas.py`,
  and routes: `health`, `ingest`, `chat`, `title`, `vector_db`, plus
  `onboarding` and `blueprints` (from #98).
- **`agents/`** — `ChatAgent` (`chat_agent.py`) runs one flat tool loop over a
  single message list, capped by `_MAX_STEPS`: it searches while the model
  asks for searches, then streams the answer from that same conversation.
  Tool results carry the retrieved text, so the loop that searched is also the
  loop that answers — there is no separate gather and answer phase.
  `ChatOrchestrator` (`orchestrator.py`) maps the run's events into the SSE
  stream consumed by `/api/v1/chat`. Tools live in `agents/tools/`, registered
  via `ToolRegistry`.
- **`ingestion/`** — per-filetype parsers (`text_parser`, `pdf_parser`,
  `code_parser`, `image_parser`) behind `parser.py`, then `chunker.py` and
  `metadata_store.py`.
- **`rag/`** — `retriever.py`, `hybrid.py` (BM25 + vector hybrid retrieval),
  `citation.py`, `prompt.py`.
- **`llm/`** — `LLMClient` interface (`base.py`), implemented by
  `ollama_client.py`, `openai_client.py`, `anthropic_client.py`.
  `split_client.py` lets chat and embeddings use different backends
  (`LLM_BACKEND` for chat/generate/vision, `EMBED_BACKEND` for embeddings).
  Every backend takes a request timeout (`LLM_TIMEOUT_SECONDS`, default 600).
  The Anthropic client sets prompt-cache breakpoints on the system prompt and
  the last message — see `anthropic_client._user_message` before changing how
  messages are serialised, since caching matches on exact bytes.
- **`store/`** — `VectorStore` interface (`base.py`), `chroma_store.py`
  implementation.
- **`onboarding/`** (from #98) — deterministic staged pipeline for
  generating personalized onboarding paths, not a free agentic loop:
  `pipeline.py` runs select → filter → retrieve → synthesize → validate →
  emit, yielding `StageProgress` markers the API turns into SSE events.
  `blueprints.py` loads/selects blueprint content (from the top-level
  `blueprints/drafts/` and `blueprints/versions/` dirs); `synthesis.py` adds
  the LLM-personalized layer; `quality.py` gates the result (schema,
  coverage, grounding, human-owned invariants) and falls back to
  blueprint-only output if the LLM output fails a gate.

## Conventions

- `src` is on `pythonpath` for tests — import as `from agents.base import
  Agent`, not `from src.agents...`.
- Don't assume Ollama-only: check `llm/base.py`'s `LLMClient` interface when
  touching anything that calls an LLM, since backend is configurable per
  deployment (`LLM_BACKEND` / `EMBED_BACKEND`).
- Chat is one agent running one loop. Extend it with a new `Tool` in
  `agents/tools/` rather than a sub-agent to delegate to: every tier costs a
  serialized round-trip ahead of the user's first token, which is what the
  previous orchestrator/synthesis split cost. Tool results must carry their
  chunk text — a count-only summary forces a second pass to write the answer.
- **Tools must be read-only and thread-safe.** A turn's tool calls run
  concurrently (`ChatAgent._run_tools`), so a tool that mutates shared state
  would race. Anything cached across calls needs its own lock, the way
  `rag.hybrid.BM25IndexCache` guards its rebuild.
- `AGENT_DEBUG=1` logs each agent's reasoning (LLM text + tool calls) to
  stderr — useful when debugging agent behavior.
- The onboarding pipeline is intentionally deterministic/staged rather than
  agentic, specifically so a bad LLM output degrades to a blueprint-only path
  instead of breaking — keep that property when modifying it.

## Tests

`tests/` mirrors `src/` (`tests/onboarding/`, `tests/ingestion/`,
`tests/llm/`, etc). Reuse the fakes in `tests/stubs/llm.py` and
`tests/stubs/store.py` for `LLMClient`/`VectorStore` test doubles instead of
hand-rolling mocks. PDF/image fixtures for ingestion tests live in
`tests/ingestion/fixtures/`.
