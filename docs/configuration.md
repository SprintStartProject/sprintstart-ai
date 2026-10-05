# Configuration

## Providers

Set `LLM_BACKEND` explicitly: if it is empty the service falls back to Ollama. Chat and embeddings can use different providers via `EMBED_BACKEND`.

### OpenAI and compatible endpoints (default)

```env
LLM_BACKEND=openai
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBED_MODEL=text-embedding-3-small   # no default, must be set
CHROMA_PATH=./data/chroma                   # unset means in-memory, data is lost on restart
```

`OPENAI_BASE_URL` defaults to `https://api.openai.com/v1`. Point it at Azure, LiteLLM, OpenRouter or vLLM to use those instead. Set `OPENAI_VISION_MODEL` to enable image captioning.

### Anthropic

Anthropic has no embeddings API, so pair it with an OpenAI-compatible embedding backend:

```env
LLM_BACKEND=anthropic
ANTHROPIC_API_KEY=...
EMBED_BACKEND=openai
OPENAI_API_KEY=...
OPENAI_EMBED_MODEL=...
```

`OPENAI_EMBED_BASE_URL` and `OPENAI_EMBED_API_KEY` override the endpoint and key for embeddings only.

### Ollama

Fine for small local experiments, but running it on consumer hardware does not hold up against a full project corpus.

```bash
ollama pull llama3.2 && ollama pull nomic-embed-text
```

```env
LLM_BACKEND=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
OLLAMA_NUM_CTX=32768          # Ollama's default silently truncates large prompts
```

In Docker, use `OLLAMA_BASE_URL=http://host.docker.internal:11434` to reach Ollama on the host.

## Variables

`.env.example` is the reference, with every variable commented. The ones you will touch most:

| Variable | Purpose |
|---|---|
| `LLM_BACKEND` / `EMBED_BACKEND` | `openai` (also LiteLLM, OpenRouter, any compatible endpoint), `anthropic` or `ollama`. Set `LLM_BACKEND` explicitly. `EMBED_BACKEND` is optional and splits embeddings off. |
| `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_CHAT_MODEL`, `OPENAI_EMBED_MODEL`, `OPENAI_VISION_MODEL` | OpenAI-compatible backend. Chat model defaults to `gpt-4o-mini`. The embedding model has no default. |
| `OPENAI_MAX_TOKENS`, `OPENAI_REASONING_MAX_TOKENS` | Optional output limit and opt-in reasoning budget for streamed answers. The limit must exceed the reasoning budget. |
| `OPENAI_EMBED_DIMENSIONS` | Fixed vector size for models that support it. Must match the existing Chroma collection, so changing it means re-creating the collection. |
| `ANTHROPIC_API_KEY`, `ANTHROPIC_CHAT_MODEL`, `ANTHROPIC_THINKING_BUDGET_TOKENS` | Anthropic backend and its opt-in reasoning budget. |
| `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, `OLLAMA_EMBED_MODEL`, `OLLAMA_VISION_MODEL`, `OLLAMA_NUM_CTX` | Ollama endpoint, models and context window. |
| `CHROMA_PATH` | Persistent ChromaDB directory. Unset means in-memory, and data is lost on restart. |
| `CHUNK_SIZE`, `CHUNK_OVERLAP` | Chunking. |
| `INGEST_CONCURRENCY`, `INGEST_MAX_CONTENT_LENGTH`, `INGEST_MAX_BINARY_BYTES` | Ingest limits. Oversized payloads are skipped (`chunk_count=0`). |
| `LLM_TIMEOUT_SECONDS` | Request timeout for every backend (default 600, `0` disables). |
| `AGENT_DEBUG` | `1` logs each agent's reasoning and tool calls to stderr. |
