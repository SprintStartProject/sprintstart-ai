# 2 As-is architecture

> **Status: draft (2026-10-22).** Recovered from `main` at `c188b82` (103 Python files, 18.7k lines in
> `src/`, ~1,000 test functions). Numbers come from the passes described in the
> [README](README.md#how-the-numbers-were-produced).

## 2.1 Shape

One FastAPI application, one deployable. `src/` is ten top-level packages. Their size says where
the code is:

| Package | Lines | What it holds |
|---|---:|---|
| `onboarding` | 4,829 | The Buddy turn (`buddy_agent`, `buddy_open`, `buddy_compact`, `buddy_persona`) and the four assembly jobs (`phase`, `orientation`, `diagram`, `starter_work`) with their models |
| `api` | 4,803 | Routes, 1,563 lines of request/response models (`schemas.py`), SSE framing, stream contract, dependency wiring |
| `ingestion` | 2,708 | Parsers (text, PDF, code, image), two chunkers, ingestion metadata (SQLite), source state |
| `llm` | 2,031 | `LLMClient` protocol, Ollama / OpenAI / Anthropic clients, split client, tool-call recovery |
| `insights` | 1,138 | FAQ grouping and classification, knowledge gaps, PII redaction |
| `store` | 1,073 | `VectorStore` protocol and the Chroma implementation |
| `rag` | 1,054 | Hybrid retrieval (vector + BM25, RRF), filters, context window, source classification |
| `agents` | 649 | Tool registry and the retrieval / grep tools, artifact summaries |
| `skills` | 276 | Skill suggestion |
| `projects` | 128 | Project industry evaluation |

The service is stateless toward the business: it keeps the corpus (Chroma) and ingestion metadata
(SQLite) and nothing about users, paths or conversations. The backend sends what a turn needs and
runs the Buddy's backend tools itself (backend ADR-005). The wire contracts are `api/schemas.py`
(pinned in `contracts/`) and the Buddy stream schema.

## 2.2 What exists to hold the structure

- **`.importlinter`:** one layers contract, `api > features > agents > rag > ingestion > store > llm`,
  where the features (`onboarding | insights | projects | skills`) are independent of each other.
  It passes (101 files, 376 dependencies), but only because it **ignores 14 imports**.
- **Ruff limits ratchet:** complexity, branches, arguments, statements and blind `except` are
  checked against a baseline of 56 entries that may only shrink.
- **Gates:** coverage 92.6 %, mutation testing on added code, duplication (119 lines), dead code
  (17 findings), pyright on `src/`.
- **No written architecture.** No drivers, no ADR, no description of the layers beyond the
  `.importlinter` comment and the package list in `AGENTS.md`.

## 2.3 The layering is inverted at the bottom

The contract says `rag` sits above `ingestion`, which sits above `store`. The imports disagree.
Three pairs of packages import each other, and two features reach back into `api`:

| Direction | Ignored imports | Imported through |
|---|---:|---|
| `ingestion` → `rag` | 4 | `ingestion.mapper` and `metadata_store` import `rag.filters` / `rag.types`; `source_state_store` imports `rag.source_filter` |
| `store` → `rag` | 5 | `store.base` and `chroma_store` import `rag.types`, `rag.source_filter`; `chroma_store` imports `rag.filters` |
| `store` → `ingestion` | 3 | `store.base` and `chroma_store` import `ingestion.source_role`; `chroma_store` imports `ingestion.metadata_store` |
| `projects`, `skills` → `api` | 2 | `projects.evaluation` and `skills.suggestion` import their response models from `api.schemas` |

The opposite directions are allowed by the contract, which is what closes the cycles: `rag` imports
`ingestion.source_role` and `store.base`, and `ingestion` imports `store`. So `ingestion` ↔ `rag`,
`store` ↔ `rag` and `store` ↔ `ingestion` are all mutual pairs. The 14 ignored imports are the
upward half of those pairs plus the two `api` imports.

The cause is one thing: **the shared vocabulary of the corpus lives in the wrong packages.**
`Chunk`, `ScoredChunk`, `RetrievalFilters` and `SourceSystem` are in `rag.types`. `SourceRole` is in
`ingestion.source_role`. `SourceExclusions` is in `rag.source_filter`. The id and filter encoding
helpers are in `rag.filters`, mixed with `where_filter_for_chroma`, which is Chroma-specific and
belongs next to the store. `store`, the lowest data layer, needs all of them, so it reaches up.
`IngestionMetadataStore` is a SQLite store that `ChromaVectorStore` imports (for `RevocationReason`);
it is persistence but lives in `ingestion`.

The second cause is smaller: the two response models the features borrow from `api.schemas` are
feature models by nature.

## 2.4 Orchestration sits in the API layer

`api/routes` is 4,000 lines and several route modules hold the business flow, not just the HTTP
adaptation:

| Route module | Lines | What it holds |
|---|---:|---|
| `ingest_run.py` | 433 | The batch ingest pipeline. `_ingest_one` is 239 lines, the longest function in the service. |
| `ingest.py` | 273 | The single-artifact ingest pipeline. `ingest` is 156 lines. |
| `buddy.py` | 347 | Request mapping and SSE framing for the Buddy endpoints; `buddy_agent_stream` is 70 lines. |
| `grading.py` | 200 | Answer grading (`_grade`, 54 lines) inside the route module. |
| `artifact_projects.py` | 148 | Membership rewriting (`_rewrite_one`, 36 lines) inside the route module. |

Seven route handlers are longer than 45 lines: `_ingest_one` (239), `ingest` (156),
`summarize_artifact` in `summaries.py` (76), `buddy_agent_stream` (70), `ingest_run` (57), `_grade`
(54) and `generate_title` in `title.py` (48). All other handlers are shorter, and most route modules
are thin adapters. The problem is concentrated in the two ingest modules.

`ingest.py` and `ingest_run.py` import almost the same set of `ingestion`, `llm` and `store`
modules and each implements parse → classify → chunk → embed → store → revoke memberships. There
are two ingest pipelines, in the API layer, with no shared pipeline object.

## 2.5 Inside `onboarding`

`onboarding` is already two clusters that do not import each other. They share nothing except two
small helpers:

| Cluster | Files | Lines |
|---|---|---:|
| **Buddy** | `buddy_agent`, `buddy_open`, `buddy_compact`, `buddy_persona`, `query_fence`, `vocabulary` | 1,800 |
| **Assembly** | `assembly` (shared stages), `phase`, `orientation`, `diagram`, `starter_work`, their `*_models`, `corpus` (fingerprint gate), `citations`, `progress`, `similarity`, `models` | 3,029 |

The four assembly jobs follow one shape (query → retrieve → generate → one correction round →
resolve against evidence → emit progress), and `assembly.py` already holds the shared stages. Each
job is still one long `stream_*` function around those stages.

## 2.6 Long functions and raw values

| Measure | Value |
|---|---:|
| Functions | 499 |
| Longer than 30 lines | 111 |
| Longer than 60 lines | 40 |
| Longer than 100 lines | 14 |
| `*_id` parameters typed `str` / `int` | 50 (`artifact_id` 29, `project_id` 15, `connector_id` 4) |
| `status` / `kind` / `role` / `type` parameters typed `str` | 1 |
| `dict[str, Any]` annotations / uses of `Any` | 8 / 29 |

The 14 functions over 100 lines:

| Lines | Function |
|---:|---|
| 239 | `api.routes.ingest_run._ingest_one` |
| 219 | `onboarding.buddy_agent.stream_agent_turn` |
| 173 | `onboarding.phase.stream_phase` |
| 161 | `onboarding.orientation.stream_orientation` |
| 161 | `onboarding.diagram.stream_diagram` |
| 156 | `api.routes.ingest.ingest` |
| 148 | `onboarding.starter_work.stream_starter_work_pool` |
| 143 | `ingestion.context_aware_chunker.chunk_text_context_aware` |
| 140 | `skills.suggestion.suggest_skills` |
| 121 | `onboarding.buddy_open.stream_session` |
| 120 | `onboarding.buddy_persona.build_persona` |
| 120 | `llm.openai_client.chat_stream` |
| 116 | `store.chroma_store.query` |
| 111 | `ingestion.pdf_parser.parse_pdf` |

The same four ideas recur in the long ones: the ingest flow, the `stream_*` assembly jobs, the
Buddy turn and provider stream handling.

Types are otherwise in good shape: all but one function has a return annotation, and there are no
stringly-typed statuses. The type debt is the **identifiers**: a project id, an artifact id and a
connector id are all `str` and can be swapped without the type checker noticing.

## 2.7 Process-wide state

| State | Where | Note |
|---|---|---|
| BM25 index | `rag.retriever._BM25_CACHE` | One index over the **whole** corpus, shared by all callers and rebuilt when the shared corpus revision changes. Project scoping is applied after the fact. Memory grows with the corpus. |
| Chroma clients | `store.chroma_store._CLIENT_CACHE` | One per path |
| LLM, store and metadata clients | `api.dependencies` (`lru_cache`) | The composition root |

This is acceptable for one instance (see [drivers §1.6](01-drivers.md#16-what-is-deliberately-not-a-driver)),
but the BM25 index is the one place where the memory goal Q7 is exposed.

## 2.8 Tests and contracts

- ~1,000 test functions in 98 files under `tests/`, mirroring `src/`. The Buddy and the assembly
  jobs have contract suites with prompt goldens and a scripted LLM stub
  (`tests/onboarding/pipeline_fixtures.py`).
- `contracts/agent-stream-events.schema.json` pins the Buddy stream; the backend checks the same file.
- The user journeys run end to end in the frontend repository against an **AI stub**
  (`e2e/ai-stub`), so no journey exercises this service. Its own behaviour is covered by the suites
  above and by the mutation gate, not by a journey.

## 2.9 Findings

| # | Finding | Evidence |
|---|---|---|
| F1 | The corpus vocabulary sits in `rag` and `ingestion`, so the lowest layer imports the upper ones. Three package pairs are cyclic, hidden by 14 ignored imports. | §2.3 |
| F2 | Two ingest pipelines live in route modules and duplicate each other's flow. | §2.4 |
| F3 | `api.schemas` is imported by two features that should be independent of `api`. | §2.3 |
| F4 | `onboarding` holds two independent clusters under one package name. | §2.5 |
| F5 | The long functions are four recurring shapes, not 14 unrelated cases. | §2.6 |
| F6 | Ids are untyped strings; the type debt is concentrated in three identifiers. | §2.6 |
| F7 | No document says why the layers are what they are. | §2.2 |
