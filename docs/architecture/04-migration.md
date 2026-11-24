# 4 Migration

> **Status: draft (2026-10-22), proposed.** From the [as-is](02-as-is.md) to the
> [target](03-target.md) in small PRs. Every PR passes the full gate set (`ruff`, limits, `pyright`,
> `pytest`, `lint-imports`, the ratchet) and none changes behaviour or an HTTP contract. Each stage
> has an exit measure, so progress shows up as numbers.

## 4.1 Tracked numbers

| Number | Today | Target | Source |
|---|---:|---:|---|
| Ignored imports in `.importlinter` | 14 | 0 | `.importlinter` |
| Mutual package pairs | 3 | 0 | `lint-imports` |
| Features importing `api` | 2 | 0 | `lint-imports` |
| Route modules that import parsers, chunkers or the Chroma store | 2 (`ingest`, `ingest_run`) | 0 | forbidden contract |
| Route handlers over 45 lines | 7 (see [§2.4](02-as-is.md#24-orchestration-sits-in-the-api-layer)) | 0 | AST |
| Ingest pipelines | 2 | 1 | `ingestion.pipeline` |
| Functions over 100 lines | 14 | 0 | `ruff-limits` / AST |
| Functions over 60 lines | 40 | ≤ 15 | AST |
| `*_id` parameters typed `str` | 50 | 0 | AST |
| Packages named in the contract | 10 | 12 (`corpus` added, `onboarding` split into `buddy` and `assembly`) | `.importlinter` |

## 4.2 Stages

### Stage A: make the layering true (cheap, high yield)

Moves only. No logic changes, and every PR leaves the suite unchanged.

| PR | Change | Exit measure |
|---|---|---|
| A1 | Create `corpus`. Move `rag.types`, `ingestion.source_role`, `rag.source_filter` and the pure helpers of `rag.filters` (`normalize_source_system`, the project-id encoding and metadata keys, `matches_retrieval_filters`, `has_narrowing_filters`, timestamp helpers) into it. | 14 → 4 |
| A2 | Move `where_filter_for_chroma` from `rag.filters` to `store`. | 4 → 3 |
| A3 | Move `IngestionMetadataStore` and `SourceStateStore` into `store`. | 3 → 2 |
| A4 | `projects.evaluation` and `skills.suggestion` return feature result types; `api` maps them to `schemas`. | 2 → 0 |
| A5 | Rewrite `.importlinter` to the seven layers of [§3.1](03-target.md#31-layers) with an empty `ignore_imports`, plus the `corpus` contract. | mutual pairs 3 → 0 |

### Stage B: routes adapt, services orchestrate

| PR | Change | Exit measure |
|---|---|---|
| B1 | Extract `ingestion/pipeline.py` from `ingest_run.py` and `ingest.py`, with the shared steps as separate small functions. Both routes call it. | ingest pipelines 2 → 1; `_ingest_one` 239 → ≤ 60 |
| B2 | Move answer grading (`grading.py`), the membership rewrite (`artifact_projects.py`), the summary generation in `summaries.py` and the title generation in `title.py` out of the route modules into the package that owns each, and let `buddy_agent_stream` hand its event mapping to the Buddy. | route handlers over 45 lines 7 → 0 (with B1) |
| B3 | Add the `forbidden` contracts for `api.routes` and `fastapi` ([§3.4](03-target.md#enforcement)). | rules 3 and 4 enforced |

### Stage C: one package per feature

Moves only.

| PR | Change |
|---|---|
| C1 | `onboarding` → `buddy` (agent, open, compact, persona, query fence, vocabulary). |
| C2 | `onboarding` → `assembly` (assembly stages, phase, orientation, diagram, starter work, their models, corpus fingerprint gate, citations, progress, similarity). |
| C3 | Add the `independence` contract over `buddy`, `assembly`, `insights`, `projects`, `skills`. |

### Stage D: long functions (P7)

Each PR removes one of the recurring shapes from [§2.6](02-as-is.md#26-long-functions-and-raw-values).
Prompt goldens and contract suites must stay green without edits.

| PR | Change | Exit measure |
|---|---|---|
| D1 | `phase.stream_phase` becomes a declared sequence of `assembly` stages. | 173 → ≤ 60 |
| D2 | Same for `orientation.stream_orientation` and `diagram.stream_diagram`. | 161, 161 → ≤ 60 |
| D3 | Same for `starter_work.stream_starter_work_pool`. | 148 → ≤ 60 |
| D4 | `buddy_agent.stream_agent_turn`: separate the hop loop, the tool dispatch and the event framing. | 219 → ≤ 60 |
| D5 | `buddy_open.stream_session`, `buddy_persona.build_persona`. | 121, 120 → ≤ 60 |
| D6 | `suggest_skills`, `chunk_text_context_aware`, `parse_pdf`. | 140, 143, 111 → ≤ 60 |
| D7 | Provider stream handling: `openai_client.chat_stream`, `chroma_store.query`. | 120, 116 → ≤ 60 |

### Stage E: typed identifiers (P8)

| PR | Change | Exit measure |
|---|---|---|
| E1 | Add `ProjectId`, `ArtifactId`, `ConnectorId` to `corpus`. Convert at the API boundary (`schemas` stay `str` on the wire). | types exist, `api` converts |
| E2 | Retrieval, ingestion and persistence take the types. | `str` ids 50 → ≤ 15 |
| E3 | Features and agent tools take the types. | 50 → 0 |

## 4.3 Order and dependencies

A → B → C is the order that keeps each PR small: A makes `corpus` and the layer rule exist, B removes
the orchestration that A's layer rule would otherwise flag in `api`, and C is a rename that is only
cheap once the imports are clean. D and E are independent of C and of each other and can run in
parallel branches after A.

## 4.4 Open questions

1. Whether to give each feature its own wire models (splitting `api/schemas.py`) once A4 is done.
   It changes no structure, so it is left to a later pass.
2. Whether the BM25 index should be per project. Nothing measures it today; track the index size
   after stage A before deciding.
3. Whether `agents` stays a separate layer or becomes a sub-package of `buddy`. Today only the
   Buddy uses it; keep it until a second consumer or a third tool set appears.
