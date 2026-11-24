# 3 Target architecture

> **Status: draft (2026-10-22), proposed.** The structure below follows from the
> [drivers](01-drivers.md) and removes the findings F1–F7 of the [as-is](02-as-is.md#29-findings).
> It keeps what already works (the layer idea, the stateless contract, the staged assembly jobs) and
> changes the minimum needed to make the layering true.

## 3.1 Layers

Calls go down. Nothing imports a package above it.

| Layer | Packages | Responsibility |
|---|---|---|
| 1 Adapters | `api` | HTTP: routes, wire models (`schemas`), SSE framing, the composition root (`dependencies`). No business flow. |
| 2 Features | `buddy`, `assembly`, `insights`, `projects`, `skills` | One user-facing capability each. Independent of each other. Own their prompts, models and orchestration. |
| 3 Agent kit | `agents` | The tool registry and the retrieval / grep tools the Buddy loop uses. |
| 4 Retrieval | `rag` | Hybrid search, context window, citations, source classification by path. |
| 5 Ingestion | `ingestion` | Parse, chunk, embed and store an artifact, as one pipeline. |
| 6 Persistence | `store` | `VectorStore`, the Chroma implementation, and the ingestion metadata and source-state stores. |
| 7 Leaves | `corpus`, `llm` | `corpus`: the vocabulary of the indexed material (chunk, filters, ids, source role). `llm`: the model interface and its three providers. Neither imports another of our packages. |

`corpus` is new; the rest exist. The change from today is that `corpus` takes over what `rag`,
`ingestion` and `store` were all importing from each other, and `onboarding` becomes `buddy` and
`assembly`.

```
api
 └─ buddy · assembly · insights · projects · skills        (independent of each other)
     └─ agents
         └─ rag
             └─ ingestion
                 └─ store
                     └─ corpus · llm
```

## 3.2 Principles

| # | Principle | Removes | Driver |
|---|---|---|---|
| P1 | **Calls go down; the contract has no exceptions.** Every layer may import the layers below it and nothing above. `ignore_imports` is empty. | F1 | Q5, Q6 |
| P2 | **The corpus vocabulary is a leaf.** `Chunk`, `ScoredChunk`, `RetrievalFilters`, `SourceSystem`, `SourceRole`, `SourceExclusions`, the project-id encoding and the pure filter predicates live in `corpus`, with no I/O and no dependency on our other packages. Chroma's `where` translation lives in `store`. | F1 | Q1, Q5 |
| P3 | **Persistence owns its stores.** The SQLite metadata and source-state stores sit beside the vector store, because `ChromaVectorStore` already depends on them. | F1 | Q5 |
| P4 | **Routes adapt, services orchestrate.** A route validates the request, calls one function and frames the response. The ingest flow (parse → classify → chunk → embed → store → revoke) is one `ingestion.pipeline` used by both ingest endpoints. | F2 | Q6, Q3 |
| P5 | **Features own their models.** A feature returns its own result types; `api` maps them to wire models. Features never import `api`. | F3 | Q5 |
| P6 | **A feature is one thing.** The Buddy (a conversation with tools) and the assembly jobs (staged, deterministic generation) are separate packages that share only what they import from lower layers. | F4 | Q6 |
| P7 | **An assembly job is a declared sequence of shared stages**, not one long function. `assembly.py` already holds the stages; each job composes them. | F5 | Q3, Q5 |
| P8 | **Identifiers are types.** `ProjectId`, `ArtifactId` and `ConnectorId` are `NewType`s in `corpus`, converted from the wire at the API boundary. | F6 | Q1 |

### What deliberately stays

- **The stateless contract** (backend ADR-005). The service stores the corpus and its ingestion
  metadata, nothing about users or conversations.
- **Project isolation in both halves of hybrid retrieval.** P2 moves the predicates, it does not
  change them; the isolation tests move with them.
- **One process-wide BM25 index.** It is a known memory exposure (as-is §2.7), but no driver
  requires a per-project index yet. The migration leaves it and tracks the corpus size.
- **`api/schemas.py` as the wire contract.** It is large (1,563 lines) because it is the contract;
  splitting it by feature is possible later but changes nothing structural.

## 3.3 Rules

1. A package imports only from the layers below it, and from `corpus` and `llm`.
2. The five features do not import each other.
3. `api` is the only package that imports `fastapi`. Features, retrieval, ingestion and persistence
   are plain Python.
4. A route function has no loop over artifacts, chunks or model output. If it needs one, the loop
   belongs to a feature or `ingestion`.
5. A new model provider touches `llm` and `api.dependencies` only. A new assembly job touches
   `assembly` and one route.
6. No function over 60 lines is added, and the ratchet does not let an existing one grow.

## 3.4 Enforcement {#enforcement}

| Rule | How it is kept |
|---|---|
| 1, P1 | `.importlinter` layers contract with the seven layers above and an empty `ignore_imports`. The existing ratchet already fails when an ignored entry stops matching, so the list can only shrink to zero. |
| 2, P6 | An `independence` contract over the five feature packages. |
| 3 | A `forbidden` contract: everything except `api` may not import `fastapi`. |
| 4, P4 | A `forbidden` contract: `api.routes` may not import parsers, chunkers or the Chroma store directly. |
| P2 | A `forbidden` contract: `corpus` imports none of our other packages. |
| 6 | `ruff-limits` baseline, which may only shrink. |
| P8 | `pyright` on `src/`: swapping a `ProjectId` and an `ArtifactId` becomes a type error. |
| All | The ratchet runs on every PR. |
