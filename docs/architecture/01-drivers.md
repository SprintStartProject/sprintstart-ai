# 1 Drivers

> **Status: draft (2026-10-22), awaiting confirmation.** Nobody wrote these drivers down at the
> start. They were reconstructed from the product, from the backend's confirmed
> [drivers](https://github.com/LinseCed/sprintstart-backend/blob/main/docs/architecture/01-drivers.md),
> and from what the code, the contracts and the deployment show this service had to cope with. The
> AI service has the same product goals as the backend, so this document only restates what changes
> for a Python service that holds no business state.

## 1.1 What the service is for

The backend owns users, projects, paths and boards. This service owns everything that needs a
language model or a vector index:

- **Ingest:** parse, chunk and embed a project's sources (code, text, PDF, images) into one index per
  deployment, scoped by project.
- **Retrieve:** hybrid search (vector + BM25, fused with RRF) with citations.
- **Buddy:** one tool-using turn of the assistant, a greeting, and the compaction of old turns into
  a memory note. Tools that need business data are run by the backend (backend ADR-005).
- **Assemble:** deterministic, staged generation of grounded content: phase content, task
  orientation, subject diagrams and starter-work candidates.
- **Insights:** FAQ grouping, knowledge-gap detection, skill suggestion, project evaluation,
  summaries and titles.

## 1.2 Stakeholders

| Stakeholder | Expects from the architecture |
|---|---|
| Backend team | A stable, contract-tested HTTP surface. A failed call never leaves the backend half-updated. |
| AI sub-team (rotating students) | Finding the code for a feature quickly. Changing a prompt or a pipeline without breaking retrieval. |
| New hire / PM (via the backend) | Answers that cite the project's own material and never show another project's data. |
| Operators | A single container that starts within the probe window and fits the memory limit. |
| Examiners | A traceable path from goals to structure, and evidence that the structure holds. |

## 1.3 Quality goals

Ordered by priority. When two goals conflict, the higher one wins.

| # | Goal | Why it matters here |
|---|---|---|
| Q1 | **Project isolation** | Every retrieval is scoped to the caller's projects and fails closed: a chunk with no project is visible to nobody. A leak here is a leak of company knowledge. |
| Q2 | **Groundedness** | Generated content must cite the evidence it was built from, and drop what the evidence does not support. An uncited claim is worse than a missing one. |
| Q3 | **Testability without a model** | LLM output is nondeterministic. Everything around the call (retrieval, parsing, validation, event framing) must be testable with a scripted model, and the wire contracts pinned. |
| Q4 | **Robustness against the model** | Calls take 5–30 s and fail midway. A failed or cancelled call ends in a readable `error` event, never a hung stream or a half-written result. |
| Q5 | **Changeability** | New pipelines, prompts and model providers keep arriving. Adding one must not touch retrieval or ingestion. |
| Q6 | **Understandability** | The team rotates. A feature has to be findable and readable on its own. |
| Q7 | **Operability on limited resources** | One instance on a shared university cluster. The BM25 index and the embedding calls must stay within the memory and time limits. |

## 1.4 Quality scenarios

*Measure* names what the [migration](04-migration.md) tracks.

| Goal | Scenario | Measure |
|---|---|---|
| Q1 | A request names project A. | `where_filter_for_chroma` and the BM25 side both apply the project filter, verified by tests that insert a chunk for project B and one without a project. |
| Q2 | The model returns a section whose evidence ids are not in the retrieved set. | The section is dropped, not shown. Contract tests per pipeline pin this. |
| Q3 | A developer changes a prompt. | Prompt goldens show the diff. The suite runs without Ollama, OpenAI or Anthropic. |
| Q4 | The provider drops the stream halfway. | The SSE stream ends with an `error` event; nothing is persisted by this service. |
| Q5 | A developer adds a new assembly pipeline (e.g. "glossary"). | New code lives in one feature package; no change to `rag`, `ingestion`, `store` or `llm`. |
| Q5 | The team switches the embedding provider. | The change is confined to `llm` and the composition root. |
| Q6 | A new developer has to fix a bug in ingestion of PDFs. | The code is under `ingestion`, nothing in `api` is involved, and no function on the path is longer than 60 lines. |
| Q7 | The corpus grows. | The one process-wide BM25 index is rebuilt only when the shared corpus revision changes. Its memory grows with the whole corpus, so this is measured, not assumed. |

## 1.5 Constraints

| Constraint | Source |
|---|---|
| Python, FastAPI, one deployable container | ADR-001 |
| Everything LLM, RAG and embeddings lives here; business rules and permissions live in the backend | ADR-001, backend ADR-005 |
| ChromaDB as the vector store, SQLite for ingestion metadata | Deployment |
| Three model providers behind one interface: Ollama, OpenAI-compatible, Anthropic | Team decision, `llm/` |
| Wire contracts with the backend are JSON (`api/schemas.py`) and JSON Schema (`contracts/`) | Seam 1 of the project |
| Deployed by ArgoCD to the university cluster | `sprintstart-k8s`, `uni-cluster` |

## 1.6 What is deliberately not a driver

- **Horizontal scaling.** One instance per environment is enough, so process-local caches (the
  Chroma client cache, BM25 indexes, the `lru_cache`d clients) are acceptable as long as Q7 holds.
- **Business state.** The service stores the corpus and its ingestion metadata, nothing about users,
  paths or conversations. Anything a turn needs is sent with the request.
- **Agent autonomy.** The Buddy loop is bounded and its writes become proposals in the backend; the
  assembly pipelines are staged and deterministic by design, not agentic.
