# API reference

Everything lives under `/api/v1`. Streaming endpoints use Server-Sent Events. Full schemas are at `/docs` on a running instance.

<details open>
<summary><b>Ingestion and the vector store</b></summary>

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/ingest` | Parse, chunk and embed one document. Text, code, PDF and images (base64, needs a vision model; without one, images yield no chunks). Re-ingesting an `artifact_id` replaces its chunks. |
| `POST` | `/ingest/sync` | Batch-ingest a completed GitHub ingestion run. |
| `POST` | `/artifacts/projects/sync` | Rewrite which projects an indexed artifact belongs to. |
| `DELETE` | `/projects/{project_id}/memberships` | Drop a deleted project from the whole corpus. |
| `PATCH` | `/connectors/{id}`, `/sources/{id}` | Enable or disable a connector or its sources. |
| `GET` | `/vector-db/status`, `/vector-db/chunks`, `/vector-db/artifacts/{id}/chunks` | Inspect what is indexed. |
| `POST` | `/vector-db/search` | Search the index directly. |
| `DELETE` | `/vector-db/artifacts/{id}` | Delete an artifact's chunks. |

</details>

<details open>
<summary><b>Chat and buddy</b></summary>

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/chat` | Cited, streamed answer scoped to a `projectId`. |
| `POST` | `/generate-title` | Short title from a prompt. |
| `POST` | `/artifacts/{id}/summary` | Streamed summary of one artifact. |
| `POST` | `/onboarding/buddy/open/stream` | Open a buddy visit and greet the hire. |
| `POST` | `/onboarding/buddy/agent` | One tool-using buddy turn (stateless). |
| `POST` | `/onboarding/buddy/compact` | Fold older turns into the mentor's durable memory note. |

</details>

<details open>
<summary><b>Onboarding</b></summary>

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/onboarding/path`, `/onboarding/path/yaml` | Personalized path, as SSE or YAML. |
| `POST` | `/onboarding/blueprints/generate` | Draft blueprints from a project's corpus. |
| `POST` | `/onboarding/phase`, `/onboarding/phase/stream` | Fill one `AI_ENHANCED` blueprint phase with steps and a knowledge check. |
| `POST` | `/onboarding/orientation`, `/onboarding/orientation/stream` | Task-scoped orientation packet. |
| `POST` | `/onboarding/diagram`, `/onboarding/diagram/stream` | Subject-scoped, cited diagram. |
| `POST` | `/onboarding/starter-work/mine`, `/onboarding/starter-work/mine/stream` | Mine open issues for starter-work candidates. |

</details>

<details open>
<summary><b>Insights and helpers</b></summary>

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/insights/knowledge-gaps/detect` | Documentation-coverage gaps per component. |
| `POST` | `/insights/faq/group`, `/classify`, `/groups/merge` | Group recurring questions, file a new one, propose merges. |
| `POST` | `/skills/suggest` | Suggest skills for a project role. |
| `POST` | `/grade-answers` | Semantically grade short-text knowledge-check answers. |
| `POST` | `/projects/{id}/industry/evaluate` | Infer a project's industry or domain from its artifacts. |
| `GET` | `/health` | Service and LLM backend health. |

</details>

### Chat stream events

`/chat` streams newline-delimited JSON events:

| Event | Meaning |
|---|---|
| `tool_use` | A capability the agent invoked, in order (`name`, `kind`). |
| `reasoning` | A live reasoning fragment. Never part of the answer or persisted content. |
| `token` | One fragment of the answer. |
| `citation` | A source chunk used for the answer. |
| `done` | End of stream. |
| `error` | Emitted instead of the above on failure. |
