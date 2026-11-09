# 4 Migration

> **Status: in progress (updated 2026-10-22).** Stages A, B and C are done, the folder structure pass is
> open as #101, stage D is half done and stage E has begun. From the [as-is](02-as-is.md) to the
> [target](03-target.md) in small PRs. Every PR passes the full gate set (`ruff`, limits, `pyright`,
> `pytest`, `lint-imports`, the ratchet, and mutation testing on the code it adds) and none changes
> behaviour or an HTTP contract. Each stage has an exit measure, so progress shows up as numbers.

## 4.1 Tracked numbers

*Before* is the as-is snapshot (`c188b82`, [§2](02-as-is.md)). *Now* is `main` at `0aa3429`, after
PRs #88 to #97, measured with the same scripts. The package count includes the folder structure pass (#101), which is open.

| Number | Before | Now | Target | Source |
|---|---:|---:|---:|---|
| Ignored imports in `.importlinter` | 14 | **0** | 0 | `lint-imports` |
| Mutual package pairs | 5 | **0** | 0 | AST pass |
| Features importing `api` | 2 | **0** | 0 | `lint-imports` |
| Route modules that import parsers, chunkers or the Chroma store | 2 | **0**, now enforced | 0 | forbidden contract |
| Route handlers over 45 lines | 7 | **0** | 0 | AST |
| Ingest pipelines | 2 | **1** | 1 | `ingestion.pipeline` |
| Functions over 100 lines | 14 | **7** | 0 | AST |
| Functions over 60 lines | 40 | 36 | ≤ 15 | AST |
| Functions over 30 lines | 111 | 115 | — | AST |
| Functions | 499 | 579 | — | AST |
| Files over 400 lines | 15 | 10 | — | line count |
| `*_id` parameters typed `str` | 50 | 49 | 0 | AST |
| Packages named in the layers contract | 10 | 12 (with #101) | 12 | `.importlinter` |
| Duplicated lines | 119 | 82 | — | `metrics.sh` |
| Dead-code findings | 15 | 15 | — | `metrics.sh` |
| `ruff-limits` baseline | 56 entries, 93 violations | 53 entries, 74 violations | 0 | `ruff-limits-baseline.json` |
| Passing tests | 1,079 | 1,154 | — | `pytest` |
| Source lines | 18,701 | 19,639 | — | line count |

Three things in this table are less good than they look and are worth reading as such:

- **The count of long functions barely moves below 100 lines.** Splitting a 170-line function gives a
  60-line function and ten small ones, so the number over 30 lines went *up* by four and the number
  over 60 went down by only four. What fell is the long tail: seven of the 14 functions over 100 lines
  are gone, and the longest is now 143 lines instead of 239.
- **Source grew by 5 %.** Named stages, small dataclasses and the shared pipeline cost lines. The
  gain is that each function now does one thing, not that there is less code.
- **Coverage was not re-measured here.** The recorded 92.6 % is enforced by CI, which the changed
  code passes.

## 4.2 Stages

### Stage A: make the layering true — done

Moves only. No logic changes, and every PR left the suite unchanged.

| PR | Change | Ignored imports |
|---|---|---|
| A1 · #88 | Created `corpus` and moved the chunk and filter types, `SourceRole`, `SourceExclusions` and the pure filter predicates into it. | 14 → 4 |
| A2 · #89 | Moved `where_filter_for_chroma` from `rag` to `store`. | 4 → 3 |
| A3 · #90 | Moved `IngestionMetadataStore` and `SourceStateStore` into `store`. | 3 → 2 |
| A4 · #91 | `projects.evaluation` and `skills.suggestion` return feature result types; `api` maps them to the wire models. | 2 → 0 |

The planned A5 (rewrite `.importlinter`) needed no PR of its own: `corpus` joined the bottom layer in
A1 and the `ignore_imports` key was deleted in A4, so a new violation now fails the check.

### Stage B: routes adapt, services orchestrate — done

| PR | Change |
|---|---|
| B1 · #92 | One ingest pipeline: shared steps in `ingestion/pipeline.py`, the upload and synced-artifact flows in `upload_ingest.py` and `sync_ingest.py`. `_ingest_one` 239 lines and `ingest` 156 lines are gone. Sixteen characterization tests were written first. |
| B2 · #93 | Answer grading and title generation moved to `onboarding/grading.py` and `onboarding/titles.py`. |
| B2 · #94 | The Buddy and summary streams share `api.sse.stream_events`; `ArtifactSummaryAgent.events` owns the loop over model output; the buffered and streaming Buddy endpoints share one set of turn options. |
| B3 · #95 | Two `forbidden` contracts: `api.routes` may not import parsers, chunkers or the Chroma store directly, and only `api` may import `fastapi` or `starlette`. |

Two changes from the original plan. The membership rewrite in `artifact_projects.py` stayed where it
is: its handler is 36 lines and imports nothing it should not, so moving it would have been churn.
And the `forbidden` contract allows indirect imports, because the ingest routes legitimately reach
the parsers through the pipeline.

### Stage C: one package per feature — done (#99)

Moves only. `onboarding` held two features with no import between them, so the split needed no new
seam.

| PR | Change |
|---|---|
| C1+C2 · #99 | `onboarding` → `buddy` (agent, events, evidence, opening, compact, persona, query fence, vocabulary) and `assembly` (the four jobs with their drafts and models, grading, titles, citations, progress, similarity, `stages`, `fingerprint`). Tests mirror the layout and the shared golden helper moves to `tests/support`. |
| C3 · #99 | Both packages join the independent features in the layers contract. A planted `buddy` → `assembly` import fails it. |

Two files were renamed to avoid confusion: `assembly.py` became `assembly/stages.py`, and
`onboarding/corpus.py` became `assembly/fingerprint.py`, so it is not mistaken for the top-level `corpus`
package. The planned separate PRs C1 to C3 were one, because the split is only checkable once both
packages and the contract exist together.

### Folder structure pass — open (#101)

Renames only, after stage C. Three groupings, one commit each:

- `assembly` (21 flat files) became four job packages, `phase/`, `orientation/`, `diagram/` and
  `starter_work/`, each with `job.py`, `draft.py` and `models.py`.
- `ingestion` (17 flat files) became `parsing/`, `chunking/` and `pipeline/` (`steps`, `upload`, `sync`).
  The "routes adapt" contract now names two packages instead of eight modules.
- `agents` was removed. It held the Buddy's tool kit and one unrelated file, so the tools became
  `buddy/tools/` and `artifact_summary` became its own feature, `summaries/`. The layers contract lost
  a layer and gained a feature.

Left alone on purpose: `projects` (139 lines) and `skills` (303 lines) are small, but each is one
cohesive feature, and merging them would weaken the independence contract. `store` still holds both
the vector store and the SQLite stores, and `api/schemas.py` is still one 1,563-line file.

### Stage D: long functions (P7) — half done

| PR | Change | Result |
|---|---|---|
| D1 · #96 | The four assembly jobs become named stages. Each job is two files: `<job>.py` (the stages) and `<job>_draft.py` (what the model drafts and how it is read); the starter-work result types move to `starter_work_models.py`. | 173 → 65, 161 → 64, 161 → 62, 148 → 78 |
| D2 · #97 | The Buddy turn becomes a `_Turn` with named hop, search, record and forced-answer methods; the events move to `buddy_events.py` and the search tools to `buddy_evidence.py`. | 219 → 64 |

The planned form of D1 ("a declared sequence of shared `assembly` stages") was not built. The four jobs
share the retrieval and generation stages that were already in `assembly.py`, but their grounding and
outcome steps differ too much for one generic runner to read better than four short pipelines.

`suggest_skills` (140 → 36) is #100. Still over 100 lines: `chunk_text_context_aware` (143),
`buddy.opening.stream_session` (121), `buddy.persona.build_persona` (120),
`openai_client.chat_stream` (120), `chroma_store.query` (116) and `parse_pdf` (111).

### Stage E: typed identifiers (P8) — begun

`corpus/ids.py` holds `ArtifactId` and `SourceId`, introduced with B1 because the ratchet rejects new
raw `*_id: str` fields. They are used by the ingest pipeline, the summary agent and their routes, and
converted from the wire at the route. `ProjectId` and `ConnectorId` are not added yet:
`api.schemas.ProjectId` is already a validated string alias, so a domain `ProjectId` needs a decision
about naming first. Remaining raw ids: `artifact_id` 28, `project_id` 15, `connector_id` 4.

## 4.3 What the gates taught us

The gates shaped the work, and three lessons repeat:

- **A file over the size limit may not grow, whatever the reason.** Extracting stages added scaffolding,
  so the four assembly jobs and the Buddy agent each had to be split to shrink. That is why a
  refactoring of functions produced new modules.
- **Moved code counts as added code for mutation testing.** Every move needed tests that pin what
  moved: the skill suggestion change (#91) failed the gate at 58 % until it had them, and the
  streaming and assembly PRs were hardened with them before pushing. Prompt goldens do most of this
  work.
- **Characterization tests before extraction find wrong assumptions.** Writing them for the ingest
  paths showed that the sync route prepends the title before the size check and that an unreadable
  PDF completes with zero chunks. The refactoring then kept both, on purpose.

One side effect to know about: log messages emitted by moved code now come from the new module's
logger (`assembly.phase.draft`, not the module the code used to live in).

## 4.4 Open questions

1. Whether to give each feature its own wire models (splitting `api/schemas.py`, 1,563 lines). It
   changes no structure, so it is left to a later pass.
2. Whether the BM25 index should be per project. Nothing measures it today; track the index size
   before deciding.
3. What to call a domain `ProjectId` next to the wire alias of the same name.
4. What to do about the files that stay over 400 lines and are not on this plan: `chroma_store.py`
   (961), `metadata_store.py` (741), the three provider clients (653, 433, 428) and `buddy_persona.py`
   (645). They are grandfathered by the ratchet and not covered by any stage; a stage F would own them.
   `buddy/agent.py` (513) and the two chunkers (445, 440) are also over, but smaller and partly
   covered by stage D.
