# ADR (proposed) 001: The corpus vocabulary is a leaf package

**Status:** Proposed (2026-10-22), awaiting confirmation.

## Context

`.importlinter` says `rag` is above `ingestion`, which is above `store`. The code disagrees: `store`
imports `rag.types`, `rag.source_filter`, `rag.filters`, `ingestion.source_role` and
`ingestion.metadata_store`; `ingestion` imports `rag.types`, `rag.filters` and `rag.source_filter`.
The contract only passes because it ignores those 14 imports (as-is §2.3).

All of them have one cause. The types that describe the indexed material (`Chunk`, `ScoredChunk`,
`RetrievalFilters`, `SourceSystem`, `SourceRole`, `SourceExclusions`) and the pure helpers around
them (project-id encoding, filter predicates) were written where they were first needed, in `rag`
and `ingestion`. The lowest layer, `store`, needs the same vocabulary, so it imports upward.

Drivers: Q1 project isolation (the filter semantics must be one definition), Q5 changeability.

## Options

1. **Keep the types where they are and widen the contract.** Declare `rag`, `ingestion` and `store`
   one layer. Cheap, but it deletes the only structure the service has and hides the cycles for good.
2. **Define protocols in `store` and make `rag` and `ingestion` implement them.** Inverts the
   dependency, but the vocabulary is data, not behaviour: protocols for `Chunk` and a filter
   dataclass would be the types again under another name.
3. **Move the vocabulary into its own leaf package, `corpus`,** with no I/O and no dependency on our
   other packages. Every layer may import it.

## Decision

Option 3. `corpus` holds the chunk and filter types, `SourceRole` / `SourceExclusions`, the
project-id encoding and metadata keys, and the pure predicates (`matches_retrieval_filters`,
`has_narrowing_filters`). Chroma's `where` translation moves to `store`, because only the Chroma
store uses it. The SQLite metadata and source-state stores move to `store` as well, since
`ChromaVectorStore` already depends on them.

## Consequences

- `ignore_imports` can shrink to zero and the layers contract becomes true (migration stage A).
- The project-isolation semantics have one home, so the fail-closed tests live next to them.
- `corpus` is a place people will want to add things. The contract forbids it from importing our
  other packages, and anything that does I/O does not belong there.
- A pure move: no behaviour changes, and every existing test keeps its assertions.
