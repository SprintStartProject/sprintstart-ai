from typing import NewType

#: Identifies an artifact across the vector store and the ingestion metadata.
ArtifactId = NewType("ArtifactId", str)

#: Identifies an artifact at its source, e.g. ``github:owner/repo:FILE:src/App.kt``.
SourceId = NewType("SourceId", str)

#: Identifies one chunk in the vector store.
ChunkId = NewType("ChunkId", str)
