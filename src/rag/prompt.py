from rag.types import ScoredChunk


def chunk_header(chunk: ScoredChunk) -> str:
    """Label a chunk for the model — used for the buddy's tool results."""
    parts = [chunk.artifact_type or "FILE", chunk.filename]
    if chunk.language:
        parts.append(chunk.language)
    if chunk.source_url:
        parts.append(chunk.source_url)
    return "[" + " | ".join(parts) + "]"
