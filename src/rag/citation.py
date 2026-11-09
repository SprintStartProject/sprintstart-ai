from corpus.types import Citation, ScoredChunk


def build_citations(chunks: list[ScoredChunk]) -> list[Citation]:
    return [
        Citation(
            artifact_id=chunk.artifact_id,
            start_line=chunk.start_line,
            start_page=chunk.start_page,
        )
        for chunk in chunks
    ]
