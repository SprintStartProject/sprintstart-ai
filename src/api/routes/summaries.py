from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from api.dependencies import get_llm, get_store
from api.schemas import ArtifactSummaryRequest
from api.sse import stream_events
from corpus.ids import ArtifactId
from corpus.types import Chunk
from llm.base import LLMClient
from store.base import VectorStore
from summaries.artifact_summary import (
    ArtifactSummaryAgent,
    SummaryEvent,
    SummaryStage,
    SummaryToken,
)

router = APIRouter(prefix="/artifacts", tags=["artifacts"])


def _payload(event: SummaryEvent) -> dict[str, object]:
    if isinstance(event, SummaryStage):
        return {"type": "stage", "name": event.name, "detail": event.detail}
    if isinstance(event, SummaryToken):
        return {"type": "token", "content": event.text}
    return {
        "type": "citation",
        "artifact_id": event.artifact_id,
        "filename": event.filename,
        "source_url": event.source_url,
    }


def _chunks_or_404(
    store: VectorStore, artifact_id: ArtifactId, max_chunks: int, missing: str
) -> list[Chunk]:
    chunks = store.list_chunks_by_artifact(
        artifact_id=artifact_id,
        limit=max_chunks,
        offset=0,
    )
    if not chunks:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=missing)
    return chunks


@router.post(
    "/{artifact_id}/summary",
    summary="Summarize an artifact (streaming)",
    response_class=StreamingResponse,
    description=(
        "Streams an AI-generated summary of the artifact over Server-Sent Events.\n\n"
        "Event sequence:\n"
        "1. `stage` events while notes are gathered from the source (not streamed "
        "itself -- internal working notes, not the final summary)\n"
        "2. `token` events streaming the final summary text\n"
        "3. One `citation` event per source cited in the summary\n"
        "4. Exactly one `done` event\n\n"
        "On error, a single `error` event is emitted instead and the stream closes."
    ),
)
def summarize_artifact(
    artifact_id: str,
    body: ArtifactSummaryRequest,
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> StreamingResponse:
    chunks = _chunks_or_404(
        store,
        ArtifactId(artifact_id),
        body.max_chunks,
        f"Artifact {artifact_id!r} was not found or has no chunks.",
    )
    previous_chunks: list[Chunk] | None = None
    if body.previous_artifact_id is not None:
        previous_chunks = _chunks_or_404(
            store,
            ArtifactId(body.previous_artifact_id),
            body.max_chunks,
            f"Previous artifact {body.previous_artifact_id!r} "
            "was not found or has no chunks.",
        )

    def payloads() -> Iterator[dict[str, object]]:
        agent = ArtifactSummaryAgent(llm)
        for event in agent.events(ArtifactId(artifact_id), chunks, previous_chunks):
            yield _payload(event)
        yield {"type": "done"}

    return StreamingResponse(
        stream_events(payloads(), what="artifact summary stream"),
        media_type="text/event-stream",
    )
