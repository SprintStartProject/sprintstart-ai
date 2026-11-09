from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status

from api.dependencies import get_llm, get_store
from api.schemas import (
    ValidationErrorResponse,
    VectorDbChunkListResponse,
    VectorDbChunkResponse,
    VectorDbScoredChunkResponse,
    VectorDbSearchRequest,
    VectorDbSearchResponse,
    VectorDbStatusResponse,
)
from corpus.types import Chunk, ScoredChunk
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from store.base import VectorStore

router = APIRouter(prefix="/vector-db", tags=["vector-db"])


def _chunk_to_response(chunk: Chunk) -> VectorDbChunkResponse:
    return VectorDbChunkResponse(
        id=chunk.id,
        artifact_id=chunk.artifact_id,
        filename=chunk.filename,
        text=chunk.text,
        position=chunk.position,
        kind=chunk.kind,
        start_line=chunk.start_line,
        start_page=chunk.start_page,
    )


def _scored_chunk_to_response(
    scored_chunk: ScoredChunk,
) -> VectorDbScoredChunkResponse:
    return VectorDbScoredChunkResponse(
        id=scored_chunk.id,
        artifact_id=scored_chunk.artifact_id,
        filename=scored_chunk.filename,
        text=scored_chunk.text,
        position=scored_chunk.position,
        kind=scored_chunk.kind,
        score=scored_chunk.score,
        start_line=scored_chunk.start_line,
        start_page=scored_chunk.start_page,
    )


def _store_backend_name(store: VectorStore) -> str:
    name = type(store).__name__.lower()

    if "chroma" in name:
        return "chroma"

    return name


@router.get(
    "/status",
    response_model=VectorDbStatusResponse,
    summary="Get vector database status",
    description="Returns the configured vector store backend and chunk count.",
)
def get_vector_db_status(
    store: Annotated[VectorStore, Depends(get_store)],
) -> VectorDbStatusResponse:
    try:
        return VectorDbStatusResponse(
            backend=_store_backend_name(store),
            chunk_count=store.count(),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to read vector database status: {exc}",
        ) from exc


@router.get(
    "/chunks",
    response_model=VectorDbChunkListResponse,
    summary="List vector database chunks",
    description=(
        "Lists stored chunks with pagination. Embeddings and internal Chroma "
        "objects are not returned."
    ),
)
def list_chunks(
    store: Annotated[VectorStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> VectorDbChunkListResponse:
    try:
        chunks = store.list_chunks(limit=limit, offset=offset)
        total = store.count()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list vector database chunks: {exc}",
        ) from exc

    return VectorDbChunkListResponse(
        items=[_chunk_to_response(chunk) for chunk in chunks],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.get(
    "/artifacts/{artifact_id}/chunks",
    response_model=VectorDbChunkListResponse,
    summary="List chunks for an artifact",
    description=(
        "Returns paginated vector database chunks belonging to one artifact. "
        "Embeddings and internal Chroma objects are not returned."
    ),
    responses={
        404: {
            "model": ValidationErrorResponse,
            "description": "No chunks found for the artifact.",
        }
    },
)
def list_chunks_by_artifact(
    artifact_id: str,
    store: Annotated[VectorStore, Depends(get_store)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> VectorDbChunkListResponse:
    try:
        total = store.count_by_artifact(artifact_id)

        if total == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No chunks found for artifact_id {artifact_id!r}.",
            )

        chunks = store.list_chunks_by_artifact(
            artifact_id=artifact_id,
            limit=limit,
            offset=offset,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list chunks for artifact {artifact_id!r}: {exc}",
        ) from exc

    return VectorDbChunkListResponse(
        items=[_chunk_to_response(chunk) for chunk in chunks],
        limit=limit,
        offset=offset,
        total=total,
    )


@router.delete(
    "/artifacts/{artifact_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete chunks for an artifact",
    description=(
        "Deletes all vector database chunks belonging to the given artifact_id. "
        "Returns 204 when chunks were deleted and 404 when no chunks exist."
    ),
    responses={
        204: {"description": "Artifact chunks deleted successfully."},
        404: {
            "model": ValidationErrorResponse,
            "description": "No chunks found for the artifact.",
        },
    },
)
def delete_artifact_chunks(
    artifact_id: str,
    store: Annotated[VectorStore, Depends(get_store)],
) -> Response:
    try:
        deleted_count = store.delete(artifact_id)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete chunks for artifact {artifact_id!r}: {exc}",
        ) from exc

    if deleted_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No chunks found for artifact_id {artifact_id!r}.",
        )

    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/search",
    response_model=VectorDbSearchResponse,
    summary="Search vector database",
    description=(
        "Embeds the query with the configured LLM backend and searches the "
        "configured vector store directly. Intended for debugging and admin use."
    ),
    responses={
        503: {
            "model": ValidationErrorResponse,
            "description": "LLM backend unavailable while embedding the query.",
        }
    },
)
def search_vector_db(
    request: VectorDbSearchRequest,
    store: Annotated[VectorStore, Depends(get_store)],
    llm: Annotated[LLMClient, Depends(get_llm)],
) -> VectorDbSearchResponse:
    try:
        embedding = llm.embed(request.query)
        chunks = store.query(
            embedding=embedding,
            top_k=request.top_k,
            min_score=request.min_score,
        )
    except LLMUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to search vector database: {exc}",
        ) from exc

    return VectorDbSearchResponse(
        items=[_scored_chunk_to_response(chunk) for chunk in chunks]
    )
