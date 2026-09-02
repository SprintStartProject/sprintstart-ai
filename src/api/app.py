import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()

from api.dependencies import get_llm, get_store  # noqa: E402
from api.routes import (  # noqa: E402
    blueprints,
    buddy,
    chat,
    diagram,
    grading,
    health,
    ingest,
    ingest_run,
    insights,
    knowledge_gaps,
    onboarding,
    orientation,
    projects,
    sources,
    starter_work,
    summaries,
    title,
    vector_db,
)
from llm.errors import LLMUnavailableError  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    try:
        get_llm().embed("ping")
    except ValueError as exc:
        logger.error("Embedding model is not configured: %s", exc)
        raise
    except LLMUnavailableError as exc:
        logger.warning("LLM backend unreachable at startup: %s", exc)

    # Construct Chroma before requests can fan out into chat and asynchronous
    # analytics on separate worker threads. The adapter also serialises cold
    # cache misses, but warming here removes that latency and race window from
    # the first user request altogether.
    try:
        get_store()
    except Exception as exc:
        # Match the LLM connectivity warm-up: a transient dependency failure
        # must not prevent the service from starting, and the uncached next
        # request will retry construction.
        logger.warning("Chroma store unavailable at startup: %s", exc)
    yield


app = FastAPI(
    title="SprintStart AI Service",
    version="0.1.0",
    description=(
        "RAG-based AI service. Exposes document ingestion and streaming chat. "
        "All streaming responses use Server-Sent Events (SSE)."
    ),
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})


api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(chat.router)
api_router.include_router(buddy.router)
api_router.include_router(ingest.router)
api_router.include_router(ingest_run.router)
api_router.include_router(title.router)
api_router.include_router(vector_db.router)
api_router.include_router(onboarding.router)
api_router.include_router(orientation.router)
api_router.include_router(diagram.router)
api_router.include_router(starter_work.router)
api_router.include_router(blueprints.router)
api_router.include_router(summaries.router)
api_router.include_router(sources.router)
api_router.include_router(knowledge_gaps.router)
api_router.include_router(insights.router)
api_router.include_router(grading.router)
api_router.include_router(projects.router)

app.include_router(api_router)
