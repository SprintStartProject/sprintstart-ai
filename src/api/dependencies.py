import logging
import os
from collections.abc import Callable
from functools import lru_cache

from fastapi import Depends

from agents.orchestrator import ChatOrchestrator
from ingestion.metadata_store import IngestionMetadataStore
from ingestion.source_state_store import SourceStateStore
from llm.anthropic_client import AnthropicClient
from llm.base import LLMClient
from llm.ollama_client import OllamaClient
from llm.openai_client import OpenAIClient
from llm.split_client import SplitLLMClient
from onboarding.orchestrator import OnboardingOrchestrator
from rag.retriever import get_bm25_cache
from rag.types import RetrievalFilters
from store.base import VectorStore
from store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


# Matches the Anthropic and OpenAI SDKs' own default. Applied to Ollama too,
# which otherwise waits forever: a wedged daemon would hang the request rather
# than surfacing as an error the route can turn into an SSE `error` event.
_DEFAULT_LLM_TIMEOUT_SECONDS = 600.0


def _llm_timeout() -> float | None:
    """Request timeout for every LLM backend, in seconds.

    ``LLM_TIMEOUT_SECONDS=0`` disables it — the pre-existing behaviour for
    Ollama, and worth keeping reachable for a long-running local batch.
    """
    raw = os.getenv("LLM_TIMEOUT_SECONDS", "").strip()
    if not raw:
        return _DEFAULT_LLM_TIMEOUT_SECONDS
    timeout = float(raw)
    return timeout if timeout > 0 else None


def _build_client(backend: str, is_embed: bool = False) -> LLMClient:
    backend = backend.lower()
    timeout = _llm_timeout()

    if backend in {"ollama", "local"}:
        num_ctx = os.getenv("OLLAMA_NUM_CTX", "").strip()
        return OllamaClient(
            host=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL"),
            vision_model=os.getenv("OLLAMA_VISION_MODEL"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            num_ctx=int(num_ctx) if num_ctx else None,
            timeout=timeout,
        )

    if backend in {"openai", "openai-compatible", "litellm"}:
        base_url = (
            (os.getenv("OPENAI_EMBED_BASE_URL") if is_embed else None)
            or os.getenv("OPENAI_BASE_URL")
            or "https://api.openai.com/v1"
        )
        api_key = (
            (os.getenv("OPENAI_EMBED_API_KEY") if is_embed else None)
            or os.getenv("OPENAI_API_KEY")
            or "unused"
        )
        return OpenAIClient(
            base_url=base_url,
            api_key=api_key,
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("OPENAI_EMBED_MODEL"),
            vision_model=os.getenv("OPENAI_VISION_MODEL"),
            timeout=timeout,
        )

    if backend in {"anthropic", "claude"}:
        raw_thinking_budget = os.getenv("ANTHROPIC_THINKING_BUDGET_TOKENS", "").strip()
        thinking_budget = int(raw_thinking_budget) if raw_thinking_budget else 0
        return AnthropicClient(
            api_key=os.getenv("ANTHROPIC_API_KEY") or "",
            chat_model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5"),
            vision_model=os.getenv("ANTHROPIC_VISION_MODEL"),
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
            thinking_budget_tokens=thinking_budget if thinking_budget > 0 else None,
            timeout=timeout,
        )

    raise ValueError(f"Unknown LLM backend: {backend!r}")


@lru_cache
def get_llm() -> LLMClient:
    chat = _build_client(
        os.getenv("LLM_BACKEND") or os.getenv("LLM_PROVIDER") or "local",
        is_embed=False,
    )
    embed_backend = os.getenv("EMBED_BACKEND") or os.getenv("EMBED_PROVIDER")
    if embed_backend is None:
        return chat

    return SplitLLMClient(
        chat=chat,
        embed=_build_client(embed_backend, is_embed=True),
    )


@lru_cache
def get_store() -> VectorStore:
    path = os.getenv("CHROMA_PATH", "").strip() or None
    if path is None:
        logger.warning(
            "CHROMA_PATH is not set — using ephemeral in-memory store, "
            "data will not persist"
        )
    return ChromaVectorStore(path=path)


@lru_cache
def get_ingestion_metadata_store() -> IngestionMetadataStore:
    path = os.getenv("APP_DB_PATH", "").strip() or "data/sprintstart.db"
    return IngestionMetadataStore(path=path)


@lru_cache
def get_source_state_store() -> SourceStateStore:
    path = os.getenv("APP_DB_PATH", "").strip() or "data/sprintstart.db"
    return SourceStateStore(path=path)


OrchestratorFactory = Callable[[RetrievalFilters | None], ChatOrchestrator]


def get_orchestrator_factory(
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    source_state: SourceStateStore = Depends(get_source_state_store),
) -> OrchestratorFactory:
    """Build chat orchestrators bound to a request's retrieval filters.

    The filters (notably the project scope) come from the request body, which
    a dependency cannot see, so the route gets a factory rather than a
    ready-made orchestrator.
    """
    exclusions = source_state.get_exclusions()

    def build(filters: RetrievalFilters | None) -> ChatOrchestrator:
        return ChatOrchestrator(llm, store, exclusions, filters)

    return build


def get_onboarding_orchestrator(
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
) -> OnboardingOrchestrator:
    return OnboardingOrchestrator(llm, store, bm25_cache=get_bm25_cache())
