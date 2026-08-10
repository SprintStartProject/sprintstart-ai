import logging
import os
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
from store.base import VectorStore
from store.chroma_store import ChromaVectorStore

logger = logging.getLogger(__name__)


def _build_client(backend: str) -> LLMClient:
    backend = backend.lower()

    if backend in {"ollama", "local"}:
        num_ctx = os.getenv("OLLAMA_NUM_CTX", "").strip()
        return OllamaClient(
            host=os.getenv("OLLAMA_BASE_URL"),
            model=os.getenv("OLLAMA_MODEL"),
            embed_model=os.getenv("OLLAMA_EMBED_MODEL"),
            vision_model=os.getenv("OLLAMA_VISION_MODEL"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.1")),
            num_ctx=int(num_ctx) if num_ctx else None,
        )

    if backend in {"openai", "openai-compatible", "litellm"}:
        return OpenAIClient(
            base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            api_key=os.getenv("OPENAI_API_KEY") or "unused",
            chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            embed_model=os.getenv("OPENAI_EMBED_MODEL"),
            vision_model=os.getenv("OPENAI_VISION_MODEL"),
        )

    if backend in {"anthropic", "claude"}:
        return AnthropicClient(
            api_key=os.getenv("ANTHROPIC_API_KEY") or "",
            chat_model=os.getenv("ANTHROPIC_CHAT_MODEL", "claude-haiku-4-5"),
            vision_model=os.getenv("ANTHROPIC_VISION_MODEL"),
            base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
            max_tokens=int(os.getenv("ANTHROPIC_MAX_TOKENS", "4096")),
        )

    raise ValueError(f"Unknown LLM backend: {backend!r}")


@lru_cache
def get_llm() -> LLMClient:
    chat = _build_client(
        os.getenv("LLM_BACKEND") or os.getenv("LLM_PROVIDER") or "local"
    )
    embed_backend = os.getenv("EMBED_BACKEND") or os.getenv("EMBED_PROVIDER")
    if embed_backend is None:
        return chat

    return SplitLLMClient(chat=chat, embed=_build_client(embed_backend))


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


def get_orchestrator(
    llm: LLMClient = Depends(get_llm),
    store: VectorStore = Depends(get_store),
    source_state: SourceStateStore = Depends(get_source_state_store),
) -> ChatOrchestrator:
    return ChatOrchestrator(llm, store, source_state.get_exclusions())
