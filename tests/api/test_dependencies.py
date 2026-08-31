# pyright: reportPrivateUsage=false
import pytest

from api.dependencies import (
    _DEFAULT_LLM_TIMEOUT_SECONDS,
    _build_client,
    _llm_timeout,
    get_onboarding_orchestrator,
)
from llm.openai_client import OpenAIClient
from rag.retriever import get_bm25_cache
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def test_llm_timeout_defaults_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_TIMEOUT_SECONDS", raising=False)

    assert _llm_timeout() == _DEFAULT_LLM_TIMEOUT_SECONDS


def test_llm_timeout_reads_the_configured_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "30")

    assert _llm_timeout() == 30.0


def test_llm_timeout_of_zero_disables_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Opting out has to stay reachable — a long local batch may legitimately
    outrun any ceiling, which is the behaviour Ollama had before this existed.
    """
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0")

    assert _llm_timeout() is None


def test_openai_reasoning_settings_apply_only_to_chat_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "4096")
    monkeypatch.setenv("OPENAI_REASONING_MAX_TOKENS", "1024")

    chat = _build_client("openai")
    embed = _build_client("openai", is_embed=True)

    assert isinstance(chat, OpenAIClient)
    assert chat.max_tokens == 4096
    assert chat.reasoning_max_tokens == 1024
    assert isinstance(embed, OpenAIClient)
    assert embed.max_tokens is None
    assert embed.reasoning_max_tokens is None


def test_onboarding_orchestrator_shares_the_process_wide_bm25_cache() -> None:
    """Regression test for issue #129 #8: chat/agent retrieval and onboarding
    generation must tokenize the corpus once, not maintain independent caches.
    """
    orchestrator = get_onboarding_orchestrator(
        llm=StubLLMClient(), store=StubVectorStore()
    )

    assert orchestrator._pipeline._bm25_cache is get_bm25_cache()
