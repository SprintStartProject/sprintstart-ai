# pyright: reportPrivateUsage=false
import pytest

from api.dependencies import (
    _DEFAULT_LLM_TIMEOUT_SECONDS,
    _build_client,
    _llm_timeout,
)
from llm.openai_client import OpenAIClient


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
