import asyncio
import logging

from fastapi import FastAPI

from api import app as app_module


class _WarmupLLM:
    def embed(self, text: str) -> list[float]:
        assert text == "ping"
        return [1.0]


def _run_lifespan() -> None:
    async def run() -> None:
        async with app_module.lifespan(FastAPI()):
            pass

    asyncio.run(run())


def test_lifespan_warms_llm_and_store(monkeypatch) -> None:
    store = object()
    store_calls = 0

    def get_store() -> object:
        nonlocal store_calls
        store_calls += 1
        return store

    monkeypatch.setattr(app_module, "get_llm", lambda: _WarmupLLM())
    monkeypatch.setattr(app_module, "get_store", get_store)

    _run_lifespan()

    assert store_calls == 1


def test_lifespan_logs_store_failure_and_continues(
    monkeypatch,
    caplog,
) -> None:
    def fail_store() -> object:
        raise RuntimeError("temporary Chroma failure")

    monkeypatch.setattr(app_module, "get_llm", lambda: _WarmupLLM())
    monkeypatch.setattr(app_module, "get_store", fail_store)

    with caplog.at_level(logging.WARNING, logger=app_module.__name__):
        _run_lifespan()

    assert "Chroma store unavailable at startup" in caplog.text
