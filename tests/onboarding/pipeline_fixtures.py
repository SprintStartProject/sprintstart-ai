"""Stubs and probes shared by the tests of the staged assembly pipelines."""

import json
from collections.abc import Generator

from llm.base import Message
from onboarding.progress import ProgressEvent
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

EMBED = [1.0] + [0.0] * 767


def collect[T](
    generator: Generator[ProgressEvent, None, T],
) -> tuple[list[ProgressEvent], T]:
    """Drain a progress generator, keeping both the events and the returned value."""
    events: list[ProgressEvent] = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        return events, stop.value


def script(events: list[ProgressEvent]) -> list[tuple[object, object, object]]:
    """Each event as ``(type, stage, label)``: what a watcher of the stream reads."""
    return [(e["type"], e.get("stage"), e["label"]) for e in events]


class RecordingLLM(StubLLMClient):
    """A stub that answers every generation alike and keeps what it was sent."""

    def __init__(self, generate_response: str) -> None:
        super().__init__(generate_response=generate_response, embedding=EMBED)
        self.calls: list[list[Message]] = []
        self.temperatures: list[float | None] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.calls.append(messages)
        self.temperatures.append(temperature)
        return super().generate(messages, temperature=temperature)


def make_llm(payload: dict[str, object]) -> RecordingLLM:
    return RecordingLLM(json.dumps(payload))


def unreadable_llm() -> RecordingLLM:
    return RecordingLLM("not json at all")


def make_store(*texts: str, project: str = "p1") -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id=f"a{i}",
                filename=f"doc{i}.md",
                text=text,
                embedding=EMBED,
                project_ids=(project,),
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store
