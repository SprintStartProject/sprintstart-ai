"""SSE orchestrator for onboarding-path generation.

Mirrors :class:`agents.orchestrator.ChatOrchestrator`: a generator that streams
Server-Sent Events. Path generation is multi-step, so each pipeline stage is
emitted as a ``stage`` event, followed by a single ``path`` event (structured
path + YAML + quality report) and a ``done`` event. Errors collapse to one
``error`` event, consistent with the chat endpoint.
"""

import logging
from collections.abc import Generator, Iterator

from api.schemas import BlueprintSchema
from api.sse import sse_event
from llm.base import LLMClient
from llm.errors import LLMUnavailableError
from onboarding.models import Blueprint, OnboardingPath, PersonProfile
from onboarding.pipeline import OnboardingPipeline, StageProgress
from rag.hybrid import BM25IndexCache
from store.base import VectorStore

logger = logging.getLogger(__name__)


def _drain(gen: Generator[StageProgress, None, OnboardingPath]) -> OnboardingPath:
    """Exhaust a stage generator and return its final value."""
    while True:
        try:
            next(gen)
        except StopIteration as stop:
            return stop.value


def _emit_stages(
    gen: Generator[StageProgress, None, OnboardingPath],
) -> Generator[str, None, OnboardingPath]:
    while True:
        try:
            stage = next(gen)
        except StopIteration as stop:
            return stop.value
        event: dict[str, object] = {"type": "stage", "name": stage.name}
        if stage.detail:
            event["detail"] = stage.detail
        yield sse_event(event)


def _to_blueprint_models(
    blueprints: list[BlueprintSchema],
) -> list[Blueprint]:
    return [b.to_model() for b in blueprints]


class OnboardingOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        bm25_cache: BM25IndexCache | None = None,
    ) -> None:
        self._pipeline = OnboardingPipeline(llm, store, bm25_cache=bm25_cache)

    def run(
        self,
        profile: PersonProfile,
        blueprints: list[BlueprintSchema],
        project_id: str,
    ) -> OnboardingPath:
        """Run the pipeline synchronously and return the finished path."""
        bp_models = _to_blueprint_models(blueprints)
        return _drain(
            self._pipeline.run(profile, blueprints=bp_models, project_id=project_id)
        )

    def stream(
        self,
        profile: PersonProfile,
        blueprints: list[BlueprintSchema],
        project_id: str,
    ) -> Iterator[str]:
        try:
            bp_models = _to_blueprint_models(blueprints)
            path = yield from _emit_stages(
                self._pipeline.run(profile, blueprints=bp_models, project_id=project_id)
            )

            yield sse_event(
                {
                    "type": "path",
                    "path": path.model_dump(),
                    "quality": path.quality.model_dump(),
                }
            )
            yield sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in onboarding stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )
