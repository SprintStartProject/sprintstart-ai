import logging
from collections.abc import Generator, Iterator

from agents.base import AgentRunState
from agents.orchestrator_agent import OrchestratorAgent
from agents.tools.base import Invocation
from api.schemas import HistoryEntry
from api.sse import sse_event
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from rag.source_filter import SourceExclusions
from rag.types import Citation, RetrievalFilters, ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)


def _citation_event(citation: Citation) -> str:
    return sse_event(
        {
            "type": "citation",
            "artifact_id": citation.artifact_id,
            "start_line": citation.start_line,
            "start_page": citation.start_page,
        }
    )


def _new_citation_events(
    chunks: list[ScoredChunk], seen_chunk_ids: set[str]
) -> Iterator[str]:
    """Emit citation events for any chunks not yet seen, marking them seen.

    ``chunks`` (typically ``state.chunks``) only ever grows — chunks are
    deduplicated by id and never removed — so tracking ``seen_chunk_ids``
    across calls lets this be called repeatedly as the list fills in during
    streaming without ever re-emitting the same citation twice.
    """
    new_chunks = [c for c in chunks if c.id not in seen_chunk_ids]
    if not new_chunks:
        return
    for chunk in new_chunks:
        seen_chunk_ids.add(chunk.id)
    for citation in build_citations(new_chunks):
        yield _citation_event(citation)


def _emit_tool_use_and_citations(
    gen: Generator[Invocation, None, AgentRunState],
    state: AgentRunState,
    seen_chunk_ids: set[str],
) -> Generator[str, None, AgentRunState]:
    """Drive ``gen``, streaming a ``tool_use`` event per invocation plus a
    ``citation`` event as soon as each tool call's chunks land in ``state``.

    ``state`` must be the very same ``AgentRunState`` instance the agent
    mutates internally (passed into ``gather_stream``). A tool call's chunks
    are merged into ``state.chunks`` synchronously, right *before* the
    generator's *next* yield (the following invocation, or return) — so
    checking ``state`` for unseen chunks right after each ``next()`` call,
    but *before* forwarding that call's own ``tool_use`` event, surfaces a
    citation as soon as the tool call that produced it has resolved, instead
    of only once the whole gather phase has finished.
    """
    while True:
        try:
            usage = next(gen)
        except StopIteration as stop:
            yield from _new_citation_events(state.chunks, seen_chunk_ids)
            return stop.value
        yield from _new_citation_events(state.chunks, seen_chunk_ids)
        yield sse_event({"type": "tool_use", "name": usage.name, "kind": usage.kind})


class ChatOrchestrator:
    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        exclusions: SourceExclusions = SourceExclusions(),
        filters: RetrievalFilters | None = None,
    ) -> None:
        self._agent = OrchestratorAgent(llm, store, exclusions, filters)

    def stream(self, query: str, history: list[HistoryEntry]) -> Iterator[str]:
        messages: list[Message] = [
            Message(role=h.role, content=h.content) for h in history
        ]

        try:
            state = AgentRunState()
            seen_chunk_ids: set[str] = set()
            state = yield from _emit_tool_use_and_citations(
                self._agent.gather_stream(query, messages, state),
                state,
                seen_chunk_ids,
            )

            for token in self._agent.answer_stream(query, state, messages):
                if token:
                    yield sse_event({"type": "token", "content": token})

            # Safety net: state.chunks could in principle still grow after the
            # last tool_use event (e.g. a future _seed implementation that
            # appends after gather_stream returns). Cheap no-op once
            # everything has already been streamed above.
            yield from _new_citation_events(state.chunks, seen_chunk_ids)

            yield sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in chat stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )
