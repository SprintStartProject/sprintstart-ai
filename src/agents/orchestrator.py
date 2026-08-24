import logging
from collections.abc import Iterator

from agents.chat_agent import ChatAgent, Evidence, Token
from agents.tools.base import Invocation
from api.schemas import HistoryEntry
from api.sse import sse_event
from llm.base import LLMClient, Message
from llm.errors import LLMUnavailableError
from rag.citation import build_citations
from rag.source_filter import SourceExclusions
from rag.types import RetrievalFilters, ScoredChunk
from store.base import VectorStore

logger = logging.getLogger(__name__)


class ChatOrchestrator:
    """Turns one `ChatAgent` run into the SSE stream `/api/v1/chat` serves."""

    def __init__(
        self,
        llm: LLMClient,
        store: VectorStore,
        exclusions: SourceExclusions = SourceExclusions(),
        filters: RetrievalFilters | None = None,
    ) -> None:
        self._agent = ChatAgent(llm, store, exclusions, filters)

    def stream(self, query: str, history: list[HistoryEntry]) -> Iterator[str]:
        messages: list[Message] = [
            Message(role=h.role, content=h.content) for h in history
        ]

        # A chunk resurfacing from a second tool call must not be cited twice.
        seen_chunk_ids: set[str] = set()

        try:
            for event in self._agent.run(query, messages):
                match event:
                    case Invocation(kind=kind, name=name):
                        yield sse_event(
                            {"type": "tool_use", "name": name, "kind": kind}
                        )
                    case Evidence(chunks=chunks):
                        # Emitted as each tool call resolves, so a consumer sees
                        # a source before the tokens it grounds.
                        yield from _citation_events(chunks, seen_chunk_ids)
                    case Token(text=text):
                        yield sse_event({"type": "token", "content": text})

            yield sse_event({"type": "done"})

        except LLMUnavailableError as exc:
            yield sse_event({"type": "error", "message": str(exc)})
        except Exception:
            logger.exception("Unexpected error in chat stream")
            yield sse_event(
                {"type": "error", "message": "An unexpected error occurred"}
            )


def _citation_events(
    chunks: list[ScoredChunk], seen_chunk_ids: set[str]
) -> Iterator[str]:
    fresh = [c for c in chunks if c.id not in seen_chunk_ids]
    for chunk in fresh:
        seen_chunk_ids.add(chunk.id)
    for citation in build_citations(fresh):
        yield sse_event(
            {
                "type": "citation",
                "artifact_id": citation.artifact_id,
                "start_line": citation.start_line,
                "start_page": citation.start_page,
            }
        )
