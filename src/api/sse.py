"""Server-Sent Events framing shared by the streaming orchestrators."""

import json
import logging
from collections.abc import Iterable, Iterator

from llm.errors import LLMUnavailableError

logger = logging.getLogger(__name__)


def sse_event(payload: dict[str, object]) -> str:
    """Frame a JSON payload as a single SSE ``data:`` event."""
    return f"data: {json.dumps(payload)}\n\n"


def stream_progress(
    events: Iterable[dict[str, object]], *, operation: str
) -> Iterator[str]:
    """Frame a stream of ``AiProgressEvent`` payloads as SSE.

    An LLM outage or any unexpected failure mid-stream becomes a terminal ``error``
    event rather than a truncated body — a client that has already rendered a few
    stages sees the run fail cleanly. Shared by every streaming onboarding operation
    (Seam 1 of the live-AI-visibility initiative) so the failure handling is written
    once.
    """
    try:
        for event in events:
            yield sse_event(event)
    except LLMUnavailableError as exc:
        yield sse_event({"type": "error", "operation": operation, "message": str(exc)})
    except Exception:
        logger.exception("Unexpected error in %s progress stream", operation)
        yield sse_event(
            {
                "type": "error",
                "operation": operation,
                "message": "An unexpected error occurred",
            }
        )
