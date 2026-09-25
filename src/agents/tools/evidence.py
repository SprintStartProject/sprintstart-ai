"""How a search tool's result is budgeted, ordered and shown to the model.

Shared by every agent loop that runs ``retrieve``/``grep`` (the chat agent and
the onboarding buddy), so the two cannot drift on how much evidence one call
may return or how it reads. It lives beside the tools rather than in ``rag``
because it formats a ``ToolResult``, and ``rag`` must not import ``agents``.
"""

from collections.abc import Callable

from agents.tools.base import ToolResult
from rag.prompt import chunk_header
from rag.types import ScoredChunk

# Per-chunk cap on what goes back to the model, so a handful of large chunks
# can't crowd out the conversation.
SOURCE_CHARS = 800

# Caps on a *whole* tool result. The per-chunk limit alone bounds nothing:
# `grep` returns every match in the scoped corpus, so a broad pattern can
# return hundreds of chunks that are individually small and collectively
# larger than the context window.
#
# Both are needed, and each has to bite where the other doesn't: the char
# budget bounds full-size chunks (it stops at 10 of them), the chunk count
# bounds a long tail of small ones. Keep `MAX_EVIDENCE_CHUNKS * SOURCE_CHARS`
# above `MAX_EVIDENCE_CHARS` or the char budget becomes unreachable.
#
# Applied per call rather than per turn, so what one search returns never
# depends on which others shared its turn.
MAX_EVIDENCE_CHUNKS = 12
MAX_EVIDENCE_CHARS = 8_000


def limit_evidence(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """The prefix of ``chunks`` that fits the budget, in the order given.

    Deterministic and order-preserving: the tools already return their best
    matches first, so taking a prefix keeps the most relevant sources. The
    first chunk is always kept, so an oversized one is truncated by
    ``SOURCE_CHARS`` rather than dropped entirely.
    """
    selected: list[ScoredChunk] = []
    used = 0
    for chunk in chunks:
        if len(selected) >= MAX_EVIDENCE_CHUNKS:
            break
        size = min(len(chunk.text), SOURCE_CHARS)
        if selected and used + size > MAX_EVIDENCE_CHARS:
            break
        selected.append(chunk)
        used += size
    return selected


def format_evidence(
    result: ToolResult,
    chunks: list[ScoredChunk],
    *,
    header: Callable[[ScoredChunk], str] = chunk_header,
    empty: str | None = None,
) -> str:
    """What the model sees for one tool call — the sources themselves.

    ``chunks`` is what survived the budget, which is what the caller cites; a
    dropped chunk is counted but never quoted, so the model is not asked to
    answer from text it cannot see.

    ``header`` labels each chunk (the buddy adds a test-file warning to the
    default). ``empty`` replaces the tool's own one-line summary when nothing
    survived, for a caller whose persona reacts to specific wording.
    """
    if not chunks:
        return empty or result.summary or "No matches."
    body = "\n\n---\n\n".join(
        f"{header(chunk)}\n{chunk.text[:SOURCE_CHARS]}"
        for chunk in order_evidence_for_display(chunks)
    )
    omitted = result.match_count - len(result.matched_chunks(chunks))
    if omitted:
        body += f"\n\n---\n\n({omitted} further match(es) omitted.)"
    return body


def order_evidence_for_display(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Keep artifact relevance order while restoring source order within each file.

    Budget selection receives direct hits before neighbours so context cannot
    displace a match. The model should nevertheless read each selected file in
    its natural order; grouping after selection gives us both properties.
    """
    by_artifact: dict[str, list[ScoredChunk]] = {}
    for chunk in chunks:
        by_artifact.setdefault(chunk.artifact_id, []).append(chunk)

    ordered: list[ScoredChunk] = []
    for artifact_chunks in by_artifact.values():
        if all(chunk.position is None for chunk in artifact_chunks):
            ordered.extend(artifact_chunks)
            continue
        ordered.extend(
            sorted(
                artifact_chunks,
                key=lambda chunk: (
                    chunk.start_page if chunk.start_page is not None else 0,
                    chunk.position is None,
                    chunk.position if chunk.position is not None else 0,
                ),
            )
        )
    return ordered
