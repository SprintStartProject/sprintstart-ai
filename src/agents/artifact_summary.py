from collections.abc import Generator
from dataclasses import dataclass

from llm.base import LLMClient, Message, TextDelta
from rag.types import Chunk

_MAX_BATCH_CHARS = 10_000
_MAX_SUMMARY_BATCHES = 8


@dataclass(frozen=True)
class SummaryCitation:
    artifact_id: str
    filename: str
    source_url: str


@dataclass(frozen=True)
class ArtifactSummary:
    artifact_id: str
    summary: str
    citations: list[SummaryCitation]


@dataclass(frozen=True)
class SummaryStage:
    """A progress marker yielded by :meth:`ArtifactSummaryAgent.summarize_stream`
    while it does non-streamed work (batch note-taking) ahead of the streamed
    final summary.
    """

    name: str
    detail: str = ""


class ArtifactSummaryAgent:
    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm

    def summarize_stream(
        self,
        artifact_id: str,
        chunks: list[Chunk],
        previous_chunks: list[Chunk] | None = None,
    ) -> Generator[SummaryStage | str, None, ArtifactSummary]:
        """Yields progress while gathering notes, then streams the final summary
        token by token, and returns the completed :class:`ArtifactSummary` once
        exhausted.

        Note-gathering (:meth:`_notes_for_chunks`) is not itself streamed -- it is
        internal working notes, not the user-facing text -- only the final
        summary generated from those notes is.
        """
        if not chunks:
            raise ValueError("Cannot summarize artifact without chunks.")

        yield SummaryStage("notes", "Extracting notes from the source")
        current_notes = self._notes_for_chunks("current artifact", chunks)
        previous_notes = (
            self._notes_for_chunks("previous artifact", previous_chunks)
            if previous_chunks
            else None
        )

        yield SummaryStage("summary", "Generating summary")
        messages = self._final_summary_messages(
            artifact_id=artifact_id,
            current_notes=current_notes,
            previous_notes=previous_notes,
        )
        summary_parts: list[str] = []
        for event in self._llm.stream(messages):
            # Reasoning is live-only chat metadata. Summaries persist only the
            # final user-visible text.
            if isinstance(event, TextDelta):
                summary_parts.append(event.text)
                yield event.text

        citations = _citations_for_chunks(artifact_id, chunks)

        if previous_chunks:
            citations.extend(
                _citations_for_chunks(previous_chunks[0].artifact_id, previous_chunks)
            )

        return ArtifactSummary(
            artifact_id=artifact_id,
            summary="".join(summary_parts).strip(),
            citations=citations,
        )

    def _notes_for_chunks(self, label: str, chunks: list[Chunk]) -> str:
        all_batches = _chunk_batches(chunks, _MAX_BATCH_CHARS)
        batches = all_batches[:_MAX_SUMMARY_BATCHES]

        truncated_notice = ""
        if len(all_batches) > len(batches):
            truncated_notice = (
                "\n\n[Note: Additional source batches were omitted because the "
                "artifact exceeds the summary batch limit.]"
            )

        if len(batches) == 1:
            return batches[0] + truncated_notice

        partials: list[str] = []

        for index, batch in enumerate(batches, start=1):
            messages = _grounded_summary_messages(
                user_content=(
                    f"Summarize batch {index} of the {label}.\n\n"
                    "Extract only grounded facts:\n"
                    "- key points\n"
                    "- decisions\n"
                    "- changes or version-related notes\n\n"
                    f"Source excerpts:\n{batch}"
                )
            )
            partials.append(self._llm.generate(messages).strip())

        return "\n\n".join(partials) + truncated_notice

    def _final_summary_messages(
        self,
        artifact_id: str,
        current_notes: str,
        previous_notes: str | None,
    ) -> list[Message]:
        previous_section = (
            f"\n\nPrevious version excerpts / notes:\n{previous_notes}\n"
            if previous_notes
            else (
                "\n\nNo previous artifact was provided. If the current source "
                "contains version history, summarize what changed from that. "
                "Otherwise say that no version history was found."
            )
        )

        return _grounded_summary_messages(
            user_content=(
                f"Create a short summary for artifact {artifact_id}.\n\n"
                "Return concise markdown with exactly these sections:\n"
                "## Key points\n"
                "## Decisions\n"
                "## What changed\n\n"
                "Rules:\n"
                "- Keep it short.\n"
                "- Prefer bullets.\n"
                "- Mention uncertainty if the source does not contain enough info.\n"
                "- If no decisions are present, say so.\n"
                "- If no version history is present, say so under What changed.\n\n"
                f"Current artifact excerpts / notes:\n{current_notes}"
                f"{previous_section}"
            )
        )


def _grounded_summary_messages(user_content: str) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "You are SprintStart's grounded summarization assistant. "
                "Use only the provided source excerpts. "
                "Do not use external knowledge. "
                "Do not invent facts, decisions, changes, or links. "
                "If the source excerpts do not contain enough information, say so."
            ),
        ),
        Message(role="user", content=user_content),
    ]


def _chunk_batches(chunks: list[Chunk], max_chars: int) -> list[str]:
    ordered = sorted(
        chunks,
        key=lambda chunk: chunk.position if chunk.position is not None else 0,
    )
    batches: list[str] = []
    current = ""

    for chunk in ordered:
        block = _format_chunk(chunk)

        if len(current) + len(block) > max_chars and current:
            batches.append(current)
            current = ""

        if len(block) > max_chars:
            block = block[:max_chars] + "\n[truncated]\n"

        current += block

    if current:
        batches.append(current)

    return batches


def _format_chunk(chunk: Chunk) -> str:
    position = chunk.position if chunk.position is not None else 0
    return (
        f"\n--- Source: {chunk.filename} | chunk {position} ---\n{chunk.text.strip()}\n"
    )


def _citations_for_chunks(
    artifact_id: str,
    chunks: list[Chunk],
) -> list[SummaryCitation]:
    fallback_url = f"/api/v1/vector-db/artifacts/{artifact_id}/chunks"
    source_url_by_filename: dict[str, str] = {}

    for chunk in chunks:
        current_url = source_url_by_filename.get(chunk.filename)
        chunk_url = chunk.source_url or fallback_url

        if current_url is None:
            source_url_by_filename[chunk.filename] = chunk_url
            continue

        if current_url == fallback_url and chunk.source_url is not None:
            source_url_by_filename[chunk.filename] = chunk.source_url

    return [
        SummaryCitation(
            artifact_id=artifact_id,
            filename=filename,
            source_url=source_url,
        )
        for filename, source_url in sorted(source_url_by_filename.items())
    ]
