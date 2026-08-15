from llm.base import Message
from rag.types import ScoredChunk


def chunk_header(chunk: ScoredChunk) -> str:
    """Label a chunk for the model — also used for the chat agent's tool results."""
    parts = [chunk.artifact_type or "FILE", chunk.filename]
    if chunk.language:
        parts.append(chunk.language)
    if chunk.source_url:
        parts.append(chunk.source_url)
    return "[" + " | ".join(parts) + "]"


def build_messages(
    question: str,
    chunks: list[ScoredChunk],
    history: list[Message],
) -> list[Message]:
    messages: list[Message] = []

    if chunks:
        context = "\n\n".join(
            f"{chunk_header(chunk)}\n{chunk.text}" for chunk in chunks
        )
        messages.append(
            Message(
                role="system",
                content=f"Answer based on the following context:\n\n{context}",
            )
        )

    messages.extend(history)
    messages.append(Message(role="user", content=question))

    return messages
