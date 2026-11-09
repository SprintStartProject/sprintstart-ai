from dataclasses import dataclass
from typing import Literal

ChunkKind = Literal["text", "code", "pdf", "image"]


@dataclass(slots=True)
class ParsedChunk:
    """
    Represents a processed text chunk produced by a file parser.

    Attributes:
        content (str): The raw or extracted text content of the chunk.
        kind (ChunkKind): The type/category of the chunk ("text", "code", "pdf"
        or "image").
        metadata (dict[str, str]): Additional information about the chunk,
            such as source file name, position/index, or encoding details.
    """

    content: str
    kind: ChunkKind
    metadata: dict[str, str]
