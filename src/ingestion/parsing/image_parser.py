from ingestion.models import ParsedChunk


def parse_image(filename: str, content: bytes) -> list[ParsedChunk]:
    return [
        ParsedChunk(
            content=content.decode("utf-8"),
            kind="image",
            metadata={"filename": filename},
        )
    ]
