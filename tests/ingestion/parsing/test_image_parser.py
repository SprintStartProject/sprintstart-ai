from ingestion.parsing.image_parser import parse_image

# Minimal 1×1 red PNG (base64-encoded)
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
    "/5+hHgAHggJ/PchI6QAAAABJRU5ErkJggg=="
)
_TINY_PNG_CONTENT = _TINY_PNG_B64.encode("utf-8")


def test_returns_single_image_chunk() -> None:
    chunks = parse_image("photo.png", _TINY_PNG_CONTENT)

    assert len(chunks) == 1
    assert chunks[0].kind == "image"
    assert chunks[0].metadata["filename"] == "photo.png"


def test_content_is_base64_string() -> None:
    chunks = parse_image("photo.png", _TINY_PNG_CONTENT)

    assert chunks[0].content == _TINY_PNG_B64


def test_different_filenames_stored_in_metadata() -> None:
    chunks = parse_image("diagram.jpg", _TINY_PNG_CONTENT)

    assert chunks[0].metadata["filename"] == "diagram.jpg"
