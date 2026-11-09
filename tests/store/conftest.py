from collections.abc import Iterator

import chromadb
import pytest


@pytest.fixture(autouse=True)
def empty_ephemeral_chroma() -> Iterator[None]:
    """Start every test with no collections in Chroma's process-wide in-memory store.

    Every ``chromadb.EphemeralClient()`` in a process shares one store, so a test
    that runs after another one in the same process sees its chunks.
    """
    client = chromadb.EphemeralClient()
    for collection in client.list_collections():
        client.delete_collection(collection.name)
    yield
