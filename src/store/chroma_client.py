"""One Chroma client per path for the whole process."""

import threading

import chromadb
import chromadb.api
from chromadb.config import Settings

CLIENT_CACHE: dict[str, chromadb.api.ClientAPI] = {}
_CLIENT_CACHE_LOCK = threading.Lock()


def get_persistent_client(path: str) -> chromadb.api.ClientAPI:
    client = CLIENT_CACHE.get(path)
    if client is not None:
        return client

    # FastAPI resolves synchronous dependencies in a thread pool, so the first
    # buddy turn and its asynchronous analytics fan-out can arrive here together.
    # Chroma's PersistentClient instances share process-wide, refcounted state;
    # constructing two for one path concurrently can make one failed init tear
    # down the other's system. Re-check inside the lock so exactly one thread
    # constructs and publishes the client.
    with _CLIENT_CACHE_LOCK:
        client = CLIENT_CACHE.get(path)
        if client is not None:
            return client

        settings = Settings(
            anonymized_telemetry=False,
            is_persistent=True,
            allow_reset=True,
        )
        client = chromadb.PersistentClient(path=path, settings=settings)
        CLIENT_CACHE[path] = client
        return client
