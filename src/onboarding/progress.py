"""Shared AI progress events — Seam 1 of the live-AI-visibility initiative.

One event family every streaming onboarding operation emits, so the backend
passthrough and the frontend consumer are each written once. The contract and
its invariants live on the initiative tracker (sprintstart-backend#93); the
load-bearing ones, restated where they are enforced:

* **An ``item`` event is a promise of validation.** It is emitted only after the
  element cleared the *same gate* the batch path applies — grounding, here. Never
  a token, never partial JSON. If a client saw it, it will be in the final result.
* **The ``done`` event carries the whole result.** ``stage``/``warning`` events are
  advisory (losing one costs no correctness); only ``done``'s ``result`` — and what
  the backend persists from it — is authoritative. A streaming run therefore
  produces exactly what the non-streaming call would.

The events are plain dicts because they are framed straight to SSE by
:func:`api.sse.sse_event`; this module only guarantees the shape and a monotonic
``seq``.
"""

from collections.abc import Generator
from typing import cast

# One SSE payload. ``object`` (not ``Any``) so a stray un-JSON-able value is a type
# error here rather than a 500 at serialisation time.
ProgressEvent = dict[str, object]


class ProgressStream:
    """Builds ``AiProgressEvent`` payloads for one operation, numbering them in order.

    Owns nothing but the operation name and a sequence counter — a client dedupes
    on ``(operation, seq)`` across a reconnect, so the counter must be monotonic and
    per-stream.
    """

    def __init__(self, operation: str) -> None:
        self.operation = operation
        self._seq = 0

    def _emit(self, type_: str, label: str, **fields: object) -> ProgressEvent:
        event: ProgressEvent = {
            "type": type_,
            "operation": self.operation,
            "seq": self._seq,
            "label": label,
        }
        # None-valued optional fields are omitted rather than sent as null.
        event.update({key: value for key, value in fields.items() if value is not None})
        self._seq += 1
        return event

    def stage(self, stage: str, label: str) -> ProgressEvent:
        """A coarse pipeline step is starting — advisory, for the human watching."""
        return self._emit("stage", label, stage=stage)

    def item(self, item: dict[str, object], label: str) -> ProgressEvent:
        """A single element that has cleared its validation gate (see the invariant)."""
        return self._emit("item", label, item=item)

    def warning(self, label: str) -> ProgressEvent:
        """Something was dropped or degraded, but the run continues."""
        return self._emit("warning", label, message=label)

    def done(self, label: str, result: dict[str, object]) -> ProgressEvent:
        """Terminal success, carrying the whole authoritative result."""
        return self._emit("done", label, result=result)

    def error(self, label: str) -> ProgressEvent:
        """Terminal failure; no result follows."""
        return self._emit("error", label, message=label)


def drain[T](generator: Generator[ProgressEvent, None, T]) -> T:
    """Run a progress generator to completion, discarding events, returning its result.

    This is what lets a streaming pipeline back its non-streaming twin: the sync
    entry point drives the same generator and keeps only the final value, so the two
    cannot diverge — there is one code path, watched or not.
    """
    try:
        while True:
            next(generator)
    except StopIteration as stop:
        return cast(T, stop.value)
