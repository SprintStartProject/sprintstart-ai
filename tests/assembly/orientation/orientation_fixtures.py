"""Builders shared by the orientation test modules."""

TITLE = "Fix the stale cache header on /api/v1/reports"
PIDS = frozenset({"p1"})


def section(
    step: str, title: str, chunk_ids: list[str] | None = None, body: str | None = None
) -> dict[str, object]:
    built: dict[str, object] = {
        "step": step,
        "title": title,
        "body": body if body is not None else f"{title} body",
    }
    if chunk_ids is not None:
        built["chunk_ids"] = chunk_ids
    return built


def payload(*sections: dict[str, object], **extra: object) -> dict[str, object]:
    built: dict[str, object] = {
        "summary": "What you need to change the reports cache header.",
        "sections": list(sections),
    }
    built.update(extra)
    return built
