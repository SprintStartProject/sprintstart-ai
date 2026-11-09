"""Builders shared by the diagram test modules."""

SUBJECT = "how a request reaches the database"
PIDS = frozenset({"p1"})


def node(
    node_id: str,
    label: str,
    chunk_ids: list[str] | None = None,
    kind: str = "COMPONENT",
    summary: str = "",
) -> dict[str, object]:
    built: dict[str, object] = {
        "id": node_id,
        "label": label,
        "kind": kind,
        "summary": summary,
    }
    if chunk_ids is not None:
        built["chunk_ids"] = chunk_ids
    return built


def edge(
    from_id: str, to_id: str, kind: str = "FLOWS_TO", label: str = ""
) -> dict[str, object]:
    return {"from_id": from_id, "to_id": to_id, "kind": kind, "label": label}


def payload(
    nodes: list[dict[str, object]],
    edges: list[dict[str, object]],
    **extra: object,
) -> dict[str, object]:
    built: dict[str, object] = {
        "summary": "A request lands on the controller and ends at the repository.",
        "nodes": nodes,
        "edges": edges,
    }
    built.update(extra)
    return built


TWO_CHUNKS = (
    "ReportController receives the HTTP request and validates it",
    "ReportRepository issues the SQL query against postgres",
)
