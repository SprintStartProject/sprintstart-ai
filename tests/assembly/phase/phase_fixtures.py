"""Builders shared by the phase test modules."""

PROJECT = "p1"


def step(title: str, chunk_ids: list[str] | None = None) -> dict[str, object]:
    built: dict[str, object] = {
        "title": title,
        "description": f"{title} description",
        "tasks": [],
        "resources": [],
        "estimated_minutes": 10,
        "expected_outcome": f"{title} outcome",
    }
    if chunk_ids is not None:
        built["chunk_ids"] = chunk_ids
    return built


def question(q_type: str = "SHORT_TEXT", **fields: object) -> dict[str, object]:
    built: dict[str, object] = {
        "type": q_type,
        "question": f"Explain {q_type}",
    }
    if q_type == "SHORT_TEXT":
        built["correct_answer"] = "Because the evidence says so"
    else:
        built["options"] = [
            {"label": "Right", "correct": True},
            {"label": "Wrong", "correct": False},
        ]
    built.update(fields)
    return built


def payload(
    steps: list[dict[str, object]] | None = None,
    questions: list[dict[str, object]] | None = None,
    **extra: object,
) -> dict[str, object]:
    built: dict[str, object] = {
        "steps": steps or [],
        "questions": questions or [],
    }
    built.update(extra)
    return built
