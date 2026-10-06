"""
Manual test client for the SprintStart AI API.
Run the API first: uv run uvicorn api.app:app --reload --app-dir src
"""

import httpx

BASE_URL = "http://localhost:8000/api/v1"


def check_health() -> None:
    print("--- GET /health ---")
    response = httpx.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Body:   {response.json()}\n")


def ingest_document(artifact_id: str, filename: str, content: str) -> None:
    print(f"--- POST /ingest ({filename}) ---")
    response = httpx.post(
        f"{BASE_URL}/ingest",
        json={"artifact_id": artifact_id, "filename": filename, "content": content},
    )
    print(f"Status: {response.status_code}")
    print(f"Body:   {response.json()}\n")


if __name__ == "__main__":
    check_health()

    ingest_document(
        artifact_id="sprint-42-retro",
        filename="retro.md",
        content=(
            "# Sprint 42 Retro\n\n"
            "## What went well\n"
            "Good collaboration between frontend and backend teams.\n\n"
            "## Blockers\n"
            "Missing designs delayed the auth feature by 3 days. "
            "CI pipeline was flaky and caused multiple failed deploys.\n\n"
            "## Action items\n"
            "- Design handoff process to be agreed before sprint start.\n"
            "- Investigate flaky tests in CI."
        ),
    )
