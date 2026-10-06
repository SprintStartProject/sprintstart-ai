"""Shared HTTP client for the SprintStart AI terminal tools.

All the subcommands (ingest, onboard, corpus) talk to the same running
service, so the connection handling, health check, SSE decoding, error
reporting, and document ingestion live here once. The offline chunk inspector
(`chunk_inspector_cli.py`) intentionally does not use this — it parses local
files and never touches the service.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from collections.abc import Iterator
from pathlib import Path

import httpx
from cli_colors import C
from rich.console import Console

DEFAULT_BASE_URL = os.environ.get("SPRINTSTART_URL", "http://localhost:8000")

TEXT_EXTENSIONS = {".txt", ".json", ".md", ".yaml", ".yml", ".toml"}
CODE_EXTENSIONS = {
    ".py",
    ".js",
    ".ts",
    ".tsx",
    ".go",
    ".java",
    ".kt",
    ".rs",
    ".cpp",
    ".cc",
    ".cxx",
    ".c",
    ".cs",
    ".php",
    ".rb",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
INGESTABLE = TEXT_EXTENSIONS | CODE_EXTENSIONS | IMAGE_EXTENSIONS

# Directory names (any path component) that are never ingested — caches, VCS
# metadata, and virtual environments contain no meaningful project content.
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}

# Directory names that indicate test/fixture material.  Files under these paths
# are still ingested (useful for Q&A) but tagged source_role="test" so that
# consumers like the onboarding generator exclude them from grounding evidence.
TEST_PATH_PARTS = {
    "tests",
    "test",
    "spec",
    "specs",
    "demo-corpus",
    "fixtures",
    "testdata",
    "__snapshots__",
}


def _source_role_from_path(path: Path, root: Path) -> str | None:
    """Return ``"test"`` if any component of *path* relative to *root* looks
    like test/fixture material, otherwise ``None`` (let the server classify)."""
    parts = set(path.relative_to(root).parts[:-1])  # directory parts only
    if parts & TEST_PATH_PARTS:
        return "test"
    return None


def add_base_url_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Base URL of the service (default: %(default)s).",
    )


def iter_sse(response: httpx.Response) -> Iterator[dict[str, object]]:
    """Yield decoded JSON payloads from an SSE `data:` stream."""
    for line in response.iter_lines():
        if line.startswith("data: "):
            try:
                yield json.loads(line[len("data: ") :])
            except json.JSONDecodeError:
                continue


class ServiceClient:
    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")
        self._http = httpx.Client(timeout=httpx.Timeout(None, connect=5.0))
        self.console = Console()

    @property
    def base_url(self) -> str:
        return self._base

    # --- HTTP helpers -----------------------------------------------------

    def get(self, path: str, **params: str | int | float | bool) -> httpx.Response:
        return self._http.get(f"{self._base}{path}", params=params or None)

    def post(self, path: str, json_body: dict[str, object]) -> httpx.Response:
        return self._http.post(f"{self._base}{path}", json=json_body)

    def delete(self, path: str) -> httpx.Response:
        return self._http.delete(f"{self._base}{path}")

    def events(
        self, path: str, payload: dict[str, object]
    ) -> Iterator[dict[str, object]]:
        """POST to an SSE endpoint and yield decoded events.

        Connection and HTTP errors are reported here and end the stream; the
        caller simply iterates and handles any `error` event it receives.
        """
        try:
            with self._http.stream(
                "POST", f"{self._base}{path}", json=payload
            ) as response:
                if response.status_code != 200:
                    response.read()
                    self.print_http_error(response)
                    return
                yield from iter_sse(response)
        except httpx.ConnectError:
            self.report_unreachable()
        except httpx.HTTPError as exc:
            print(C.red(f"[error] request failed: {exc}"))

    # --- health & errors --------------------------------------------------

    def health(self) -> tuple[str, str | None] | None:
        try:
            response = self._http.get(f"{self._base}/api/v1/health", timeout=20.0)
        except httpx.HTTPError:
            return None
        try:
            data = response.json()
        except (json.JSONDecodeError, ValueError):
            return ("unknown", None)
        detail = data.get("detail")
        return str(data.get("status", "unknown")), str(detail) if detail else None

    def print_banner(self, title: str) -> None:
        print(C.bold(title))
        print(C.dim(f"connected to {self._base}"))
        result = self.health()
        if result is None:
            self.report_unreachable()
            return
        status_text, detail = result
        colour = C.green if status_text == "ok" else C.yellow
        line = C.dim("health: ") + colour(status_text)
        if detail:
            line += C.dim(f" — {detail}")
        print(line)

    def report_unreachable(self) -> None:
        print(C.red(f"[error] cannot reach the service at {self._base}"))
        print(C.dim("        is it running?  uv run python -m src.main"))

    def print_http_error(self, response: httpx.Response) -> None:
        print(C.red(f"[error] {response.status_code} {self.detail(response)}"))

    @staticmethod
    def detail(response: httpx.Response) -> str:
        try:
            return str(response.json().get("detail", response.text))
        except (json.JSONDecodeError, ValueError):
            return response.text

    # --- ingestion (used by the ingest subcommand) -----------------------

    def ingest_path(self, raw_path: str, artifact_id: str | None) -> None:
        path = Path(raw_path).expanduser()
        if not path.exists():
            print(C.red(f"[error] no such path: {path}"))
            return

        if path.is_dir():
            files = sorted(
                p
                for p in path.rglob("*")
                if p.is_file()
                and p.suffix in INGESTABLE
                and not SKIP_DIRS.intersection(p.relative_to(path).parts)
            )
            if not files:
                print(C.yellow(f"no ingestable files found under {path}"))
                return
            print(C.dim(f"ingesting {len(files)} file(s) from {path}…"))
            for file in files:
                self._ingest_file(file, artifact_id=None, root=path)
        else:
            self._ingest_file(path, artifact_id=artifact_id, root=path.parent)

    def _ingest_file(
        self, path: Path, artifact_id: str | None, root: Path | None = None
    ) -> None:
        suffix = path.suffix.lower()
        if suffix not in INGESTABLE:
            hint = ""
            if suffix == ".pdf":
                hint = " (PDFs aren't supported over this JSON API)"
            print(C.yellow(f"  skip {path.name}: unsupported type {suffix!r}{hint}"))
            return

        try:
            raw = path.read_bytes()
        except OSError as exc:
            print(C.red(f"  fail {path.name}: {exc}"))
            return

        if suffix in IMAGE_EXTENSIONS:
            content = base64.b64encode(raw).decode("ascii")
        else:
            content = raw.decode("utf-8", errors="replace")

        body: dict[str, object] = {
            # Relative path (not bare name) so same-named files in different
            # folders get distinct ids — a shared id would overwrite the corpus.
            "artifact_id": artifact_id
            or (str(path.relative_to(root)).replace("\\", "/") if root else path.name),
            "filename": path.name,
            "content": content,
        }
        if root is not None:
            role = _source_role_from_path(path, root)
            if role is not None:
                body["source_role"] = role

        try:
            response = self.post("/api/v1/ingest", body)
        except httpx.ConnectError:
            self.report_unreachable()
            return
        except httpx.HTTPError as exc:
            print(C.red(f"  fail {path.name}: {exc}"))
            return

        if response.status_code == 200:
            data = response.json()
            count = data.get("chunk_count", "?")
            note = C.yellow(" (0 chunks — nothing stored)") if count == 0 else ""
            print(
                C.green(f"  ok   {path.name}")
                + C.dim(f" → artifact '{body['artifact_id']}', {count} chunk(s)")
                + note
            )
        else:
            print(
                C.red(
                    f"  fail {path.name}: "
                    f"{response.status_code} {self.detail(response)}"
                )
            )

    def close(self) -> None:
        self._http.close()
