"""Unified terminal client for SprintStart AI.

A single entrypoint over the running service with subcommands that share one
HTTP client (`_client.ServiceClient`):

    sprintstart ingest <path> [id]         ingest a file or directory
    sprintstart onboard -a backend -e junior   generate an onboarding path
    sprintstart corpus                     show what's ingested (status + artifacts)
    sprintstart corpus list [--artifact ID]
    sprintstart corpus search "<query>"

The offline chunk inspector lives separately in `chunk_inspector_cli.py`, since
it parses local files and never talks to the service.

Blueprint management (generate, versions, rollback) is owned by the backend.

Run with:  uv run python scripts/sprintstart.py <subcommand> ...
"""

from __future__ import annotations

import argparse

import corpus_cli
import ingest_cli
import onboarding_cli
from _client import add_base_url_arg

_COMMANDS = {
    "ingest": ingest_cli,
    "onboard": onboarding_cli,
    "corpus": corpus_cli,
}


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="sprintstart",
        description="Terminal client for SprintStart AI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name, module in _COMMANDS.items():
        sub = subparsers.add_parser(name, help=module.__doc__)
        # --base-url goes after the subcommand: `sprintstart ingest --base-url ...`.
        add_base_url_arg(sub)
        module.add_arguments(sub)
        sub.set_defaults(func=module.run)

    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
