"""Parsing of the onboarding scope identifier.

A scope names *what* a blueprint covers (the global scope ``"global"`` or a
named area ``"area:<name>"``) and, since project separation, *which project* it
was generated for. The project-qualified form prefixes the project segment:

    project:<project-id>|global
    project:<project-id>|area:backend

The string form lives on :class:`~onboarding.models.Blueprint`; this value
object centralizes how it is interpreted so the prefixes live in one place.
Unqualified scopes (``"global"``, ``"area:backend"``) still parse — they are
blueprints from before project separation and carry ``project is None``, which
callers treat as belonging to no project.
"""

from dataclasses import dataclass

GLOBAL = "global"
AREA_PREFIX = "area:"
PROJECT_PREFIX = "project:"
PROJECT_SEPARATOR = "|"


@dataclass(frozen=True)
class Scope:
    """A parsed scope identifier."""

    raw: str
    # The area name for an ``area:<name>`` scope; ``None`` for global or any
    # unrecognized form.
    area: str | None
    # The project the scope belongs to; ``None`` for an unqualified scope.
    project: str | None = None

    @property
    def is_global(self) -> bool:
        return self.raw.rsplit(PROJECT_SEPARATOR, 1)[-1] == GLOBAL

    @classmethod
    def parse(cls, raw: str) -> "Scope":
        project: str | None = None
        remainder = raw

        if raw.startswith(PROJECT_PREFIX) and PROJECT_SEPARATOR in raw:
            project_part, remainder = raw.split(PROJECT_SEPARATOR, 1)
            project = project_part[len(PROJECT_PREFIX) :] or None

        if remainder.startswith(AREA_PREFIX):
            return cls(raw=raw, area=remainder[len(AREA_PREFIX) :], project=project)

        return cls(raw=raw, area=None, project=project)

    @classmethod
    def build(cls, project_id: str, area: str | None = None) -> "Scope":
        """Build a project-qualified scope for ``area`` (global when ``None``)."""
        tail = GLOBAL if area is None else f"{AREA_PREFIX}{area}"
        raw = f"{PROJECT_PREFIX}{project_id}{PROJECT_SEPARATOR}{tail}"
        return cls(raw=raw, area=area, project=project_id)

    def with_project(self, project_id: str) -> "Scope":
        """Return this scope qualified with ``project_id``.

        Re-qualifying is idempotent for the same project and lets the backend
        keep sending plain ``global`` / ``area:<name>`` scope names.
        """
        if self.project == project_id:
            return self
        tail = (
            self.raw.split(PROJECT_SEPARATOR, 1)[1]
            if self.project is not None
            else self.raw
        )
        raw = f"{PROJECT_PREFIX}{project_id}{PROJECT_SEPARATOR}{tail}"
        return Scope(raw=raw, area=self.area, project=project_id)
