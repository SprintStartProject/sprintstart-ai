"""Scope-selection of blueprints.

Blueprints are owned and persisted by the backend. The AI service receives them
as request input and is stateless. This module provides the pure selection logic
that picks the right blueprint(s) for a given profile and project.
"""

import logging

from onboarding.models import Blueprint, PersonProfile
from onboarding.scope import Scope

logger = logging.getLogger(__name__)


def select_blueprints(
    blueprints: list[Blueprint],
    profile: PersonProfile,
    project_id: str,
) -> list[Blueprint]:
    """Keep the project's ``global`` blueprints and those matching its area.

    Selection is project-scoped and fail-closed: a blueprint whose scope belongs
    to another project — or to no project at all, i.e. generated before project
    separation — is never selected, because its steps were drafted from a corpus
    the requesting project may not be allowed to see. Re-generate blueprints per
    project (``POST /onboarding/blueprints/generate``) to replace unqualified
    ones.

    An unknown working area yields a global-only path (no matching area scope).
    Global blueprints are ordered first.
    """
    area = profile.working_area.strip().lower()

    in_project = [b for b in blueprints if Scope.parse(b.scope).project == project_id]
    skipped = len(blueprints) - len(in_project)
    if skipped:
        logger.info(
            "Skipped %d blueprint(s) not scoped to project %s", skipped, project_id
        )

    selected = [b for b in in_project if Scope.parse(b.scope).is_global]
    selected.extend(b for b in in_project if Scope.parse(b.scope).area == area)
    return selected
