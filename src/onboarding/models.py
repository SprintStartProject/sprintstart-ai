"""Domain models shared across the onboarding jobs.

What is left here is what survived the retirement of the per-user path
generator: a person's self-reported profile (which the interviewer still reads),
citations, and the baseline -- a scoped, versioned *competency selection*.

The prose-step models that used to live here described a path generated per
hire. Content is now a shared module owned by a competency, so there is nothing
per-person left to model.
"""

from pydantic import BaseModel, Field

# Coarse, ordinal experience levels used to gate steps by ``min_experience`` and
# to tune the synthesis verbosity. Single source of truth for both, so the two
# consumers can't disagree about what a level means. Synonyms map to the same
# rank. Unknown values rank as 0 (most inclusive) so unseen experience never
# crashes and required steps are re-injected by the coverage gate regardless.
EXPERIENCE_LEVELS: dict[str, int] = {
    "intern": 1,
    "entry": 1,
    "junior": 1,
    "mid": 2,
    "intermediate": 2,
    "senior": 3,
    "lead": 4,
    "staff": 4,
    "principal": 5,
}


def experience_rank(level: str | None) -> int:
    """Ordinal rank for a coarse experience level; unknown/None -> 0."""
    if level is None:
        return 0
    return EXPERIENCE_LEVELS.get(level.strip().lower(), 0)


# Ordinal skill-proficiency levels, aligned 1:1 with the backend ``SkillLevel``
# enum and the frontend ``SkillLevel`` union so all three services agree on what
# a level means. Matching is case-insensitive; unknown values rank as 0 (treated
# as no demonstrated proficiency) so an unseen level never crashes generation.
SKILL_LEVELS: dict[str, int] = {
    "beginner": 1,
    "intermediate": 2,
    "advanced": 3,
    "expert": 4,
}


def skill_rank(level: str | None) -> int:
    """Ordinal rank for a skill-proficiency level; unknown/None -> 0."""
    if level is None:
        return 0
    return SKILL_LEVELS.get(level.strip().lower(), 0)


def proficiency_rank(skills: "list[SkillAssessment]") -> int:
    """Overall proficiency for gating/verbosity, derived from a person's skills.

    Taken as the highest skill level held (``0`` when no skills are listed, which
    is the most inclusive — every step surfaces). The :data:`SKILL_LEVELS` scale
    lines up rank-for-rank with :data:`EXPERIENCE_LEVELS`
    (beginner↔junior, intermediate↔mid, advanced↔senior, expert↔lead), so this
    rank is directly comparable to a step's ``min_experience`` rank.
    """
    return max((skill_rank(s.level) for s in skills), default=0)


class SkillAssessment(BaseModel):
    """A single assessed skill with a proficiency level.

    ``name`` is a free-form skill tag (e.g. ``kotlin``); ``level`` is one of the
    canonical :data:`SKILL_LEVELS` (``beginner``..``expert``), case-insensitive,
    with unknown values ranking as 0 so an unseen level never crashes generation.
    This mirrors the backend ``SkillAssessmentDto`` / frontend ``UserSkillAssessment``
    so a user's leveled skills survive end to end instead of being flattened away.
    """

    name: str
    level: str = Field(default="beginner", description="beginner..expert")


class PersonProfile(BaseModel):
    """Who the path is generated for.

    ``skills`` carry per-skill proficiency levels — the source of truth for how
    experienced the person is. Step gating and synthesis verbosity derive an
    overall level from them via :func:`proficiency_rank`. ``tags`` keep the input
    forward-compatible with richer targeting.
    """

    working_area: str = Field(description="e.g. backend, frontend, devops")
    skills: list[SkillAssessment] = Field(default_factory=list[SkillAssessment])
    tags: list[str] = Field(default_factory=list)


class CitationRef(BaseModel):
    """A resolved reference to an ingested document chunk."""

    filename: str
    chunk_id: str
    source_url: str | None = None
