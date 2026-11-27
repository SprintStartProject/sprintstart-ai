"""Turns the files of an offboarding risk map into the components of an interview.

The backend ranks *files* (deterministically, no LLM). People do not think in files, and
twenty questions about twenty files of one package would ask the same thing twenty
times, so the files are grouped by their directory and the interview is planned per
component. What each file's metrics say -- nobody else knows it, its lines were written
by people who left, it is a hotspot, it drags other files along -- becomes the *trigger*
a question has to answer to.
"""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath

_HOTSPOT_FROM = 0.5
_ABANDONED_FROM = 0.5
_ROOT = "(root)"


class Trigger(StrEnum):
    """The finding of the risk map that makes a question worth asking."""

    ORPHANED = "orphaned"
    ABANDONED = "abandoned"
    HOTSPOT = "hotspot"
    COUPLED = "coupled"
    CONCENTRATED = "concentrated"


# How urgent each finding is: a file nobody else knows loses everything with the leaver;
# one that only one other person knows still has a fallback.
TRIGGER_WEIGHT: dict[Trigger, float] = {
    Trigger.ORPHANED: 1.0,
    Trigger.ABANDONED: 0.9,
    Trigger.HOTSPOT: 0.8,
    Trigger.COUPLED: 0.7,
    Trigger.CONCENTRATED: 0.6,
}


@dataclass(frozen=True)
class RiskFile:
    """One file of the risk map with the metrics that put it there.

    ``level`` is the backend's ``ORPHANED`` (nobody who knows it stays) or ``HOARDED``
    (one does). ``kind`` is ``CODE``, ``PROSE`` or ``CONFIG`` when the file was
    measured.
    """

    repository: str
    path: str
    level: str
    score: float
    kind: str | None = None
    hotspot: float | None = None
    abandoned_share: float | None = None
    coupled_with: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True)
class Component:
    """Files of one directory of one repository, asked about together."""

    repository: str
    directory: str
    files: tuple[RiskFile, ...]

    @property
    def name(self) -> str:
        return f"{self.repository}/{self.directory}"

    @property
    def risk(self) -> float:
        """The highest score of its files: as urgent as its riskiest file."""
        return max(file.score for file in self.files)

    @property
    def triggers(self) -> frozenset[Trigger]:
        return frozenset(t for file in self.files for t in triggers_of(file))


def triggers_of(file: RiskFile) -> set[Trigger]:
    """The findings of one file; every file at risk is at least concentrated."""
    found = {Trigger.CONCENTRATED}
    if file.level == "ORPHANED":
        found.add(Trigger.ORPHANED)
    if (file.hotspot or 0.0) >= _HOTSPOT_FROM:
        found.add(Trigger.HOTSPOT)
    if (file.abandoned_share or 0.0) >= _ABANDONED_FROM:
        found.add(Trigger.ABANDONED)
    if file.coupled_with:
        found.add(Trigger.COUPLED)
    return found


def _directory(path: str) -> str:
    parent = str(PurePosixPath(path).parent)
    return _ROOT if parent == "." else parent


def components_of(
    files: list[RiskFile], max_components: int, max_files: int
) -> list[Component]:
    """Groups files by repository and directory, riskiest component first.

    Each component keeps at most ``max_files`` of its riskiest files, and only the
    ``max_components`` riskiest components are kept, so the interview stays short.
    """
    groups: dict[tuple[str, str], list[RiskFile]] = {}
    for file in files:
        groups.setdefault((file.repository, _directory(file.path)), []).append(file)
    components = [
        Component(
            repository,
            directory,
            tuple(sorted(members, key=lambda f: -f.score)[:max_files]),
        )
        for (repository, directory), members in groups.items()
    ]
    components.sort(key=lambda c: (-c.risk, c.name))
    return components[:max_components]
