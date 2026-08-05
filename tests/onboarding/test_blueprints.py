from onboarding.blueprints import select_blueprints
from onboarding.models import Blueprint, BlueprintStep, PersonProfile

PROJECT = "project-1"


def _scope(tail: str, project: str = PROJECT) -> str:
    return f"project:{project}|{tail}"


def test_select_includes_global_and_matching_area() -> None:
    blueprints = [
        Blueprint(scope=_scope("global"), steps=[BlueprintStep(id="a", title="A")]),
        Blueprint(
            scope=_scope("area:backend"), steps=[BlueprintStep(id="b", title="B")]
        ),
        Blueprint(
            scope=_scope("area:frontend"), steps=[BlueprintStep(id="c", title="C")]
        ),
    ]
    profile = PersonProfile(working_area="backend")

    selected = select_blueprints(blueprints, profile, PROJECT)

    assert [b.scope for b in selected] == [_scope("global"), _scope("area:backend")]


def test_select_unknown_area_yields_global_only() -> None:
    blueprints = [
        Blueprint(scope=_scope("global"), steps=[BlueprintStep(id="a", title="A")]),
        Blueprint(
            scope=_scope("area:backend"), steps=[BlueprintStep(id="b", title="B")]
        ),
    ]
    profile = PersonProfile(working_area="quantum-computing")

    selected = select_blueprints(blueprints, profile, PROJECT)

    assert [b.scope for b in selected] == [_scope("global")]


def test_select_ignores_other_projects_blueprints() -> None:
    blueprints = [
        Blueprint(scope=_scope("global"), steps=[BlueprintStep(id="a", title="A")]),
        Blueprint(
            scope=_scope("global", "project-2"),
            steps=[BlueprintStep(id="b", title="B")],
        ),
        Blueprint(
            scope=_scope("area:backend", "project-2"),
            steps=[BlueprintStep(id="c", title="C")],
        ),
    ]
    profile = PersonProfile(working_area="backend")

    selected = select_blueprints(blueprints, profile, PROJECT)

    assert [b.scope for b in selected] == [_scope("global")]


def test_select_ignores_unqualified_legacy_blueprints() -> None:
    """Blueprints from before project separation belong to no project."""
    blueprints = [
        Blueprint(scope="global", steps=[BlueprintStep(id="a", title="A")]),
        Blueprint(scope="area:backend", steps=[BlueprintStep(id="b", title="B")]),
    ]
    profile = PersonProfile(working_area="backend")

    assert select_blueprints(blueprints, profile, PROJECT) == []
