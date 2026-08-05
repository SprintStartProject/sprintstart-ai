from onboarding.scope import Scope


def test_parse_unqualified_scopes() -> None:
    """Scopes from before project separation still parse, with no project."""
    global_scope = Scope.parse("global")
    assert global_scope.is_global
    assert global_scope.area is None
    assert global_scope.project is None

    area = Scope.parse("area:backend")
    assert not area.is_global
    assert area.area == "backend"
    assert area.project is None


def test_parse_project_qualified_scopes() -> None:
    global_scope = Scope.parse("project:p1|global")
    assert global_scope.is_global
    assert global_scope.area is None
    assert global_scope.project == "p1"

    area = Scope.parse("project:p1|area:backend")
    assert not area.is_global
    assert area.area == "backend"
    assert area.project == "p1"


def test_build_produces_qualified_scopes() -> None:
    assert Scope.build("p1").raw == "project:p1|global"
    assert Scope.build("p1", "backend").raw == "project:p1|area:backend"


def test_with_project_qualifies_and_requalifies() -> None:
    assert Scope.parse("global").with_project("p1").raw == "project:p1|global"
    assert (
        Scope.parse("area:backend").with_project("p1").raw == "project:p1|area:backend"
    )
    # Re-qualifying to a different project replaces the project segment rather
    # than nesting it.
    assert (
        Scope.parse("project:p1|area:backend").with_project("p2").raw
        == "project:p2|area:backend"
    )


def test_with_project_is_idempotent_for_the_same_project() -> None:
    scope = Scope.parse("project:p1|area:backend")

    assert scope.with_project("p1") is scope


def test_build_round_trips_through_parse() -> None:
    for scope in (Scope.build("p1"), Scope.build("p1", "backend")):
        assert Scope.parse(scope.raw) == scope
