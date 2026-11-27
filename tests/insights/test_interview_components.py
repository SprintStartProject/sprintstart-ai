from dataclasses import replace

from insights.interview_components import RiskFile, Trigger, components_of, triggers_of

_BASE = RiskFile(repository="github:acme/app", path="a.kt", level="HOARDED", score=0.5)


def _file(path: str, score: float = 0.5) -> RiskFile:
    return replace(_BASE, path=path, score=score)


def test_every_file_at_risk_is_at_least_concentrated() -> None:
    assert triggers_of(_BASE) == {Trigger.CONCENTRATED}


def test_each_finding_of_the_risk_map_becomes_a_trigger() -> None:
    file = replace(
        _BASE,
        level="ORPHANED",
        hotspot=0.5,
        abandoned_share=0.5,
        coupled_with=("b.kt",),
    )

    assert triggers_of(file) == set(Trigger)


def test_findings_below_their_threshold_are_no_triggers() -> None:
    file = replace(_BASE, hotspot=0.49, abandoned_share=0.49)

    assert triggers_of(file) == {Trigger.CONCENTRATED}


def test_files_are_grouped_by_repository_and_directory() -> None:
    files = [
        _file("src/auth/Login.kt"),
        _file("src/auth/Token.kt"),
        _file("README.md"),
        replace(_file("src/auth/Login.kt"), repository="github:acme/other"),
    ]

    names = {c.name for c in components_of(files, 10, 10)}

    assert names == {
        "github:acme/app/src/auth",
        "github:acme/app/(root)",
        "github:acme/other/src/auth",
    }


def test_the_riskiest_component_comes_first_and_the_list_is_capped() -> None:
    files = [_file("low/a.kt", 0.1), _file("high/a.kt", 0.9), _file("mid/a.kt", 0.5)]

    components = components_of(files, 2, 10)

    assert [c.directory for c in components] == ["high", "mid"]
    assert components[0].risk == 0.9


def test_a_component_keeps_only_its_riskiest_files() -> None:
    files = [_file("src/a.kt", 0.2), _file("src/b.kt", 0.8), _file("src/c.kt", 0.5)]

    component = components_of(files, 1, 2)[0]

    assert [f.path for f in component.files] == ["src/b.kt", "src/c.kt"]


def test_a_component_has_the_findings_of_all_its_files() -> None:
    files = [
        replace(_file("src/a.kt"), level="ORPHANED"),
        replace(_file("src/b.kt"), hotspot=0.9),
    ]

    component = components_of(files, 1, 5)[0]

    assert component.triggers == {
        Trigger.ORPHANED,
        Trigger.HOTSPOT,
        Trigger.CONCENTRATED,
    }
