# pyright: reportPrivateUsage=false
# Unit-tests the pipeline's gate internals directly (coverage/invariants/filter).
import pytest

from onboarding import pipeline
from onboarding.models import (
    Blueprint,
    BlueprintStep,
    CitationRef,
    OnboardingPath,
    PathPhase,
    PathStep,
    PersonProfile,
    QualityReport,
    SkillAssessment,
    content_id,
    experience_rank,
)
from onboarding.pipeline import (
    OnboardingPipeline,
    _build_phases,
    _enforce_coverage,
    _enforce_invariants,
    _step_applies,
)
from onboarding.quality import evaluate
from rag.types import Chunk, RetrievalFilters
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def _path(phases: list[PathPhase]) -> OnboardingPath:
    return OnboardingPath(
        working_area="backend",
        phases=phases,
        quality=QualityReport(
            coverage=0.0, grounded_ratio=0.0, ordering_valid=False, score=0.0
        ),
    )


def test_experience_rank_unknown_is_zero() -> None:
    assert experience_rank("junior") == 1
    assert experience_rank("astronaut") == 0
    assert experience_rank(None) == 0


def test_required_step_always_applies_regardless_of_proficiency() -> None:
    step = BlueprintStep(
        id="x", title="X", requirement="required", min_experience="senior"
    )
    profile = PersonProfile(working_area="backend")

    assert _step_applies(step, profile) is True


def test_recommended_step_filtered_by_min_experience() -> None:
    # A step's min_experience is gated against the person's overall proficiency,
    # derived from their highest skill level (advanced↔senior on the shared scale).
    step = BlueprintStep(
        id="x", title="X", requirement="recommended", min_experience="senior"
    )
    junior = PersonProfile(
        working_area="backend",
        skills=[SkillAssessment(name="python", level="beginner")],
    )
    senior = PersonProfile(
        working_area="backend",
        skills=[SkillAssessment(name="python", level="advanced")],
    )

    assert _step_applies(step, junior) is False
    assert _step_applies(step, senior) is True


def test_recommended_step_filtered_by_audience() -> None:
    step = BlueprintStep(
        id="x", title="X", requirement="recommended", audience=["frontend"]
    )
    backend = PersonProfile(working_area="backend")
    assert _step_applies(step, backend) is False

    tagged = PersonProfile(working_area="backend", tags=["frontend"])
    assert _step_applies(step, tagged) is True


def test_coverage_gate_reinjects_missing_required_step() -> None:
    blueprints = [
        Blueprint(
            scope="global",
            steps=[
                BlueprintStep(id="sec", title="Security", requirement="required"),
                BlueprintStep(id="acc", title="Accounts", requirement="required"),
            ],
        )
    ]
    # Path is missing the required "acc" step.
    phases = [
        PathPhase(
            title="Getting started",
            steps=[PathStep(id="sec", title="Security", requirement="required")],
        )
    ]
    profile = PersonProfile(working_area="backend")

    _enforce_coverage(phases, blueprints, profile)

    present = {s.id for p in phases for s in p.steps}
    assert present == {"sec", "acc"}


def test_invariants_gate_is_noop_with_no_invariants() -> None:
    phases = [
        PathPhase(
            title="Getting started",
            steps=[PathStep(id="acc", title="Accounts", requirement="required")],
        )
    ]

    _enforce_invariants(phases)

    ids = [s.id for p in phases for s in p.steps]
    assert ids == ["acc"]


def test_invariants_gate_injects_missing_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invariant = PathStep(id="must-have", title="Must have", requirement="required")
    monkeypatch.setattr(pipeline, "_INVARIANTS", [invariant])
    phases = [PathPhase(title="Getting started", steps=[])]

    _enforce_invariants(phases)

    assert [s.id for p in phases for s in p.steps] == ["must-have"]


def test_quality_rubric_scores_perfect_path() -> None:
    phases = [
        PathPhase(
            title="Getting started",
            steps=[
                PathStep(id="sec", title="Security", requirement="required"),
                PathStep(
                    id="llm-1",
                    title="Extra",
                    requirement="recommended",
                    origin="llm",
                    citations=[CitationRef(filename="a.md", chunk_id="c1")],
                ),
            ],
        )
    ]
    path = _path(phases)

    report = evaluate(path, required_step_ids={"sec"})

    assert report.coverage == 1.0
    assert report.grounded_ratio == 1.0
    assert report.ordering_valid is True
    assert report.score == 1.0


def test_quality_rubric_flags_missing_coverage_and_bad_ordering() -> None:
    phases = [
        PathPhase(
            title="Getting started",
            steps=[
                PathStep(id="rec", title="Recommended", requirement="recommended"),
                PathStep(id="req", title="Required", requirement="required"),
            ],
        )
    ]
    path = _path(phases)

    report = evaluate(path, required_step_ids={"req", "missing"})

    assert report.coverage == 0.5
    assert report.ordering_valid is False
    assert any("missing required" in n for n in report.notes)


def test_content_id_is_stable_and_normalized() -> None:
    # Identity is the normalized title: case- and whitespace-insensitive, stable.
    assert content_id("Set up the DB") == content_id("set up the db")
    assert content_id("  Set   up  the DB ") == content_id("Set up the DB")
    assert content_id("Set up the DB").startswith("step-")
    assert content_id("A") != content_id("B")


def test_skill_tag_match_surfaces_step_outside_audience() -> None:
    # A devops-only step the person would not otherwise see, pulled in because
    # they listed kubernetes as a skill (skills are role-orthogonal).
    step = BlueprintStep(
        id="x",
        title="Set up the cluster",
        requirement="recommended",
        audience=["devops"],
        tags=["kubernetes"],
    )
    skilled = PersonProfile(
        working_area="backend",
        skills=[SkillAssessment(name="kubernetes", level="advanced")],
    )
    assert _step_applies(step, skilled) is True

    # Without the matching skill, the devops-only step stays hidden for backend.
    plain = PersonProfile(working_area="backend")
    assert _step_applies(step, plain) is False
    # Surfacing by skill never promotes a recommended step to required.
    assert step.requirement == "recommended"


def test_build_phases_dedups_step_across_scopes() -> None:
    shared = BlueprintStep(
        id=content_id("Shared step"), title="Shared step", requirement="required"
    )
    blueprints = [
        Blueprint(scope="global", steps=[shared]),
        Blueprint(scope="area:backend", steps=[shared.model_copy(deep=True)]),
    ]
    profile = PersonProfile(working_area="backend")

    phases, _ = _build_phases(blueprints, profile)

    all_ids = [s.id for phase in phases for s in phase.steps]
    assert all_ids.count(content_id("Shared step")) == 1  # earlier scope wins


def test_cross_scope_status_merge_required_wins() -> None:
    # The same step is recommended globally but required for backend; the
    # rendered path must show it once, as required.
    recommended = BlueprintStep(
        id=content_id("Shared"), title="Shared", requirement="recommended"
    )
    required = recommended.model_copy(update={"requirement": "required"})
    blueprints = [
        Blueprint(scope="global", steps=[recommended]),
        Blueprint(scope="area:backend", steps=[required]),
    ]
    profile = PersonProfile(working_area="backend")

    phases, _ = _build_phases(blueprints, profile)

    rendered = [s for p in phases for s in p.steps if s.id == content_id("Shared")]
    assert len(rendered) == 1
    assert rendered[0].requirement == "required"


def test_retrieve_per_step_uses_step_specific_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_retrieve(*, question: str, **_: object) -> list[object]:
        captured.append(question)
        return []

    monkeypatch.setattr(pipeline, "hybrid_retrieve", fake_retrieve)

    store = StubVectorStore()
    store.add(
        [Chunk(id="c1", artifact_id="a1", filename="d.md", text="x", embedding=[0.1])]
    )
    pipe = OnboardingPipeline(StubLLMClient(), store)
    steps = [
        BlueprintStep(id="x", title="Install dependencies", requirement="required"),
        BlueprintStep(
            id="y", title="Understand the RAG pipeline", requirement="recommended"
        ),
    ]

    pipe._retrieve_per_step(steps, "project-1")

    assert len(captured) == 2
    assert any("Install dependencies" in q for q in captured)
    assert any("Understand the RAG pipeline" in q for q in captured)


def test_retrieve_per_step_scopes_evidence_to_the_project(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[RetrievalFilters | None] = []

    def fake_retrieve(
        *, filters: RetrievalFilters | None = None, **_: object
    ) -> list[object]:
        captured.append(filters)
        return []

    monkeypatch.setattr(pipeline, "hybrid_retrieve", fake_retrieve)

    store = StubVectorStore()
    store.add(
        [Chunk(id="c1", artifact_id="a1", filename="d.md", text="x", embedding=[0.1])]
    )
    pipe = OnboardingPipeline(StubLLMClient(), store)

    pipe._retrieve_per_step([BlueprintStep(id="x", title="Install")], "project-1")

    assert captured == [RetrievalFilters(project_id="project-1")]


def test_build_phases_drops_semantic_duplicate_across_scopes() -> None:
    """An area step with different id but overlapping content is dropped."""
    global_step = BlueprintStep(
        id=content_id("Install Python 3.12 and verify version"),
        title="Install Python 3.12 and verify version",
        description="Ensure Python 3.12 is installed on your machine and verify.",
        requirement="recommended",
    )
    # Different title/id but very similar content (high Jaccard overlap).
    area_step = BlueprintStep(
        id=content_id("Verify Python 3.12 installation and version"),
        title="Verify Python 3.12 installation and version",
        description="Ensure Python 3.12 is installed on your machine and verify.",
        requirement="recommended",
    )
    blueprints = [
        Blueprint(scope="global", steps=[global_step]),
        Blueprint(scope="area:backend", steps=[area_step]),
    ]
    profile = PersonProfile(working_area="backend")

    phases, _ = _build_phases(blueprints, profile)

    all_ids = [s.id for p in phases for s in p.steps]
    assert global_step.id in all_ids
    assert area_step.id not in all_ids  # semantic duplicate dropped


def test_build_phases_keeps_semantically_different_steps() -> None:
    """Two steps with genuinely different content both survive."""
    global_step = BlueprintStep(
        id=content_id("Install Python 3.12"),
        title="Install Python 3.12",
        description="Install the Python runtime.",
        requirement="recommended",
    )
    area_step = BlueprintStep(
        id=content_id("Configure Docker networking"),
        title="Configure Docker networking",
        description="Set up container networking for local dev.",
        requirement="recommended",
    )
    blueprints = [
        Blueprint(scope="global", steps=[global_step]),
        Blueprint(scope="area:backend", steps=[area_step]),
    ]
    profile = PersonProfile(working_area="backend")

    phases, _ = _build_phases(blueprints, profile)

    all_ids = [s.id for p in phases for s in p.steps]
    assert global_step.id in all_ids
    assert area_step.id in all_ids


def test_build_phases_deduplicates_required_step_when_concept_already_covered() -> None:
    """A required area step is dropped when its concept is already covered globally.

    The old behaviour kept required steps unconditionally (bypassing semantic
    dedup), which caused duplicate steps to appear across phases.  The new
    behaviour deduplicates all steps; the coverage gate only re-injects a
    missing required step when no semantically equivalent step is present.
    """
    global_step = BlueprintStep(
        id=content_id("Install Python 3.12 and verify version"),
        title="Install Python 3.12 and verify version",
        description="Ensure Python 3.12 is installed on your machine and verify.",
        requirement="recommended",
    )
    area_step = BlueprintStep(
        id=content_id("Verify Python 3.12 installation and version"),
        title="Verify Python 3.12 installation and version",
        description="Ensure Python 3.12 is installed on your machine and verify.",
        requirement="required",
    )
    blueprints = [
        Blueprint(scope="global", steps=[global_step]),
        Blueprint(scope="area:backend", steps=[area_step]),
    ]
    profile = PersonProfile(working_area="backend")

    phases, _ = _build_phases(blueprints, profile)

    all_ids = [s.id for p in phases for s in p.steps]
    # Global step covers the concept; area step is a semantic duplicate and
    # should be dropped — the concept appears exactly once in the path.
    assert global_step.id in all_ids
    assert area_step.id not in all_ids


def test_enforce_coverage_reinjects_unique_required_step() -> None:
    """A required step with no semantic equivalent in the path is re-injected."""
    global_step = BlueprintStep(
        id=content_id("Set up IDE"),
        title="Set up IDE",
        description="Install and configure your development environment.",
        requirement="recommended",
    )
    area_step = BlueprintStep(
        id=content_id("Configure Kubernetes access"),
        title="Configure Kubernetes access",
        description=(
            "Set up kubeconfig and verify cluster connectivity for deployments."
        ),
        requirement="required",
    )
    blueprints = [
        Blueprint(scope="global", steps=[global_step]),
        Blueprint(scope="area:backend", steps=[area_step]),
    ]
    profile = PersonProfile(working_area="backend")

    phases, _ = _build_phases(blueprints, profile)

    all_ids = [s.id for p in phases for s in p.steps]
    assert global_step.id in all_ids
    assert area_step.id in all_ids  # genuinely unique required step kept


def test_path_serializes_to_yaml_round_trip() -> None:
    import yaml

    path = _path([PathPhase(title="P", steps=[PathStep(id="a", title="A")])])

    loaded = yaml.safe_load(path.to_yaml())

    assert loaded["working_area"] == "backend"
    assert loaded["phases"][0]["steps"][0]["id"] == "a"
