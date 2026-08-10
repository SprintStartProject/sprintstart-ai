import json

from ingestion.source_role import SourceRole
from onboarding.generation import (
    corpus_fingerprint,
    filter_semantic_duplicates,
    generate_blueprints,
)
from onboarding.models import (
    Blueprint,
    BlueprintStep,
    SkeletonRef,
    StepRecord,
    content_id,
)
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

# Non-zero embedding so the stub store returns a perfect cosine match.
_EMBED = [1.0] + [0.0] * 767
_PROJECT = "project-1"
_SCOPE = f"project:{_PROJECT}|area:backend"


def _llm(steps: list[dict[str, object]]) -> StubLLMClient:
    llm = StubLLMClient(generate_response=json.dumps({"steps": steps}))
    llm.embedding = _EMBED
    return llm


def _store(*texts: str, project_ids: tuple[str, ...] = (_PROJECT,)) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id="a1",
                filename=f"doc{i}.md",
                text=text,
                embedding=_EMBED,
                project_ids=project_ids,
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


def _active(*steps: BlueprintStep, version: str = "1") -> Blueprint:
    """An authored active blueprint as the backend would pass it in."""
    return Blueprint(
        scope=_SCOPE, version=version, source="authored", steps=list(steps)
    )


def test_first_time_generation_drafts_grounded_steps() -> None:
    store = _store("backend onboarding deploy runbook local db setup")
    llm = _llm(
        [
            {
                "id": "deploy-runbook",
                "title": "Read the deploy runbook",
                "requirement": "required",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcomes = generate_blueprints(llm, store, project_id=_PROJECT, scopes=[_SCOPE])

    assert [(o.scope, o.status) for o in outcomes] == [(_SCOPE, "created")]
    bp = outcomes[0].blueprint
    assert bp is not None
    assert bp.source == "generated"
    assert bp.version == "1"
    assert [s.id for s in bp.steps] == [content_id("Read the deploy runbook")]
    assert bp.steps[0].citations[0].chunk_id == "c1"
    assert bp.provenance is not None
    assert bp.provenance.corpus_fingerprint == corpus_fingerprint(store, _PROJECT)


def test_unchanged_corpus_is_a_noop() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [{"id": "x", "title": "X", "requirement": "required", "chunk_ids": ["c1"]}]
    )

    first = generate_blueprints(llm, store, project_id=_PROJECT, scopes=[_SCOPE])
    # The backend persists the result and passes it back as the active blueprint.
    again = generate_blueprints(
        llm,
        store,
        project_id=_PROJECT,
        scopes=[_SCOPE],
        active=[first[0].blueprint],  # type: ignore[list-item]
    )

    assert again[0].status == "unchanged"


def test_corpus_change_updates_active_with_new_version() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [{"id": "x", "title": "X", "requirement": "required", "chunk_ids": ["c1"]}]
    )

    first = generate_blueprints(llm, store, project_id=_PROJECT, scopes=[_SCOPE])
    active = first[0].blueprint
    assert active is not None

    store.add(
        [
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="d2.md",
                text="new",
                embedding=_EMBED,
                project_ids=(_PROJECT,),
            )
        ]
    )
    outcomes = generate_blueprints(
        llm, store, project_id=_PROJECT, scopes=[_SCOPE], active=[active]
    )

    assert outcomes[0].status == "updated"
    assert outcomes[0].draft_version == "2"


def test_another_projects_corpus_change_does_not_invalidate_blueprints() -> None:
    """The fingerprint is per project, so a foreign ingest is not a change."""
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [{"id": "x", "title": "X", "requirement": "required", "chunk_ids": ["c1"]}]
    )

    first = generate_blueprints(llm, store, project_id=_PROJECT, scopes=[_SCOPE])
    active = first[0].blueprint
    assert active is not None

    store.add(
        [
            Chunk(
                id="other-1",
                artifact_id="other-artifact",
                filename="other.md",
                text="another project's secret runbook",
                embedding=_EMBED,
                project_ids=("project-2",),
            )
        ]
    )
    outcomes = generate_blueprints(
        llm, store, project_id=_PROJECT, scopes=[_SCOPE], active=[active]
    )

    assert outcomes[0].status == "unchanged"


def test_generation_ignores_other_projects_evidence() -> None:
    store = _store("another project's runbook", project_ids=("project-2",))
    llm = _llm([{"title": "Deploy", "chunk_ids": ["c1"]}])

    outcomes = generate_blueprints(llm, store, project_id=_PROJECT, scopes=[_SCOPE])

    assert outcomes[0].status == "skipped"
    assert outcomes[0].blueprint is None


def test_plain_scope_names_are_qualified_with_the_project() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm([{"title": "Deploy", "chunk_ids": ["c1"]}])

    outcomes = generate_blueprints(
        llm, store, project_id=_PROJECT, scopes=["area:backend"]
    )

    assert [o.scope for o in outcomes] == [_SCOPE]


def test_invariant_removal_is_blocked_and_reinjected() -> None:
    sec = "Security policy"
    active = _active(
        BlueprintStep(id=content_id(sec), title=sec, requirement="required"),
    )
    store = _store("backend onboarding deploy runbook")
    # The draft omits the required "security" step entirely.
    llm = _llm([{"title": "Deploy", "requirement": "recommended", "chunk_ids": ["c1"]}])

    outcomes = generate_blueprints(
        llm, store, project_id=_PROJECT, scopes=[_SCOPE], active=[active]
    )

    assert outcomes[0].status == "escalated"
    bp = outcomes[0].blueprint
    assert bp is not None
    sec_id = content_id(sec)
    ids = {s.id for s in bp.steps}
    assert sec_id in ids  # protected step re-injected, never silently dropped
    assert bp.provenance is not None
    assert any(sec_id in note for note in bp.provenance.notes)


def test_invariant_flag_protects_recommended_step() -> None:
    ethics = "Ethics training"
    active = _active(
        BlueprintStep(
            id=content_id(ethics),
            title=ethics,
            requirement="recommended",
            invariant=True,
        ),
    )
    store = _store("backend onboarding")
    llm = _llm([{"title": "Deploy", "chunk_ids": ["c1"]}])

    outcomes = generate_blueprints(
        llm, store, project_id=_PROJECT, scopes=[_SCOPE], active=[active]
    )

    assert outcomes[0].status == "escalated"
    bp = outcomes[0].blueprint
    assert bp is not None
    assert content_id(ethics) in {s.id for s in bp.steps}


def test_ungrounded_steps_are_dropped() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [
            {"id": "grounded", "title": "Grounded", "chunk_ids": ["c1"]},
            {"id": "ungrounded", "title": "Ungrounded", "chunk_ids": ["missing"]},
        ]
    )

    outcomes = generate_blueprints(llm, store, project_id=_PROJECT, scopes=[_SCOPE])

    bp = outcomes[0].blueprint
    assert bp is not None
    assert [s.id for s in bp.steps] == [content_id("Grounded")]


def test_identical_title_proposals_dedup() -> None:
    store = _store("backend onboarding deploy runbook")
    llm = _llm(
        [
            {"title": "Read the deploy runbook", "chunk_ids": ["c1"]},
            {"title": "Read the deploy runbook", "chunk_ids": ["c1"]},
        ]
    )

    outcomes = generate_blueprints(llm, store, project_id=_PROJECT, scopes=[_SCOPE])

    bp = outcomes[0].blueprint
    assert bp is not None
    # Same title -> same content id -> collapsed to a single step.
    assert [s.id for s in bp.steps] == [content_id("Read the deploy runbook")]


def test_empty_corpus_is_skipped() -> None:
    outcomes = generate_blueprints(
        _llm([]), StubVectorStore(), project_id=_PROJECT, scopes=[_SCOPE]
    )
    assert outcomes[0].status == "skipped"


def testfilter_semantic_duplicates_drops_similar_steps() -> None:
    """Area steps whose embeddings are too close to global steps are filtered."""
    similar_embed = [1.0, 0.1] + [0.0] * 766
    different_embed = [0.0, 1.0] + [0.0] * 766

    def embed_fn(text: str) -> list[float]:
        if "python" in text.lower():
            return similar_embed
        return different_embed

    global_steps = [
        BlueprintStep(
            id=content_id("Verify Python 3.12+ installation"),
            title="Verify Python 3.12+ installation",
            description="Ensure Python 3.12 or later is installed.",
        ),
    ]
    dup_id = content_id("Verify Python and Tooling Prerequisites")
    unique_id = content_id("Understand the RAG pipeline")
    pool = {
        dup_id: StepRecord(
            id=dup_id,
            title="Verify Python and Tooling Prerequisites",
            description="Ensure Python 3.12 or later is installed.",
        ),
        unique_id: StepRecord(
            id=unique_id,
            title="Understand the RAG pipeline",
            description="Learn how retrieval augmented generation works.",
        ),
    }
    refs = [SkeletonRef(id=dup_id), SkeletonRef(id=unique_id)]
    llm = StubLLMClient(embed_fn=embed_fn)

    kept = filter_semantic_duplicates(refs, pool, global_steps, llm)

    kept_ids = {r.id for r in kept}
    assert dup_id not in kept_ids  # similar to global → dropped
    assert unique_id in kept_ids  # different → kept


def testfilter_semantic_duplicates_keeps_dissimilar_steps() -> None:
    """Area steps with low similarity to global steps survive the filter."""
    call_count = {"n": 0}

    def embed_fn(text: str) -> list[float]:
        call_count["n"] += 1
        vec = [0.0] * 768
        vec[call_count["n"] % 768] = 1.0
        return vec

    global_steps = [
        BlueprintStep(
            id=content_id("Verify Python 3.12+"),
            title="Verify Python 3.12+",
        ),
    ]
    docker_id = content_id("Set up Docker Compose")
    pool = {
        docker_id: StepRecord(
            id=docker_id,
            title="Set up Docker Compose",
            description="Run docker-compose up for local development.",
        ),
    }
    refs = [SkeletonRef(id=docker_id)]
    llm = StubLLMClient(embed_fn=embed_fn)

    kept = filter_semantic_duplicates(refs, pool, global_steps, llm)

    assert [r.id for r in kept] == [docker_id]


def testfilter_semantic_duplicates_within_scope() -> None:
    """Two near-identical steps in the same scope collapse to the first."""

    def embed_fn(text: str) -> list[float]:
        # Both "verify python" variants map to the same vector; docker differs.
        if "python" in text.lower():
            return [1.0, 0.0] + [0.0] * 766
        return [0.0, 1.0] + [0.0] * 766

    first_id = content_id("Verify Python version")
    dup_id = content_id("Check the installed Python version")
    docker_id = content_id("Start Docker Compose")
    pool = {
        first_id: StepRecord(
            id=first_id,
            title="Verify Python version",
            description="Confirm the Python interpreter version.",
        ),
        dup_id: StepRecord(
            id=dup_id,
            title="Check the installed Python version",
            description="Confirm the Python interpreter version.",
        ),
        docker_id: StepRecord(
            id=docker_id,
            title="Start Docker Compose",
            description="Bring up the local stack.",
        ),
    }
    refs = [SkeletonRef(id=first_id), SkeletonRef(id=dup_id), SkeletonRef(id=docker_id)]
    llm = StubLLMClient(embed_fn=embed_fn)

    # No global steps → pure within-scope dedup (as for the global scope itself).
    kept = filter_semantic_duplicates(refs, pool, [], llm)

    kept_ids = [r.id for r in kept]
    assert first_id in kept_ids  # first occurrence wins
    assert dup_id not in kept_ids  # within-scope duplicate dropped
    assert docker_id in kept_ids  # genuinely different step survives


def test_corpus_fingerprint_changes_when_source_role_changes() -> None:
    """Reclassifying content as test material must invalidate blueprints.

    Chunk ids hash artifact_id, position and content only, so the id survives
    the reclassification — but test chunks are excluded from onboarding
    grounding, and a blueprint drafted from them is no longer supportable.
    """
    store = StubVectorStore()

    def chunk(source_role: SourceRole) -> Chunk:
        return Chunk(
            id="c1",
            artifact_id="a1",
            filename="doc.md",
            text="backend onboarding deploy runbook",
            embedding=_EMBED,
            project_ids=(_PROJECT,),
            source_role=source_role,
        )

    store.add([chunk("primary")])
    before = corpus_fingerprint(store, _PROJECT)

    store.add([chunk("test")])

    assert corpus_fingerprint(store, _PROJECT) != before


def test_corpus_fingerprint_changes_when_filename_changes() -> None:
    """The filename appears in citations, so a rename changes the output."""
    store = StubVectorStore()

    def chunk(filename: str) -> Chunk:
        return Chunk(
            id="c1",
            artifact_id="a1",
            filename=filename,
            text="backend onboarding deploy runbook",
            embedding=_EMBED,
            project_ids=(_PROJECT,),
        )

    store.add([chunk("runbook.md")])
    before = corpus_fingerprint(store, _PROJECT)

    store.add([chunk("docs/runbook.md")])

    assert corpus_fingerprint(store, _PROJECT) != before


def test_corpus_fingerprint_stable_for_unchanged_corpus() -> None:
    store = _store("backend onboarding deploy runbook")

    assert corpus_fingerprint(store, _PROJECT) == corpus_fingerprint(store, _PROJECT)
