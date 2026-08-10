import json
from collections.abc import Generator

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from onboarding import starter_work
from onboarding.corpus import corpus_fingerprint
from onboarding.progress import ProgressEvent
from onboarding.starter_work import (
    generate_starter_work_pool,
    stream_starter_work_pool,
)
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_EMBED = [1.0] + [0.0] * 767


def _collect[T](
    generator: Generator[ProgressEvent, None, T],
) -> tuple[list[ProgressEvent], T]:
    """Drain a progress generator, keeping both the events and the returned value."""
    events: list[ProgressEvent] = []
    try:
        while True:
            events.append(next(generator))
    except StopIteration as stop:
        return events, stop.value


def _metadata_store() -> IngestionMetadataStore:
    return IngestionMetadataStore(path=":memory:")


def _issue_artifact(**overrides: object) -> ArtifactRecord:
    defaults: dict[str, object] = dict(
        id="a1",
        filename="issue-1.md",
        content_type="text/plain",
        source_type="github",
        size_bytes=10,
        chunk_count=1,
        status="completed",
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        artifact_type="ISSUE",
        state="OPEN",
        source_id="github:org/repo:ISSUE:1",
        source_url="https://github.com/org/repo/issues/1",
        labels=["good first issue"],
    )
    defaults.update(overrides)
    return ArtifactRecord(**defaults)  # type: ignore[arg-type]


def _add_issue_chunk(
    store: StubVectorStore, artifact_id: str, title: str, body: str
) -> None:
    store.add(
        [
            Chunk(
                id=f"chunk-{artifact_id}",
                artifact_id=artifact_id,
                filename=f"{artifact_id}.md",
                text=f"# {title}\n\n{body}",
                embedding=_EMBED,
            )
        ]
    )


def _llm(tasks: list[dict[str, object]]) -> StubLLMClient:
    return StubLLMClient(generate_response=json.dumps({"tasks": tasks}))


def test_mines_safely_scoped_task_from_open_issue() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(
        store, "a1", "Fix typo in README", "The install section has a typo."
    )
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "Fix a typo in the README install section.",
                "competency_keys": ["docs"],
                "rationale": "Single-file text fix.",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "proposed"
    assert len(outcome.tasks) == 1
    task = outcome.tasks[0]
    assert task.source_id == "github:org/repo:ISSUE:1"
    assert task.title == "Fix typo in README"
    assert task.competency_keys == ["docs"]
    assert task.citations[0].source_url == "https://github.com/org/repo/issues/1"
    assert outcome.provenance is not None
    assert outcome.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_mines_a_tracker_issue_that_is_not_from_github() -> None:
    """A Jira-shaped issue mines exactly like a GitHub one.

    ⚠️ This is the whole of P4's corpus half seen from this side. The filter here
    was always source-agnostic -- but nothing set ``state`` on an ingested Jira
    issue, so every one of them fell out at the ``state == "OPEN"`` line and a
    project whose tracker is Jira mined an *empty pool*. It read as "no good
    first issues here" rather than "we cannot see your tracker", which is why it
    survived the connector landing. The backend now folds Jira's status category
    into that state; this pins that nothing on this side needs GitHub's shape --
    no ``owner/repo`` in the source id, no labels, no GitHub URL.
    """
    metadata_store = _metadata_store()
    metadata_store.save_artifact(
        _issue_artifact(
            source_type="jira",
            source_id="10042",
            source_url="https://example.atlassian.net/browse/ONB-42",
            labels=[],
        )
    )
    store = StubVectorStore()
    _add_issue_chunk(
        store, "a1", "Write up the retro notes", "Summarise what the team agreed."
    )
    llm = _llm(
        [
            {
                "source_id": "10042",
                "safely_scoped": True,
                "summary": "Write up the retro notes.",
                "competency_keys": ["facilitation"],
                "rationale": "Bounded, and the outcome is a single document.",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "proposed"
    task = outcome.tasks[0]
    assert task.source_id == "10042"
    assert task.title == "Write up the retro notes"
    assert task.citations[0].source_url == (
        "https://example.atlassian.net/browse/ONB-42"
    )


def test_issue_somebody_else_is_already_on_is_not_proposed() -> None:
    """Starter work is work a hire can *take*.

    An open issue somebody else is assigned to is not available, however open it
    is. A Jira board assigns its in-progress tickets, so without this the pool
    offered new hires work other people were doing -- and a hire cannot tell,
    because the proposal reads exactly like any other.
    """
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact(has_assignee=True))
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "x",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"
    assert outcome.tasks == []


def test_an_unassigned_issue_is_still_proposed() -> None:
    """A definite ``False`` is the tracker saying nobody has taken it."""
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact(has_assignee=False))
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo in README", "The install section.")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "Fix a typo.",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "proposed"
    assert len(outcome.tasks) == 1


def test_an_issue_whose_assignment_is_unknown_is_still_proposed() -> None:
    """⚠️ ``None`` is "we cannot tell", never "nobody".

    GitHub issues have assignees this system does not ingest, so every one of
    them arrives unknown. Treating unknown as taken would empty the pool for
    every existing project; treating it as free is what P0 guaranteed --
    engineering behaviour byte-identical. The default fixture carries no
    assignment, which is exactly the GitHub case.
    """
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    assert metadata_store.get_artifact("a1") is not None
    assert metadata_store.get_artifact("a1").has_assignee is None  # type: ignore[union-attr]

    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo in README", "The install section.")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "Fix a typo.",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "proposed"
    assert len(outcome.tasks) == 1


def test_closed_issue_is_never_proposed() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact(state="CLOSED"))
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "x",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"
    assert outcome.tasks == []


def test_already_pooled_issue_is_not_reproposed() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "x",
            }
        ]
    )

    outcome = generate_starter_work_pool(
        llm, store, metadata_store, active_source_ids=["github:org/repo:ISSUE:1"]
    )

    assert outcome.status == "skipped"
    assert outcome.tasks == []


def test_competency_key_outside_known_set_is_dropped() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "x",
                "competency_keys": ["docs", "invented-key"],
            }
        ]
    )

    outcome = generate_starter_work_pool(
        llm, store, metadata_store, active_competency_keys=["docs"]
    )

    assert outcome.tasks[0].competency_keys == ["docs"]


def test_unchanged_corpus_with_matching_fingerprint_is_a_noop() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = _llm([])

    outcome = generate_starter_work_pool(
        llm, store, metadata_store, last_fingerprint=corpus_fingerprint(store)
    )

    assert outcome.status == "unchanged"
    assert outcome.tasks == []


def test_empty_corpus_is_skipped() -> None:
    metadata_store = _metadata_store()
    store = StubVectorStore()
    llm = _llm([])

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"


def test_not_safely_scoped_is_dropped() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(
        store, "a1", "Rewrite the whole auth system", "big, vague, cross-cutting task"
    )
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": False,
                "rationale": "too large",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"
    assert outcome.tasks == []


def test_invalid_llm_json_is_skipped_not_raised() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(_issue_artifact())
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo", "body")
    llm = StubLLMClient(generate_response="not json at all")

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert outcome.status == "skipped"


def test_issue_without_indexed_chunks_is_excluded() -> None:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(
        _issue_artifact(id="a1", source_id="github:org/repo:ISSUE:1")
    )
    metadata_store.save_artifact(
        _issue_artifact(id="a2", source_id="github:org/repo:ISSUE:2")
    )
    store = StubVectorStore()
    # Only a2 has been embedded; a1's issue text isn't in the vector store yet.
    _add_issue_chunk(store, "a2", "Fix typo", "body")
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:2",
                "safely_scoped": True,
                "summary": "x",
            }
        ]
    )

    outcome = generate_starter_work_pool(llm, store, metadata_store)

    assert [t.source_id for t in outcome.tasks] == ["github:org/repo:ISSUE:2"]


# --- streaming -----------------------------------------------------------------


def _two_open_issues() -> tuple[IngestionMetadataStore, StubVectorStore]:
    metadata_store = _metadata_store()
    metadata_store.save_artifact(
        _issue_artifact(id="a1", source_id="github:org/repo:ISSUE:1")
    )
    metadata_store.save_artifact(
        _issue_artifact(id="a2", source_id="github:org/repo:ISSUE:2")
    )
    store = StubVectorStore()
    _add_issue_chunk(store, "a1", "Fix typo in README", "The install section typo.")
    _add_issue_chunk(store, "a2", "Add a unit test", "Cover the date formatter.")
    return metadata_store, store


def test_stream_emits_a_scoped_task_as_an_item_and_a_done() -> None:
    metadata_store, store = _two_open_issues()
    llm = _llm(
        [
            {
                "source_id": "github:org/repo:ISSUE:1",
                "safely_scoped": True,
                "summary": "Fix a typo.",
                "competency_keys": [],
                "rationale": "Small, contained.",
            },
            {
                "source_id": "github:org/repo:ISSUE:2",
                "safely_scoped": False,
                "summary": "",
                "competency_keys": [],
                "rationale": "Needs discussion.",
            },
        ]
    )

    events, outcome = _collect(stream_starter_work_pool(llm, store, metadata_store))

    # Only the safely-scoped issue is emitted as an item -- the rejected one never
    # appears, which is the whole promise of an `item` event.
    items = [e for e in events if e["type"] == "item"]
    assert [i["item"]["source_id"] for i in items] == [  # type: ignore[index]
        "github:org/repo:ISSUE:1"
    ]
    # seq is monotonic across the whole stream.
    assert [e["seq"] for e in events] == list(range(len(events)))
    assert events[-1]["type"] == "done"
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.status == "proposed"


def test_streaming_result_equals_the_non_streaming_pool() -> None:
    metadata_store, store = _two_open_issues()
    tasks: list[dict[str, object]] = [
        {
            "source_id": "github:org/repo:ISSUE:1",
            "safely_scoped": True,
            "summary": "Fix a typo.",
            "competency_keys": [],
            "rationale": "Small, contained.",
        }
    ]

    _, streamed = _collect(stream_starter_work_pool(_llm(tasks), store, metadata_store))
    synchronous = generate_starter_work_pool(_llm(tasks), store, metadata_store)

    assert streamed.status == synchronous.status
    assert [t.source_id for t in streamed.tasks] == [
        t.source_id for t in synchronous.tasks
    ]


def test_an_empty_corpus_streams_a_skipped_done_not_an_error() -> None:
    events, outcome = _collect(
        stream_starter_work_pool(_llm([]), StubVectorStore(), _metadata_store())
    )

    assert outcome.status == "skipped"
    assert events[-1]["type"] == "done"
    assert "error" not in [e["type"] for e in events]


def _many_issues(
    metadata_store: IngestionMetadataStore, store: StubVectorStore, count: int
) -> None:
    """``count`` open, unassigned, chunked issues.

    Ids are zero-padded so the sort order is obvious to read.
    """
    for i in range(count):
        aid = f"a{i:03d}"
        metadata_store.save_artifact(
            _issue_artifact(
                id=aid,
                source_id=f"github:org/repo:ISSUE:{i:03d}",
                source_url=f"https://github.com/org/repo/issues/{i}",
            )
        )
        _add_issue_chunk(store, aid, f"Issue {i}", "body")


def test_a_corpus_larger_than_the_cap_judges_only_the_cap() -> None:
    """⚠️ Uncapped, one prompt grows with the corpus until it blocks whoever ran it."""
    metadata_store = _metadata_store()
    store = StubVectorStore()
    _many_issues(metadata_store, store, starter_work._MAX_CANDIDATES_PER_RUN + 7)

    outcome = generate_starter_work_pool(_llm([]), store, metadata_store)

    assert outcome.candidates_considered == starter_work._MAX_CANDIDATES_PER_RUN


def test_what_the_cap_left_out_is_counted_not_dropped_silently() -> None:
    """A pool missing most of the corpus reads as "no good first work here"."""
    metadata_store = _metadata_store()
    store = StubVectorStore()
    _many_issues(metadata_store, store, starter_work._MAX_CANDIDATES_PER_RUN + 7)

    outcome = generate_starter_work_pool(_llm([]), store, metadata_store)

    assert any("7 more eligible issue(s)" in note for note in outcome.notes)


def test_a_corpus_within_the_cap_says_nothing_about_deferral() -> None:
    """The note is about a real remainder, never noise on an ordinary run."""
    metadata_store = _metadata_store()
    store = StubVectorStore()
    _many_issues(metadata_store, store, 3)

    outcome = generate_starter_work_pool(_llm([]), store, metadata_store)

    assert outcome.candidates_considered == 3
    assert not any("eligible issue(s)" in note for note in outcome.notes)


def test_the_capped_slice_is_stable_across_runs() -> None:
    """⚠️ A stable order is what makes the cap fair.

    An arbitrary one re-judges the same issues every run while others are never
    reached at all -- the deferred remainder would never actually be picked up.
    """
    metadata_store = _metadata_store()
    store = StubVectorStore()
    _many_issues(metadata_store, store, starter_work._MAX_CANDIDATES_PER_RUN + 7)

    first, _ = starter_work._load_candidates(
        store, metadata_store, exclude_source_ids=set()
    )
    second, _ = starter_work._load_candidates(
        store, metadata_store, exclude_source_ids=set()
    )

    assert [c.source_id for c in first] == [c.source_id for c in second]


def test_issues_already_pooled_do_not_consume_the_cap() -> None:
    """Dedup happens before the cap, so a full pool does not starve the next run."""
    metadata_store = _metadata_store()
    store = StubVectorStore()
    _many_issues(metadata_store, store, starter_work._MAX_CANDIDATES_PER_RUN + 7)
    already = {f"github:org/repo:ISSUE:{i:03d}" for i in range(7)}

    candidates, eligible_total = starter_work._load_candidates(
        store, metadata_store, exclude_source_ids=already
    )

    assert eligible_total == starter_work._MAX_CANDIDATES_PER_RUN
    assert len(candidates) == starter_work._MAX_CANDIDATES_PER_RUN
    assert not any(c.source_id in already for c in candidates)
