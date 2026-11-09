from onboarding.orientation import assemble_orientation, stream_orientation
from onboarding.orientation_models import OrientationOutcome
from rag.types import Chunk
from tests.onboarding.orientation_fixtures import PIDS, TITLE, payload, section
from tests.onboarding.pipeline_fixtures import EMBED, collect, make_llm, make_store
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


def test_assembles_a_step_segmented_packet_with_the_sources_it_drew_on() -> None:
    store = make_store(
        "run make dev to start the service locally with the reports cache",
        "pull requests need one review and a green pipeline before merge",
    )
    llm = make_llm(
        payload(
            section("SET_UP", "Run it locally", ["c1"]),
            section("OPEN_THE_PR", "How review works here", ["c2"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.status == "assembled"
    packet = outcome.packet
    assert packet is not None
    assert [s.step for s in packet.sections] == ["SET_UP", "OPEN_THE_PR"]
    assert packet.task_title == TITLE
    # The packet states which existing material it stands on.
    assert {s.filename for s in packet.sources} == {"doc1.md", "doc2.md"}


def test_sections_are_ordered_by_step_not_by_the_model() -> None:
    """A hire must never be handed "open the PR" before "find the code"."""
    store = make_store(
        "pull request review policy", "the reports module lives in src/api"
    )
    llm = make_llm(
        payload(
            section("OPEN_THE_PR", "Review", ["c1"]),
            section("FIND_THE_CODE", "Where it lives", ["c2"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.packet is not None
    assert [s.step for s in outcome.packet.sections] == ["FIND_THE_CODE", "OPEN_THE_PR"]


def test_drops_an_ungrounded_section_but_keeps_the_packet() -> None:
    """A statement with no source does not ship -- that is the whole rule."""
    store = make_store("run make dev to start the service locally")
    llm = make_llm(
        payload(
            section("SET_UP", "Grounded", ["c1"]),
            section("CHECK_LOCALLY", "Invented", ["nope"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.packet is not None
    assert [s.title for s in outcome.packet.sections] == ["Grounded"]
    assert outcome.sections_dropped == 1


def test_no_section_kind_is_exempt_from_citing() -> None:
    """Unlike a module's TASK/CHECK pages, every packet section is a claim."""
    store = make_store("run make dev to start the service locally")
    llm = make_llm(
        payload(
            section("SET_UP", "Grounded", ["c1"]),
            section("MAKE_THE_CHANGE", "Uncited", []),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.packet is not None
    assert [s.step for s in outcome.packet.sections] == ["SET_UP"]


def test_drops_a_section_of_a_step_that_does_not_exist() -> None:
    store = make_store("run make dev to start the service locally")
    llm = make_llm(
        payload(
            section("SET_UP", "Grounded", ["c1"]),
            section("DEPLOY_IT", "Not a step of ours", ["c1"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.packet is not None
    assert [s.step for s in outcome.packet.sections] == ["SET_UP"]
    assert outcome.sections_dropped == 1


def test_collapses_sources_that_restate_each_other() -> None:
    """The README and the wiki saying the same thing is load paid for twice."""
    duplicate = "run make dev to start the reports service locally on port 8080"
    store = make_store(
        duplicate, duplicate + " .", "pull requests need one approving review"
    )
    llm = make_llm(payload(section("SET_UP", "Run it locally", ["c1"])))

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.chunks_collapsed == 1
    assert outcome.chunks_retrieved == 2


def test_collapses_two_sections_of_a_step_that_say_the_same_thing() -> None:
    store = make_store("run make dev to start the service locally on port 8080")
    body = "Run `make dev` to start the service locally on port 8080."
    llm = make_llm(
        payload(
            section("SET_UP", "Run it", ["c1"], body=body),
            section(
                "SET_UP", "Starting the service", ["c1"], body=body + " Then wait."
            ),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.packet is not None
    assert [s.title for s in outcome.packet.sections] == ["Run it"]
    assert outcome.sections_dropped == 1


def test_skips_when_every_section_is_ungrounded() -> None:
    """An empty state, never a fabricated packet."""
    store = make_store("run make dev to start the service locally")
    llm = make_llm(payload(section("SET_UP", "Invented", ["nope"])))

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.status == "skipped"
    assert outcome.packet is None


def test_empty_corpus_is_skipped_not_an_empty_packet() -> None:
    outcome = assemble_orientation(
        make_llm(payload()), StubVectorStore(), task_title=TITLE, project_ids=PIDS
    )

    assert outcome.status == "skipped"
    assert outcome.packet is None
    assert outcome.notes == ["corpus is empty"]


def test_unparseable_output_is_skipped_not_a_half_packet() -> None:
    store = make_store("run make dev to start the service locally")
    llm = StubLLMClient(generate_response="I'm afraid I can't do that.")
    llm.embedding = EMBED

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.status == "skipped"
    assert outcome.packet is None


def test_broken_json_is_corrected_once_rather_than_losing_the_packet() -> None:
    """A local model's one missing quote used to cost the hire the whole packet."""
    store = make_store("run make dev to start the service locally")
    llm = make_llm(payload(section("SET_UP", "Run it locally", ["c1"])))
    good = llm.generate
    replies = iter(['{"summary": "cut off', None])
    seen: list[list[object]] = []

    def flaky(messages: list[object], **kwargs: object) -> str:
        seen.append(messages)
        reply = next(replies)
        return reply if reply is not None else good(messages, **kwargs)  # type: ignore[arg-type]

    llm.generate = flaky  # type: ignore[method-assign]

    outcome = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert outcome.status == "assembled"
    # The second call carries the broken reply and asks for it corrected.
    assert len(seen) == 2
    assert "could not be validated" in str(seen[1][-1])


def test_unchanged_corpus_serves_the_cached_packet() -> None:
    store = make_store("run make dev to start the service locally")
    llm = make_llm(payload(section("SET_UP", "Run it locally", ["c1"])))

    first = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)
    assert first.provenance is not None
    again = assemble_orientation(
        llm,
        store,
        task_title=TITLE,
        project_ids=PIDS,
        last_fingerprint=first.provenance.corpus_fingerprint,
    )

    assert again.status == "unchanged"
    assert again.packet is None


def test_a_moved_corpus_regenerates_rather_than_serving_stale_guidance() -> None:
    store = make_store("run make dev to start the service locally")
    llm = make_llm(payload(section("SET_UP", "Run it locally", ["c1"])))

    first = assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)
    assert first.provenance is not None

    store.add(
        [
            Chunk(
                id="c9",
                artifact_id="a9",
                filename="CONTRIBUTING.md",
                text="the dev command is now `just dev`, make dev was removed",
                embedding=EMBED,
                project_ids=("p1",),
            )
        ]
    )
    again = assemble_orientation(
        llm,
        store,
        task_title=TITLE,
        project_ids=PIDS,
        last_fingerprint=first.provenance.corpus_fingerprint,
    )

    assert again.status == "assembled"


def test_assembly_is_sampled_deterministically() -> None:
    """Reloading the page must not hand a hire different instructions."""
    store = make_store("run make dev to start the service locally")
    llm = make_llm(payload(section("SET_UP", "Run it locally", ["c1"])))
    temperatures: list[object] = []
    inner = llm.generate

    def recording(messages: list[dict[str, str]], **kwargs: object) -> str:
        temperatures.append(kwargs.get("temperature"))
        return inner(messages)  # type: ignore[arg-type]

    llm.generate = recording  # type: ignore[method-assign]

    assemble_orientation(llm, store, task_title=TITLE, project_ids=PIDS)

    assert temperatures == [0.0]


def test_the_task_and_its_paths_reach_the_prompt() -> None:
    store = make_store("the reports module lives in src/api/reports.py")
    prompts: list[str] = []
    llm = make_llm(payload(section("FIND_THE_CODE", "Where it lives", ["c1"])))
    inner = llm.generate

    def recording(messages: list[dict[str, str]], **kwargs: object) -> str:
        prompts.append(str(messages[1]["content"]))
        return inner(messages)  # type: ignore[arg-type]

    llm.generate = recording  # type: ignore[method-assign]

    assemble_orientation(
        llm,
        store,
        task_title=TITLE,
        project_ids=PIDS,
        task_body="The header is computed once at boot.",
        labels=["good first issue"],
        touched_paths=["src/api/reports.py"],
    )

    assert TITLE in prompts[0]
    assert "good first issue" in prompts[0]
    assert "src/api/reports.py" in prompts[0]


def test_takes_nothing_about_the_hire() -> None:
    """Orientation is a property of the task: two claimants read the same packet."""
    import inspect

    params = set(inspect.signature(assemble_orientation).parameters)

    assert not {p for p in params if "user" in p or "hire" in p}


# --- streaming -----------------------------------------------------------------


def test_stream_emits_per_step_stages_grounded_items_and_a_done() -> None:
    store = make_store(
        "run make dev to start the service locally with the reports cache",
        "pull requests need one review and a green pipeline before merge",
    )
    llm = make_llm(
        payload(
            section("SET_UP", "Run it locally", ["c1"]),
            section("OPEN_THE_PR", "How review works here", ["c2"]),
        )
    )

    events, outcome = collect(
        stream_orientation(llm, store, task_title=TITLE, project_ids=PIDS)
    )

    # One retrieval stage per step of the path to a PR.
    retrieving = [e for e in events if e.get("stage") == "retrieving"]
    assert len(retrieving) == 5
    # An item per grounded section, in step order, each carrying the real section.
    items = [e for e in events if e["type"] == "item"]
    assert [i["item"]["step"] for i in items] == ["SET_UP", "OPEN_THE_PR"]  # type: ignore[index]
    # seq is monotonic across the whole stream.
    assert [e["seq"] for e in events] == list(range(len(events)))
    # The terminal event carries the whole outcome, and it is the returned one.
    assert events[-1]["type"] == "done"
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.status == "assembled"


def test_stream_never_emits_an_ungrounded_section_as_an_item() -> None:
    # A section citing nothing is dropped, so it must never appear as a live item --
    # that is the whole promise of an `item` event.
    store = make_store("run make dev to start the service locally")
    llm = make_llm(
        payload(
            section("SET_UP", "Run it locally", ["c1"]),
            section("MAKE_THE_CHANGE", "Ungrounded advice", []),
        )
    )

    events, _ = collect(
        stream_orientation(llm, store, task_title=TITLE, project_ids=PIDS)
    )

    items = [e for e in events if e["type"] == "item"]
    assert [i["item"]["step"] for i in items] == ["SET_UP"]  # type: ignore[index]


def test_streaming_result_equals_the_non_streaming_packet() -> None:
    # The stream is a view of the same computation: its final packet must be what
    # the plain call returns (provenance timestamps aside).
    store = make_store(
        "run make dev to start the service locally with the reports cache",
        "the reports module lives in src/api/reports.py",
    )
    generated = payload(
        section("SET_UP", "Run it locally", ["c1"]),
        section("FIND_THE_CODE", "Where it lives", ["c2"]),
    )

    _, streamed = collect(
        stream_orientation(
            make_llm(generated), store, task_title=TITLE, project_ids=PIDS
        )
    )
    synchronous: OrientationOutcome = assemble_orientation(
        make_llm(generated), store, task_title=TITLE, project_ids=PIDS
    )

    assert streamed.status == synchronous.status
    assert streamed.model_dump()["packet"] == synchronous.model_dump()["packet"]


def test_an_empty_corpus_streams_a_skipped_done_not_an_error() -> None:
    events, outcome = collect(
        stream_orientation(
            make_llm(payload()), StubVectorStore(), task_title=TITLE, project_ids=PIDS
        )
    )

    assert outcome.status == "skipped"
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["status"] == "skipped"  # type: ignore[index]
    assert "error" not in [e["type"] for e in events]


def test_orientation_scopes_to_requested_projects() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="p1_doc.md",
                text="run make dev for project 1",
                embedding=EMBED,
                project_ids=("p1",),
            ),
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="p2_doc.md",
                text="run make dev for project 2",
                embedding=EMBED,
                project_ids=("p2",),
            ),
        ]
    )
    llm = make_llm(payload(section("SET_UP", "Run it locally", ["c1"])))
    outcome = assemble_orientation(
        llm, store, task_title=TITLE, project_ids=frozenset({"p1"})
    )
    assert outcome.status == "assembled"
    assert outcome.packet is not None
    assert {s.filename for s in outcome.packet.sources} == {"p1_doc.md"}


def test_orientation_multi_project_union() -> None:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="c1",
                artifact_id="a1",
                filename="p1_doc.md",
                text="run make dev for project 1",
                embedding=EMBED,
                project_ids=("p1",),
            ),
            Chunk(
                id="c2",
                artifact_id="a2",
                filename="p2_doc.md",
                text="architecture layout for project 2",
                embedding=EMBED,
                project_ids=("p2",),
            ),
            Chunk(
                id="c3",
                artifact_id="a3",
                filename="p3_doc.md",
                text="pull requests for project 3",
                embedding=EMBED,
                project_ids=("p3",),
            ),
        ]
    )
    llm = make_llm(
        payload(
            section("SET_UP", "Run it locally", ["c1"]),
            section("FIND_THE_CODE", "Where it lives", ["c2"]),
        )
    )
    outcome = assemble_orientation(
        llm, store, task_title=TITLE, project_ids=frozenset({"p1", "p2"})
    )
    assert outcome.status == "assembled"
    assert outcome.packet is not None
    assert {s.filename for s in outcome.packet.sources} == {"p1_doc.md", "p2_doc.md"}
