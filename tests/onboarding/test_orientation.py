import json
from collections.abc import Generator

from onboarding.orientation import assemble_orientation, stream_orientation
from onboarding.orientation_models import OrientationOutcome
from onboarding.progress import ProgressEvent
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore


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


_EMBED = [1.0] + [0.0] * 767
_TITLE = "Fix the stale cache header on /api/v1/reports"


def _llm(payload: dict[str, object]) -> StubLLMClient:
    llm = StubLLMClient(generate_response=json.dumps(payload))
    llm.embedding = _EMBED
    return llm


def _store(*texts: str) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id=f"a{i}",
                filename=f"doc{i}.md",
                text=text,
                embedding=_EMBED,
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


def _section(
    step: str, title: str, chunk_ids: list[str] | None = None, body: str | None = None
) -> dict[str, object]:
    section: dict[str, object] = {
        "step": step,
        "title": title,
        "body": body if body is not None else f"{title} body",
    }
    if chunk_ids is not None:
        section["chunk_ids"] = chunk_ids
    return section


def _payload(*sections: dict[str, object], **extra: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "summary": "What you need to change the reports cache header.",
        "sections": list(sections),
    }
    payload.update(extra)
    return payload


def test_assembles_a_step_segmented_packet_with_the_sources_it_drew_on() -> None:
    store = _store(
        "run make dev to start the service locally with the reports cache",
        "pull requests need one review and a green pipeline before merge",
    )
    llm = _llm(
        _payload(
            _section("SET_UP", "Run it locally", ["c1"]),
            _section("OPEN_THE_PR", "How review works here", ["c2"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.status == "assembled"
    packet = outcome.packet
    assert packet is not None
    assert [s.step for s in packet.sections] == ["SET_UP", "OPEN_THE_PR"]
    assert packet.task_title == _TITLE
    # The packet states which existing material it stands on.
    assert {s.filename for s in packet.sources} == {"doc1.md", "doc2.md"}


def test_sections_are_ordered_by_step_not_by_the_model() -> None:
    """A hire must never be handed "open the PR" before "find the code"."""
    store = _store("pull request review policy", "the reports module lives in src/api")
    llm = _llm(
        _payload(
            _section("OPEN_THE_PR", "Review", ["c1"]),
            _section("FIND_THE_CODE", "Where it lives", ["c2"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.packet is not None
    assert [s.step for s in outcome.packet.sections] == ["FIND_THE_CODE", "OPEN_THE_PR"]


def test_drops_an_ungrounded_section_but_keeps_the_packet() -> None:
    """A statement with no source does not ship -- that is the whole rule."""
    store = _store("run make dev to start the service locally")
    llm = _llm(
        _payload(
            _section("SET_UP", "Grounded", ["c1"]),
            _section("CHECK_LOCALLY", "Invented", ["nope"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.packet is not None
    assert [s.title for s in outcome.packet.sections] == ["Grounded"]
    assert outcome.sections_dropped == 1


def test_no_section_kind_is_exempt_from_citing() -> None:
    """Unlike a module's TASK/CHECK pages, every packet section is a claim."""
    store = _store("run make dev to start the service locally")
    llm = _llm(
        _payload(
            _section("SET_UP", "Grounded", ["c1"]),
            _section("MAKE_THE_CHANGE", "Uncited", []),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.packet is not None
    assert [s.step for s in outcome.packet.sections] == ["SET_UP"]


def test_drops_a_section_of_a_step_that_does_not_exist() -> None:
    store = _store("run make dev to start the service locally")
    llm = _llm(
        _payload(
            _section("SET_UP", "Grounded", ["c1"]),
            _section("DEPLOY_IT", "Not a step of ours", ["c1"]),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.packet is not None
    assert [s.step for s in outcome.packet.sections] == ["SET_UP"]
    assert outcome.sections_dropped == 1


def test_collapses_sources_that_restate_each_other() -> None:
    """The README and the wiki saying the same thing is load paid for twice."""
    duplicate = "run make dev to start the reports service locally on port 8080"
    store = _store(
        duplicate, duplicate + " .", "pull requests need one approving review"
    )
    llm = _llm(_payload(_section("SET_UP", "Run it locally", ["c1"])))

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.chunks_collapsed == 1
    assert outcome.chunks_retrieved == 2


def test_collapses_two_sections_of_a_step_that_say_the_same_thing() -> None:
    store = _store("run make dev to start the service locally on port 8080")
    body = "Run `make dev` to start the service locally on port 8080."
    llm = _llm(
        _payload(
            _section("SET_UP", "Run it", ["c1"], body=body),
            _section(
                "SET_UP", "Starting the service", ["c1"], body=body + " Then wait."
            ),
        )
    )

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.packet is not None
    assert [s.title for s in outcome.packet.sections] == ["Run it"]
    assert outcome.sections_dropped == 1


def test_skips_when_every_section_is_ungrounded() -> None:
    """An empty state, never a fabricated packet."""
    store = _store("run make dev to start the service locally")
    llm = _llm(_payload(_section("SET_UP", "Invented", ["nope"])))

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.status == "skipped"
    assert outcome.packet is None


def test_empty_corpus_is_skipped_not_an_empty_packet() -> None:
    outcome = assemble_orientation(
        _llm(_payload()), StubVectorStore(), task_title=_TITLE
    )

    assert outcome.status == "skipped"
    assert outcome.packet is None
    assert outcome.notes == ["corpus is empty"]


def test_unparseable_output_is_skipped_not_a_half_packet() -> None:
    store = _store("run make dev to start the service locally")
    llm = StubLLMClient(generate_response="I'm afraid I can't do that.")
    llm.embedding = _EMBED

    outcome = assemble_orientation(llm, store, task_title=_TITLE)

    assert outcome.status == "skipped"
    assert outcome.packet is None


def test_unchanged_corpus_serves_the_cached_packet() -> None:
    store = _store("run make dev to start the service locally")
    llm = _llm(_payload(_section("SET_UP", "Run it locally", ["c1"])))

    first = assemble_orientation(llm, store, task_title=_TITLE)
    assert first.provenance is not None
    again = assemble_orientation(
        llm,
        store,
        task_title=_TITLE,
        last_fingerprint=first.provenance.corpus_fingerprint,
    )

    assert again.status == "unchanged"
    assert again.packet is None


def test_a_moved_corpus_regenerates_rather_than_serving_stale_guidance() -> None:
    store = _store("run make dev to start the service locally")
    llm = _llm(_payload(_section("SET_UP", "Run it locally", ["c1"])))

    first = assemble_orientation(llm, store, task_title=_TITLE)
    assert first.provenance is not None

    store.add(
        [
            Chunk(
                id="c9",
                artifact_id="a9",
                filename="CONTRIBUTING.md",
                text="the dev command is now `just dev`, make dev was removed",
                embedding=_EMBED,
            )
        ]
    )
    again = assemble_orientation(
        llm,
        store,
        task_title=_TITLE,
        last_fingerprint=first.provenance.corpus_fingerprint,
    )

    assert again.status == "assembled"


def test_assembly_is_sampled_deterministically() -> None:
    """Reloading the page must not hand a hire different instructions."""
    store = _store("run make dev to start the service locally")
    llm = _llm(_payload(_section("SET_UP", "Run it locally", ["c1"])))
    temperatures: list[object] = []
    inner = llm.generate

    def recording(messages: list[dict[str, str]], **kwargs: object) -> str:
        temperatures.append(kwargs.get("temperature"))
        return inner(messages)  # type: ignore[arg-type]

    llm.generate = recording  # type: ignore[method-assign]

    assemble_orientation(llm, store, task_title=_TITLE)

    assert temperatures == [0.0]


def test_the_task_and_its_paths_reach_the_prompt() -> None:
    store = _store("the reports module lives in src/api/reports.py")
    prompts: list[str] = []
    llm = _llm(_payload(_section("FIND_THE_CODE", "Where it lives", ["c1"])))
    inner = llm.generate

    def recording(messages: list[dict[str, str]], **kwargs: object) -> str:
        prompts.append(str(messages[1]["content"]))
        return inner(messages)  # type: ignore[arg-type]

    llm.generate = recording  # type: ignore[method-assign]

    assemble_orientation(
        llm,
        store,
        task_title=_TITLE,
        task_body="The header is computed once at boot.",
        labels=["good first issue"],
        touched_paths=["src/api/reports.py"],
    )

    assert _TITLE in prompts[0]
    assert "good first issue" in prompts[0]
    assert "src/api/reports.py" in prompts[0]


def test_takes_nothing_about_the_hire() -> None:
    """Orientation is a property of the task: two claimants read the same packet."""
    import inspect

    params = set(inspect.signature(assemble_orientation).parameters)

    assert not {p for p in params if "user" in p or "hire" in p}


# --- streaming -----------------------------------------------------------------


def test_stream_emits_per_step_stages_grounded_items_and_a_done() -> None:
    store = _store(
        "run make dev to start the service locally with the reports cache",
        "pull requests need one review and a green pipeline before merge",
    )
    llm = _llm(
        _payload(
            _section("SET_UP", "Run it locally", ["c1"]),
            _section("OPEN_THE_PR", "How review works here", ["c2"]),
        )
    )

    events, outcome = _collect(stream_orientation(llm, store, task_title=_TITLE))

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
    store = _store("run make dev to start the service locally")
    llm = _llm(
        _payload(
            _section("SET_UP", "Run it locally", ["c1"]),
            _section("MAKE_THE_CHANGE", "Ungrounded advice", []),
        )
    )

    events, _ = _collect(stream_orientation(llm, store, task_title=_TITLE))

    items = [e for e in events if e["type"] == "item"]
    assert [i["item"]["step"] for i in items] == ["SET_UP"]  # type: ignore[index]


def test_streaming_result_equals_the_non_streaming_packet() -> None:
    # The stream is a view of the same computation: its final packet must be what
    # the plain call returns (provenance timestamps aside).
    store = _store(
        "run make dev to start the service locally with the reports cache",
        "the reports module lives in src/api/reports.py",
    )
    payload = _payload(
        _section("SET_UP", "Run it locally", ["c1"]),
        _section("FIND_THE_CODE", "Where it lives", ["c2"]),
    )

    _, streamed = _collect(stream_orientation(_llm(payload), store, task_title=_TITLE))
    synchronous: OrientationOutcome = assemble_orientation(
        _llm(payload), store, task_title=_TITLE
    )

    assert streamed.status == synchronous.status
    assert streamed.model_dump()["packet"] == synchronous.model_dump()["packet"]


def test_an_empty_corpus_streams_a_skipped_done_not_an_error() -> None:
    events, outcome = _collect(
        stream_orientation(_llm(_payload()), StubVectorStore(), task_title=_TITLE)
    )

    assert outcome.status == "skipped"
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["status"] == "skipped"  # type: ignore[index]
    assert "error" not in [e["type"] for e in events]
