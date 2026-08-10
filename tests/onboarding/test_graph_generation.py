import json
import re
from collections.abc import Generator

from llm.base import Message
from onboarding.corpus import corpus_fingerprint
from onboarding.graph_generation import (
    generate_competency_graph,
    stream_competency_graph,
)
from onboarding.graph_models import ActiveCompetency, TombstonedCompetency
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
_OTHER_EMBED = [0.0, 1.0] + [0.0] * 766


class _RecordingLLM(StubLLMClient):
    """Records every prompt, so a test can assert how many passes ran."""

    def __init__(
        self, generate_response: str, embedding: list[float] | None = None
    ) -> None:
        super().__init__(generate_response=generate_response, embedding=embedding)
        self.prompts: list[list[Message]] = []

    def generate(
        self, messages: list[Message], *, temperature: float | None = None
    ) -> str:
        self.prompts.append(messages)
        return super().generate(messages, temperature=temperature)


def _llm(competencies: list[dict[str, object]]) -> _RecordingLLM:
    llm = _RecordingLLM(json.dumps({"competencies": competencies}))
    llm.embedding = _EMBED
    return llm


def _store(*texts: str) -> StubVectorStore:
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id=f"c{i}",
                artifact_id="a1",
                filename=f"doc{i}.kt",
                text=text,
                embedding=_EMBED,
            )
            for i, text in enumerate(texts, start=1)
        ]
    )
    return store


def test_first_time_proposal_drafts_grounded_competencies() -> None:
    store = _store("Kotlin is the primary backend language for this service")
    llm = _llm(
        [
            {
                "key": "kotlin",
                "label": "Kotlin",
                "description": "Primary backend language",
                "kind": "SKILL",
                "repo_ref": "build.gradle.kts",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "proposed"
    assert [c.key for c in outcome.competencies] == ["kotlin"]
    assert outcome.competencies[0].citations[0].chunk_id == "c1"
    assert outcome.provenance is not None
    assert outcome.provenance.corpus_fingerprint == corpus_fingerprint(store)


def test_ungrounded_competency_is_dropped() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {
                "key": "invented",
                "label": "Invented Thing",
                "kind": "SKILL",
                "chunk_ids": ["nonexistent"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "skipped"
    assert outcome.competencies == []


def test_the_vocabulary_is_proposed_in_a_single_pass() -> None:
    """A second LLM call used to draw edges between the proposed nodes. It went
    with the structure it built -- there is no ordering left to derive, so the
    run must not spend a second call finding one."""
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}]
    )

    def _embed(text: str) -> list[float]:
        return _OTHER_EMBED if text == "Our Domain Model" else _EMBED

    llm.embed_fn = _embed
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    generate_competency_graph(llm, store, active_competencies=active)

    assert len(llm.prompts) == 1


def test_the_prompt_never_asks_for_an_ordering() -> None:
    """The vocabulary is flat. Nothing reads a prerequisite any more, and the one
    consumer that used to report ordering could report a non-ordering edge as
    one -- so the model is told not to state an order at all."""
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}]
    )

    generate_competency_graph(llm, store)

    system = str(llm.prompts[0][0]["content"]).lower()
    assert "prerequisite" not in system
    # Word-boundary: "knowledge base" is not an edge.
    assert not re.search(r"\bedges?\b", system)
    assert "do not state or imply" in system


def test_active_competency_is_never_reproposed() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}]
    )
    active = [ActiveCompetency(key="kotlin", label="Kotlin", kind="SKILL")]

    outcome = generate_competency_graph(llm, store, active_competencies=active)

    assert outcome.status == "skipped"
    assert outcome.competencies == []


def test_near_duplicate_competency_is_dropped_via_embedding_similarity() -> None:
    store = _store("Kotlin is the primary backend language and Kotlin lang basics")
    llm = _llm(
        [
            {"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]},
            {
                "key": "kotlin-lang",
                "label": "Kotlin Language",
                "kind": "SKILL",
                "chunk_ids": ["c1"],
            },
        ]
    )
    # Both proposals embed identically (StubLLMClient returns a fixed embedding),
    # so the second must be dropped as a near-duplicate of the first.

    outcome = generate_competency_graph(llm, store)

    assert [c.key for c in outcome.competencies] == ["kotlin"]


def test_key_is_normalized_to_kebab_case() -> None:
    store = _store("Spring Boot powers the web layer")
    llm = _llm(
        [
            {
                "key": "Spring Boot!",
                "label": "Spring Boot",
                "kind": "SKILL",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store)

    assert outcome.competencies[0].key == "spring-boot"


def test_unchanged_corpus_with_matching_fingerprint_is_a_noop() -> None:
    """Competencies are all this derives and they are a function of the corpus,
    so an unchanged corpus short-circuits the whole run -- no retrieval, no LLM
    call. (It used to short-circuit the node pass only, because the edges
    between nodes already in the graph were not a function of the corpus.)"""
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}]
    )

    outcome = generate_competency_graph(
        llm, store, last_fingerprint=corpus_fingerprint(store)
    )

    assert outcome.status == "unchanged"
    assert outcome.competencies == []
    assert llm.prompts == []


def test_empty_corpus_is_skipped() -> None:
    store = StubVectorStore()
    llm = _llm([])

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "skipped"


def test_invalid_llm_json_is_skipped_not_raised() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _RecordingLLM("not json at all")
    llm.embedding = _EMBED

    outcome = generate_competency_graph(llm, store)

    assert outcome.status == "skipped"


def test_reruns_on_the_same_corpus_propose_the_same_vocabulary() -> None:
    """Stability across reruns: an unchanged corpus and deterministic LLM output
    (as a fixed-fixture LLM is, in these tests) reproposes the same vocabulary."""
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [{"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]}]
    )
    active = [
        ActiveCompetency(
            key="our-domain-model", label="Our Domain Model", kind="CONCEPT"
        )
    ]

    first = generate_competency_graph(llm, store, active_competencies=active)
    second = generate_competency_graph(llm, store, active_competencies=active)

    assert first.competencies == second.competencies


# --- streaming -----------------------------------------------------------------


def test_stream_emits_each_competency_as_an_item_and_a_done() -> None:
    store = _store("Kotlin is the primary backend language for our domain model")
    llm = _llm(
        [
            {"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]},
            {
                "key": "our-domain-model",
                "label": "Our Domain Model",
                "kind": "CONCEPT",
                "chunk_ids": ["c1"],
            },
        ]
    )

    # Distinct embeddings so the dedup gate doesn't collapse the two new nodes.
    def _embed(text: str) -> list[float]:
        return _OTHER_EMBED if "Domain" in text else _EMBED

    llm.embed_fn = _embed

    events, outcome = _collect(stream_competency_graph(llm, store))

    items = [e for e in events if e["type"] == "item"]
    assert [e["item"]["label"] for e in items] == [  # type: ignore[index]
        "Kotlin",
        "Our Domain Model",
    ]
    # seq is monotonic across the whole stream.
    assert [e["seq"] for e in events] == list(range(len(events)))
    # The terminal event carries the whole outcome, and it is the returned one.
    assert events[-1]["type"] == "done"
    assert events[-1]["result"] == outcome.model_dump(mode="json")
    assert outcome.status == "proposed"


def test_stream_never_emits_an_ungrounded_competency_as_an_item() -> None:
    # A competency citing nothing is dropped, so it must never appear as a live
    # item -- that is the whole promise of an `item` event.
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {"key": "kotlin", "label": "Kotlin", "kind": "SKILL", "chunk_ids": ["c1"]},
            {
                "key": "invented",
                "label": "Invented",
                "kind": "SKILL",
                "chunk_ids": ["nonexistent"],
            },
        ]
    )

    events, _ = _collect(stream_competency_graph(llm, store))

    labels: list[object] = [
        e["item"]["label"]  # type: ignore[index]
        for e in events
        if e["type"] == "item"
    ]
    assert labels == ["Kotlin"]


def test_streaming_result_equals_the_non_streaming_proposal() -> None:
    # The stream is a view of the same computation: its final outcome must be what
    # the plain call returns (provenance timestamps aside).
    store = _store("Kotlin is the primary backend language for our domain model")

    def _fresh() -> _RecordingLLM:
        return _llm(
            [
                {
                    "key": "kotlin",
                    "label": "Kotlin",
                    "kind": "SKILL",
                    "chunk_ids": ["c1"],
                }
            ]
        )

    _, streamed = _collect(stream_competency_graph(_fresh(), store))
    synchronous = generate_competency_graph(_fresh(), store)

    assert streamed.status == synchronous.status
    assert [c.key for c in streamed.competencies] == [
        c.key for c in synchronous.competencies
    ]


def test_an_empty_corpus_streams_a_skipped_done_not_an_error() -> None:
    events, outcome = _collect(stream_competency_graph(_llm([]), StubVectorStore()))

    assert outcome.status == "skipped"
    assert events[-1]["type"] == "done"
    assert events[-1]["result"]["status"] == "skipped"  # type: ignore[index]
    assert "error" not in [e["type"] for e in events]


def test_area_snaps_onto_the_spelling_already_in_use() -> None:
    """Grouping only groups if two words for one subject land in one area.

    The prompt asks the model to reuse an existing area, but asking is not
    enforcing -- so the value is resolved against what exists on the way out too.
    """
    store = _store("JWTs are validated on every request")
    llm = _llm(
        [
            {
                "key": "jwt-validation",
                "label": "JWT validation",
                "kind": "SKILL",
                "area": "  authentication ",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(
        llm, store, existing_areas=["Authentication", "Ingestion"]
    )

    assert [c.area for c in outcome.competencies] == ["Authentication"]


def test_a_genuinely_new_area_is_kept() -> None:
    store = _store("Chunks are embedded and written to the vector store")
    llm = _llm(
        [
            {
                "key": "chunking",
                "label": "Chunking",
                "kind": "CONCEPT",
                "area": " Ingestion ",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store, existing_areas=["Authentication"])

    assert [c.area for c in outcome.competencies] == ["Ingestion"]


def test_an_unplaceable_competency_keeps_a_null_area() -> None:
    """A wrong grouping is worse than none, so "could not place it" stays empty."""
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {
                "key": "kotlin",
                "label": "Kotlin",
                "kind": "SKILL",
                "area": "   ",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(llm, store, existing_areas=["Authentication"])

    assert outcome.competencies[0].area is None


def test_existing_areas_reach_the_prompt_and_ordering_still_does_not() -> None:
    store = _store("JWTs are validated on every request")
    llm = _llm(
        [
            {
                "key": "jwt-validation",
                "label": "JWT validation",
                "kind": "SKILL",
                "area": "Authentication",
                "chunk_ids": ["c1"],
            }
        ]
    )

    generate_competency_graph(llm, store, existing_areas=["Authentication"])

    system = llm.prompts[0][0]["content"]
    assert "Areas already in use:" in system
    assert "- Authentication" in system
    # An area groups; it must never be sold to the model as a sequence.
    assert "prerequisite" not in system.lower()
    assert "before what" in system  # the rule forbidding ordering survives


def test_a_tombstoned_key_is_never_re_proposed() -> None:
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {
                "key": "kotlin",
                "label": "Kotlin",
                "kind": "SKILL",
                "chunk_ids": ["c1"],
            }
        ]
    )

    outcome = generate_competency_graph(
        llm,
        store,
        tombstoned_competencies=[TombstonedCompetency(key="kotlin", label="Kotlin")],
    )

    assert outcome.competencies == []


def test_a_tombstoned_competency_cannot_return_under_a_rephrasing() -> None:
    """The case the whole mechanism exists for.

    Blocking the exact key alone leaks: the generator re-proposes the same thing
    under a new key next crawl, a PM deletes it again, and that repeats forever.
    So the tombstone's label is embedded and the similarity gate catches it.
    """
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {
                "key": "kotlin-language",  # a different key
                "label": "The Kotlin programming language",
                "kind": "SKILL",
                "chunk_ids": ["c1"],
            }
        ]
    )
    # Same embedding for the tombstone's label and the rephrased proposal, which is
    # what "means the same thing" looks like to the similarity gate.
    llm.embedding = _EMBED

    outcome = generate_competency_graph(
        llm,
        store,
        tombstoned_competencies=[TombstonedCompetency(key="kotlin", label="Kotlin")],
    )

    assert outcome.competencies == []


def test_a_tombstone_does_not_block_an_unrelated_competency() -> None:
    """Stickiness must not become a freeze on everything near what was removed."""
    store = _store("Chunks are embedded and written to the vector store")
    llm = _RecordingLLM(
        json.dumps(
            {
                "competencies": [
                    {
                        "key": "chunking",
                        "label": "Chunking",
                        "kind": "CONCEPT",
                        "chunk_ids": ["c1"],
                    }
                ]
            }
        )
    )
    # Only the tombstone embeds differently, i.e. it is about something else. The
    # query goes through this too, so everything else must keep the chunk embedding
    # or retrieval finds nothing and the run skips before proposing anything.
    llm.embed_fn = lambda text: _OTHER_EMBED if "Kotlin" in text else _EMBED

    outcome = generate_competency_graph(
        llm,
        store,
        tombstoned_competencies=[TombstonedCompetency(key="kotlin", label="Kotlin")],
    )

    assert [c.key for c in outcome.competencies] == ["chunking"]


def test_removed_competencies_are_named_in_the_prompt_as_rejected() -> None:
    """A tombstone the generator never sees is not a tombstone."""
    store = _store("Kotlin is the primary backend language")
    llm = _llm(
        [
            {
                "key": "chunking",
                "label": "Chunking",
                "kind": "CONCEPT",
                "chunk_ids": ["c1"],
            }
        ]
    )
    llm.embed_fn = lambda text: _OTHER_EMBED if "Kotlin" in text else _EMBED

    generate_competency_graph(
        llm,
        store,
        tombstoned_competencies=[TombstonedCompetency(key="kotlin", label="Kotlin")],
    )

    system = llm.prompts[0][0]["content"]
    assert "Removed competencies:" in system
    assert "- kotlin: Kotlin" in system
    # Rejected, not merely absent -- the model must not read this as a gap to fill.
    assert "they were rejected" in system
