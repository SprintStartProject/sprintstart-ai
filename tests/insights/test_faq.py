import json
from collections.abc import Callable

from ingestion.metadata_store import ArtifactRecord, IngestionMetadataStore
from insights.faq import FaqDocument, FaqQuestionInput, group_faqs
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_PROJECT = "project-1"
_VPN = [1.0, 0.0, 0.0]
_PASSWORD = [0.0, 1.0, 0.0]


def _embed_fn(text: str) -> list[float]:
    if "VPN" in text:
        return _VPN
    if "password" in text.lower():
        return _PASSWORD
    return [0.0, 0.0, 1.0]


class _ScriptedFaqLLM(StubLLMClient):
    """First `generate` call answers grouping; later calls echo redaction.

    ``group_faqs`` makes exactly two ``generate`` calls: one for clustering
    (answered from the scripted ``groups``/``discard_ids``) and one from
    ``redact_pii`` for name redaction (answered as a pass-through so tests can
    focus on clustering/documents).
    """

    def __init__(
        self,
        groups: list[list[str]],
        discard_ids: list[str] | None = None,
        embed_fn: Callable[[str], list[float]] = _embed_fn,
    ) -> None:
        super().__init__(embed_fn=embed_fn)
        self._grouping_response = json.dumps(
            {"groups": groups, "discard_ids": discard_ids or []}
        )
        self._calls = 0

    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        self._calls += 1
        if self._calls == 1:
            return self._grouping_response
        payload = json.loads(messages[-1]["content"])  # type: ignore[index]
        return json.dumps({"texts": payload["texts"]})


def _metadata_store() -> IngestionMetadataStore:
    return IngestionMetadataStore(path=":memory:")


def test_group_faqs_clusters_similar_questions_by_llm_grouping() -> None:
    llm = _ScriptedFaqLLM(groups=[["q1", "q2"], ["q3"]])
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id="q1", text="How do I get VPN access?"),
        FaqQuestionInput(id="q2", text="Can someone enable VPN for me?"),
        FaqQuestionInput(id="q3", text="How do I reset my password?"),
    ]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert [g.count for g in groups] == [2, 1]
    assert groups[0].question == "How do I get VPN access?"
    assert groups[0].questions == [
        "How do I get VPN access?",
        "Can someone enable VPN for me?",
    ]
    assert groups[1].question == "How do I reset my password?"


def test_group_faqs_discards_smalltalk() -> None:
    llm = _ScriptedFaqLLM(groups=[["q1"]], discard_ids=["q2"])
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id="q1", text="How do I get VPN access?"),
        FaqQuestionInput(id="q2", text="hey there, how you doing"),
    ]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert len(groups) == 1
    assert groups[0].question == "How do I get VPN access?"


def test_group_faqs_keeps_distinct_components_in_separate_groups() -> None:
    """Regression: 'start frontend' and 'start backend' must not merge."""
    llm = _ScriptedFaqLLM(groups=[["q1"], ["q2", "q3"]])
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id="q1", text="How to start frontend"),
        FaqQuestionInput(id="q2", text="How to start backend"),
        FaqQuestionInput(id="q3", text="How to start sprintstart-backend"),
    ]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert sorted(g.count for g in groups) == [1, 2]
    frontend_group = next(g for g in groups if g.question == "How to start frontend")
    assert frontend_group.count == 1


def test_group_faqs_falls_back_to_ungrouped_on_unparseable_llm_output() -> None:
    class _BrokenLLM(StubLLMClient):
        def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
            return "not json"

    llm = _BrokenLLM(embed_fn=_embed_fn)
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id="q1", text="How do I get VPN access?"),
        FaqQuestionInput(id="q2", text="How do I reset my password?"),
    ]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert len(groups) == 2
    assert {g.count for g in groups} == {1}


def test_group_faqs_never_drops_an_id_the_model_omits() -> None:
    """Defensive fallback: an id missing from both groups and discard_ids
    still surfaces as its own group rather than silently vanishing."""
    llm = _ScriptedFaqLLM(groups=[["q1"]])
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id="q1", text="How do I get VPN access?"),
        FaqQuestionInput(id="q2", text="How do I reset my password?"),
    ]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert len(groups) == 2


def test_group_faqs_attaches_documents_from_retrieval() -> None:
    llm = _ScriptedFaqLLM(groups=[["q1"]])
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc_001",
                filename="vpn-setup.md",
                text="How to get VPN access set up",
                embedding=_VPN,
                project_ids=(_PROJECT,),
            )
        ]
    )
    metadata_store = _metadata_store()
    metadata_store.save_completed_artifact(
        ArtifactRecord(
            id="doc_001",
            filename="VPN Setup Guide.md",
            content_type="text/markdown",
            source_type="confluence",
            size_bytes=100,
            chunk_count=1,
            status="completed",
            created_at="2026-01-01T00:00:00+00:00",
            updated_at="2026-01-01T00:00:00+00:00",
        )
    )

    questions = [FaqQuestionInput(id="q1", text="How do I get VPN access?")]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert len(groups) == 1
    assert groups[0].documents == [
        FaqDocument(id="doc_001", title="VPN Setup Guide.md", source="confluence")
    ]


def test_group_faqs_does_not_attach_another_projects_documents() -> None:
    llm = _ScriptedFaqLLM(groups=[["q1"]])
    store = StubVectorStore()
    store.add(
        [
            Chunk(
                id="chunk-1",
                artifact_id="doc_001",
                filename="vpn-setup.md",
                text="How to get VPN access set up",
                embedding=_VPN,
                project_ids=("project-2",),
            )
        ]
    )
    metadata_store = _metadata_store()

    questions = [FaqQuestionInput(id="q1", text="How do I get VPN access?")]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert len(groups) == 1
    assert groups[0].documents == []


def test_group_faqs_empty_input_returns_no_groups() -> None:
    llm = _ScriptedFaqLLM(groups=[])
    store = StubVectorStore()
    metadata_store = _metadata_store()

    assert group_faqs([], llm, store, metadata_store, project_id=_PROJECT) == []


def test_group_faqs_caps_sample_size_below_total_count() -> None:
    ids = [f"q{i}" for i in range(7)]
    llm = _ScriptedFaqLLM(groups=[ids])
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [
        FaqQuestionInput(id=qid, text=f"How do I get VPN access, {suffix}?")
        for qid, suffix in zip(ids, ["a", "b", "c", "d", "e", "f", "g"], strict=True)
    ]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert len(groups) == 1
    assert groups[0].count == 7
    assert len(groups[0].questions) <= 5


def test_group_faqs_redacts_names_from_returned_questions() -> None:
    class _RedactingLLM(_ScriptedFaqLLM):
        def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
            self._calls += 1
            if self._calls == 1:
                return self._grouping_response
            payload = json.loads(messages[-1]["content"])  # type: ignore[index]
            redacted = [t.replace("John Doe", "[NAME]") for t in payload["texts"]]
            return json.dumps({"texts": redacted})

    llm = _RedactingLLM(groups=[["q_name"]])
    store = StubVectorStore()
    metadata_store = _metadata_store()

    questions = [FaqQuestionInput(id="q_name", text="Ask John Doe for VPN access")]

    groups = group_faqs(questions, llm, store, metadata_store, project_id=_PROJECT)

    assert len(groups) == 1
    assert groups[0].question == "Ask [NAME] for VPN access"
    assert groups[0].questions == ["Ask [NAME] for VPN access"]
