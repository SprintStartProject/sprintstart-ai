import json

from ingestion.metadata_store import IngestionMetadataStore
from insights.faq import UNCATEGORIZED, FaqDocument
from insights.faq_classification import (
    ExistingCategory,
    ExistingGroup,
    classify_question,
    consolidate_categories,
    merge_groups,
)
from rag.types import Chunk
from tests.stubs.llm import StubLLMClient
from tests.stubs.store import StubVectorStore

_PROJECT = "project-1"
_VPN = [1.0, 0.0, 0.0]


def _embed_fn(text: str) -> list[float]:
    if "VPN" in text:
        return _VPN
    return [0.0, 0.0, 1.0]


class _ScriptedLLM(StubLLMClient):
    """Answers the first `generate` from a script; later calls echo redaction.

    ``classify_question`` makes exactly one ``generate`` call on the happy
    path. Its fallback path makes a second one through ``redact_pii``, which
    this answers as a pass-through so tests can tell the two paths apart.
    """

    def __init__(self, payload: object) -> None:
        super().__init__(embed_fn=_embed_fn)
        self._payload = payload if isinstance(payload, str) else json.dumps(payload)
        self.calls = 0

    def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
        self.calls += 1
        if self.calls == 1:
            return self._payload
        content = messages[-1]["content"]
        return json.dumps({"texts": json.loads(str(content))["texts"]})


def _metadata_store() -> IngestionMetadataStore:
    return IngestionMetadataStore(path=":memory:")


def _classify(llm: StubLLMClient, **kwargs: object):
    store = kwargs.pop("store", None) or StubVectorStore()
    return classify_question(
        str(kwargs.pop("question", "How do I get VPN access?")),
        list(kwargs.pop("categories", []) or []),  # type: ignore[arg-type]
        list(kwargs.pop("groups", []) or []),  # type: ignore[arg-type]
        llm,
        store,  # type: ignore[arg-type]
        _metadata_store(),
        project_id=_PROJECT,
    )


# ── classify_question ───────────────────────────────────────────────────────


def test_classify_joins_an_existing_group() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": "g1",
            "category": "Access & Accounts",
            "redacted_question": "Can someone enable VPN for me?",
        }
    )

    result = _classify(
        llm,
        question="Can someone enable VPN for me?",
        categories=[ExistingCategory(name="Access & Accounts", group_count=1)],
        groups=[
            ExistingGroup(
                id="g1",
                question="How do I get VPN access?",
                category="Access & Accounts",
                count=3,
            )
        ],
    )

    assert result.relevant
    assert result.group_id == "g1"
    assert result.category == "Access & Accounts"
    assert result.question == "Can someone enable VPN for me?"
    # An existing group already carries its documents; re-retrieving them on
    # every repeat ask is exactly the per-message cost this path avoids.
    assert result.documents == []


def test_classify_opens_a_new_group_with_its_answering_documents() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "category": "Access & Accounts",
            "redacted_question": "How do I get VPN access?",
        }
    )
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

    result = classify_question(
        "How do I get VPN access?",
        [],
        [],
        llm,
        store,
        _metadata_store(),
        project_id=_PROJECT,
    )

    assert result.group_id is None
    assert result.documents == [
        FaqDocument(id="doc_001", title="vpn-setup.md", source=None)
    ]


def test_classify_does_not_attach_another_projects_documents() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "category": "Access & Accounts",
            "redacted_question": "How do I get VPN access?",
        }
    )
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

    result = classify_question(
        "How do I get VPN access?",
        [],
        [],
        llm,
        store,
        _metadata_store(),
        project_id=_PROJECT,
    )

    assert result.documents == []


def test_classify_discards_smalltalk() -> None:
    llm = _ScriptedLLM({"relevant": False})

    result = _classify(llm, question="hey there, how you doing")

    assert not result.relevant


def test_classify_ignores_blank_input_without_calling_the_model() -> None:
    llm = _ScriptedLLM({"relevant": True})

    result = _classify(llm, question="   ")

    assert not result.relevant
    assert llm.calls == 0


def test_classify_opens_a_new_group_for_a_hallucinated_group_id() -> None:
    """An unknown id must not silently become a join, nor take its category."""
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": "does-not-exist",
            "category": "Local Setup",
            "redacted_question": "How do I start the backend?",
        }
    )

    result = _classify(
        llm,
        question="How do I start the backend?",
        groups=[ExistingGroup(id="g1", question="How do I get VPN access?", count=2)],
    )

    assert result.group_id is None
    assert result.category == "Local Setup"


def test_classify_reuses_the_existing_spelling_of_a_category() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "category": "local setup",
            "redacted_question": "How do I start the backend?",
        }
    )

    result = _classify(
        llm,
        question="How do I start the backend?",
        categories=[ExistingCategory(name="Local Setup", group_count=2)],
    )

    assert result.category == "Local Setup"


def test_classify_keeps_a_matched_groups_own_category() -> None:
    """A rephrased question must not drag an established group elsewhere."""
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": "g1",
            "category": "Something Else",
            "redacted_question": "Can someone enable VPN for me?",
        }
    )

    result = _classify(
        llm,
        question="Can someone enable VPN for me?",
        categories=[ExistingCategory(name="Access & Accounts", group_count=1)],
        groups=[
            ExistingGroup(
                id="g1",
                question="How do I get VPN access?",
                category="Access & Accounts",
                count=3,
            )
        ],
    )

    assert result.category == "Access & Accounts"


def test_classify_keeps_the_question_uncategorized_on_unparseable_output() -> None:
    llm = _ScriptedLLM("not json")

    result = _classify(llm, question="How do I get VPN access?")

    assert result.relevant
    assert result.category == UNCATEGORIZED
    assert result.group_id is None
    assert result.question == "How do I get VPN access?"


def test_classify_still_redacts_when_classification_is_unparseable() -> None:
    class _RedactingLLM(_ScriptedLLM):
        def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
            self.calls += 1
            if self.calls == 1:
                return "not json"
            texts = json.loads(str(messages[-1]["content"]))["texts"]
            return json.dumps(
                {"texts": [t.replace("John Doe", "[NAME]") for t in texts]}
            )

    llm = _RedactingLLM("")

    result = _classify(llm, question="Ask John Doe for VPN access")

    assert result.question == "Ask [NAME] for VPN access"


def test_classify_keeps_the_original_text_when_redaction_comes_back_empty() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "category": "Access & Accounts",
            "redacted_question": "   ",
        }
    )

    result = _classify(llm, question="How do I get VPN access?")

    assert result.question == "How do I get VPN access?"


# ── consolidate_categories ──────────────────────────────────────────────────


def _categories(*names: str) -> list[ExistingCategory]:
    return [ExistingCategory(name=name, group_count=1) for name in names]


def test_consolidate_is_a_no_op_below_the_ceiling() -> None:
    llm = _ScriptedLLM({"merges": [{"into": "A", "sources": ["B"]}]})

    assert consolidate_categories(_categories("A", "B"), target_max=2, llm=llm) == []
    assert llm.calls == 0


def test_consolidate_returns_the_models_merge_plan() -> None:
    llm = _ScriptedLLM(
        {"merges": [{"into": "Local Setup", "sources": ["Backend Setup"]}]}
    )

    merges = consolidate_categories(
        _categories("Local Setup", "Backend Setup", "Testing"),
        target_max=2,
        llm=llm,
    )

    assert [(m.into, m.sources) for m in merges] == [("Local Setup", ["Backend Setup"])]


def test_consolidate_accepts_an_invented_umbrella_category() -> None:
    llm = _ScriptedLLM(
        {"merges": [{"into": "Local Setup", "sources": ["Backend", "Frontend"]}]}
    )

    merges = consolidate_categories(
        _categories("Backend", "Frontend", "Testing"), target_max=2, llm=llm
    )

    assert [(m.into, m.sources) for m in merges] == [
        ("Local Setup", ["Backend", "Frontend"])
    ]


def test_consolidate_drops_unknown_source_categories() -> None:
    llm = _ScriptedLLM(
        {"merges": [{"into": "Local Setup", "sources": ["Backend", "Invented"]}]}
    )

    merges = consolidate_categories(
        _categories("Local Setup", "Backend", "Testing"), target_max=2, llm=llm
    )

    assert [(m.into, m.sources) for m in merges] == [("Local Setup", ["Backend"])]


def test_consolidate_never_merges_one_category_into_two_places() -> None:
    llm = _ScriptedLLM(
        {
            "merges": [
                {"into": "Local Setup", "sources": ["Backend"]},
                {"into": "Testing", "sources": ["Backend"]},
            ]
        }
    )

    merges = consolidate_categories(
        _categories("Local Setup", "Backend", "Testing", "Deployment"),
        target_max=2,
        llm=llm,
    )

    assert [(m.into, m.sources) for m in merges] == [("Local Setup", ["Backend"])]


def test_consolidate_drops_a_merge_whose_target_is_merged_away_later() -> None:
    """Applying both would make the result depend on the order they run in."""
    llm = _ScriptedLLM(
        {
            "merges": [
                {"into": "Backend", "sources": ["Testing"]},
                {"into": "Local Setup", "sources": ["Backend"]},
            ]
        }
    )

    merges = consolidate_categories(
        _categories("Local Setup", "Backend", "Testing", "Deployment"),
        target_max=2,
        llm=llm,
    )

    assert [(m.into, m.sources) for m in merges] == [("Local Setup", ["Backend"])]


def test_consolidate_leaves_categories_alone_on_unparseable_output() -> None:
    llm = _ScriptedLLM("not json")

    assert consolidate_categories(_categories("A", "B"), target_max=1, llm=llm) == []


# ── merge_groups ────────────────────────────────────────────────────────────


def _groups(*ids: str) -> list[ExistingGroup]:
    return [ExistingGroup(id=gid, question=f"question {gid}") for gid in ids]


def test_merge_groups_is_a_no_op_below_the_ceiling() -> None:
    llm = _ScriptedLLM({"merges": [{"into": "g1", "sources": ["g2"]}]})

    assert merge_groups(_groups("g1", "g2"), target_max=2, llm=llm) == []
    assert llm.calls == 0


def test_merge_groups_returns_the_models_merge_plan() -> None:
    llm = _ScriptedLLM({"merges": [{"into": "g1", "sources": ["g2"]}]})

    merges = merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm)

    assert [(m.into, m.sources) for m in merges] == [("g1", ["g2"])]


def test_merge_groups_rejects_an_invented_surviving_group() -> None:
    """The survivor keeps the stored samples and documents, so it must exist."""
    llm = _ScriptedLLM({"merges": [{"into": "g_new", "sources": ["g1", "g2"]}]})

    assert merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm) == []


def test_merge_groups_returns_nothing_when_nothing_is_duplicated() -> None:
    llm = _ScriptedLLM({"merges": []})

    assert merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm) == []


def test_merge_groups_leaves_groups_alone_on_unparseable_output() -> None:
    llm = _ScriptedLLM("not json")

    assert merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm) == []
