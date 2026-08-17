import json

from ingestion.metadata_store import IngestionMetadataStore
from insights.faq import FaqDocument
from insights.faq_classification import (
    ExistingGroup,
    classify_question,
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


def _classify(
    llm: StubLLMClient,
    question: str = "How do I get VPN access?",
    groups: list[ExistingGroup] | None = None,
    store: StubVectorStore | None = None,
):
    return classify_question(
        question,
        groups or [],
        llm,
        store or StubVectorStore(),
        _metadata_store(),
        project_id=_PROJECT,
    )


def _vpn_group() -> ExistingGroup:
    return ExistingGroup(
        id="g1",
        question="How do I get VPN access?",
        title="Getting VPN access",
        count=3,
    )


# ── classify_question ───────────────────────────────────────────────────────


def test_classify_joins_an_existing_group() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": "g1",
            "title": "Getting VPN access",
            "redacted_question": "Can someone enable VPN for me?",
        }
    )

    result = _classify(
        llm,
        question="Can someone enable VPN for me?",
        groups=[_vpn_group()],
    )

    assert result.relevant
    assert result.group_id == "g1"
    assert result.title == "Getting VPN access"
    assert result.question == "Can someone enable VPN for me?"
    # An existing group already carries its documents; re-retrieving them on
    # every repeat ask is exactly the per-message cost this path avoids.
    assert result.documents == []


def test_classify_opens_a_new_group_with_a_title_and_documents() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "title": "Getting VPN access",
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

    result = _classify(llm, store=store)

    assert result.group_id is None
    assert result.title == "Getting VPN access"
    assert result.documents == [
        FaqDocument(id="doc_001", title="vpn-setup.md", source=None)
    ]


def test_classify_does_not_attach_another_projects_documents() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "title": "Getting VPN access",
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

    assert _classify(llm, store=store).documents == []


def test_classify_discards_smalltalk() -> None:
    llm = _ScriptedLLM({"relevant": False})

    assert not _classify(llm, question="hey there, how you doing").relevant


def test_classify_ignores_blank_input_without_calling_the_model() -> None:
    llm = _ScriptedLLM({"relevant": True})

    result = _classify(llm, question="   ")

    assert not result.relevant
    assert llm.calls == 0


def test_classify_opens_a_new_group_for_a_hallucinated_group_id() -> None:
    """An unknown id must not silently become a join, nor take its title."""
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": "does-not-exist",
            "title": "Starting the backend locally",
            "redacted_question": "How do I start the backend?",
        }
    )

    result = _classify(
        llm,
        question="How do I start the backend?",
        groups=[_vpn_group()],
    )

    assert result.group_id is None
    assert result.title == "Starting the backend locally"


def test_classify_keeps_a_matched_groups_own_title() -> None:
    """A rephrased question must not re-title an established entry."""
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": "g1",
            "title": "Something else entirely",
            "redacted_question": "Can someone enable VPN for me?",
        }
    )

    result = _classify(
        llm,
        question="Can someone enable VPN for me?",
        groups=[_vpn_group()],
    )

    assert result.title == "Getting VPN access"


def test_classify_falls_back_to_the_question_as_the_title() -> None:
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "title": "   ",
            "redacted_question": "How do I get VPN access?",
        }
    )

    assert _classify(llm).title == "How do I get VPN access?"


def test_classify_keeps_the_question_on_unparseable_output() -> None:
    llm = _ScriptedLLM("not json")

    result = _classify(llm)

    assert result.relevant
    assert result.group_id is None
    assert result.question == "How do I get VPN access?"
    assert result.title == "How do I get VPN access?"


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
    assert result.title == "Ask [NAME] for VPN access"


def test_classify_never_returns_the_raw_question_when_redaction_is_missing() -> None:
    """The field is documented as redacted, so a response that omits it must not
    be answered with the original text. It is rejected into the fallback, which
    redacts."""
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "title": "Getting VPN access",
            "redacted_question": "   ",
        }
    )

    result = _classify(llm, question="Mail admin@corp.example for VPN access")

    assert "admin@corp.example" not in result.question
    assert "[EMAIL]" in result.question


def test_classify_redacts_what_the_model_hands_back() -> None:
    """The model is asked to redact and checked for having answered, but it is
    not trusted to have done it — the regex pass runs on its output too."""
    llm = _ScriptedLLM(
        {
            "relevant": True,
            "group_id": None,
            "title": "Getting VPN access",
            "redacted_question": "Mail admin@corp.example for VPN access",
        }
    )

    assert _classify(llm).question == "Mail [EMAIL] for VPN access"


def test_classify_keeps_addresses_out_of_the_prompt_entirely() -> None:
    """No reason to send an address to a model and then ask for it back."""
    captured: list[str] = []

    class _CapturingLLM(_ScriptedLLM):
        def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
            captured.append(str(messages[-1]["content"]))
            return super().generate(messages)

    llm = _CapturingLLM(
        {
            "relevant": True,
            "group_id": None,
            "title": "Getting VPN access",
            "redacted_question": "Mail [EMAIL] for VPN access",
        }
    )

    _classify(llm, question="Mail admin@corp.example for VPN access")

    assert "admin@corp.example" not in captured[0]
    assert "[EMAIL]" in captured[0]


def test_classify_retrieves_documents_for_a_fallback_entry() -> None:
    """A fallback opens a new entry like any other, and only new entries
    retrieve — one that skipped it would stay uncited for good, because every
    later match is treated as a repeat of an entry that already has documents."""
    llm = _ScriptedLLM("not json")
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

    result = _classify(llm, store=store)

    assert result.group_id is None
    assert result.documents == [
        FaqDocument(id="doc_001", title="vpn-setup.md", source=None)
    ]


def test_classify_sends_both_the_title_and_the_wording_of_each_candidate() -> None:
    """A summarised title can lose the component name that distinguishes two
    otherwise identical requests, so the verbatim question travels too."""
    captured: list[str] = []

    class _CapturingLLM(_ScriptedLLM):
        def generate(self, messages: list[dict[str, object]]) -> str:  # type: ignore[override]
            captured.append(str(messages[-1]["content"]))
            return super().generate(messages)

    llm = _CapturingLLM(
        {
            "relevant": True,
            "group_id": "g1",
            "title": "Getting VPN access",
            "redacted_question": "Can someone enable VPN for me?",
        }
    )

    _classify(llm, groups=[_vpn_group()])

    assert "Getting VPN access" in captured[0]
    assert "How do I get VPN access?" in captured[0]


# ── merge_groups ────────────────────────────────────────────────────────────


def _groups(*ids: str) -> list[ExistingGroup]:
    return [
        ExistingGroup(id=gid, question=f"question {gid}", title=f"Title {gid}")
        for gid in ids
    ]


def test_merge_groups_is_a_no_op_below_the_ceiling() -> None:
    llm = _ScriptedLLM({"merges": [{"into": "g1", "sources": ["g2"]}]})

    assert merge_groups(_groups("g1", "g2"), target_max=2, llm=llm) == []
    assert llm.calls == 0


def test_merge_groups_returns_the_models_merge_plan() -> None:
    llm = _ScriptedLLM({"merges": [{"into": "g1", "sources": ["g2"]}]})

    merges = merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm)

    assert [(m.into, m.sources) for m in merges] == [("g1", ["g2"])]


def test_merge_groups_rejects_an_invented_surviving_group() -> None:
    """The survivor keeps the stored samples, title and documents."""
    llm = _ScriptedLLM({"merges": [{"into": "g_new", "sources": ["g1", "g2"]}]})

    assert merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm) == []


def test_merge_groups_never_merges_one_group_into_two_places() -> None:
    llm = _ScriptedLLM(
        {
            "merges": [
                {"into": "g1", "sources": ["g2"]},
                {"into": "g3", "sources": ["g2"]},
            ]
        }
    )

    merges = merge_groups(_groups("g1", "g2", "g3", "g4"), target_max=2, llm=llm)

    assert [(m.into, m.sources) for m in merges] == [("g1", ["g2"])]


def test_merge_groups_drops_a_merge_whose_target_is_already_spoken_for() -> None:
    """Applying both would make the result depend on the order they run in, so
    the first claim on an id wins and the conflicting merge is dropped whole."""
    llm = _ScriptedLLM(
        {
            "merges": [
                {"into": "g2", "sources": ["g3"]},
                {"into": "g1", "sources": ["g2"]},
            ]
        }
    )

    merges = merge_groups(_groups("g1", "g2", "g3", "g4"), target_max=2, llm=llm)

    assert [(m.into, m.sources) for m in merges] == [("g2", ["g3"])]


def test_merge_groups_never_reuses_a_target_across_merges() -> None:
    """An id may take part in at most one merge. Two entries naming the same
    target left the outcome up to how the caller happened to apply them."""
    llm = _ScriptedLLM(
        {
            "merges": [
                {"into": "g1", "sources": ["g2"]},
                {"into": "g1", "sources": ["g3"]},
            ]
        }
    )

    merges = merge_groups(_groups("g1", "g2", "g3", "g4"), target_max=2, llm=llm)

    assert [(m.into, m.sources) for m in merges] == [("g1", ["g2"])]


def test_merge_groups_returns_nothing_when_nothing_is_duplicated() -> None:
    llm = _ScriptedLLM({"merges": []})

    assert merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm) == []


def test_merge_groups_leaves_groups_alone_on_unparseable_output() -> None:
    llm = _ScriptedLLM("not json")

    assert merge_groups(_groups("g1", "g2", "g3"), target_max=2, llm=llm) == []
