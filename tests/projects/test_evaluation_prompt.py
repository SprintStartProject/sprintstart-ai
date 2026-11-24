from corpus.types import ScoredChunk
from projects.evaluation import _build_prompt, _coerce_evidence
from tests.support.golden import assert_matches_golden


def _chunk(chunk_id: str, filename: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        id=chunk_id,
        artifact_id=f"art-{chunk_id}",
        filename=filename,
        text=text,
        score=0.9,
    )


def test_industry_prompt_matches_golden() -> None:
    messages = _build_prompt(
        [
            _chunk("c1", "README.md", "Core banking ledger."),
            _chunk("c2", "docs/arch.md", "Payment gateway."),
        ]
    )

    assert_matches_golden("industry_two_chunks", messages)


def test_evidence_keeps_strings_and_numbers_and_drops_blanks_and_other_types() -> None:
    assert _coerce_evidence(["  README.md  ", "", "   ", 3, 1.5, None, {"a": 1}]) == [
        "README.md",
        "3",
        "1.5",
    ]
