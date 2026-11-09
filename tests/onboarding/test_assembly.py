"""The stages the grounded assembly pipelines share.

The pipelines' contract suites drive these through each pipeline; this module
pins the pieces whose behaviour no single pipeline shows on its own.
"""

from onboarding.assembly import collapse_duplicates, evidence_line
from rag.types import ScoredChunk


def _scored(
    chunk_id: str,
    text: str,
    score: float,
    artifact_type: str | None = None,
    language: str | None = None,
) -> ScoredChunk:
    return ScoredChunk(
        id=chunk_id,
        artifact_id=f"a-{chunk_id}",
        filename=f"{chunk_id}.md",
        text=text,
        score=score,
        artifact_type=artifact_type,
        language=language,
    )


def test_duplicates_collapse_into_the_best_scoring_copy() -> None:
    text = "run make dev to start the reports service locally on port 8080"
    chunks = [
        _scored("weak", text, 0.4),
        _scored("best", text, 0.9),
        _scored("middle", text + " .", 0.6),
        _scored("other", "pull requests need one approving review", 0.5),
    ]

    kept, collapsed = collapse_duplicates(chunks)

    assert [c.id for c in kept] == ["best", "other"]
    assert collapsed == 2


def test_an_evidence_line_names_its_file_and_kind() -> None:
    plain = _scored("c1", "run make dev", 0.9)
    code = _scored("c2", "fun main()", 0.9, artifact_type="CODE", language="kotlin")

    assert evidence_line(plain) == "  [c1] (c1.md | FILE) run make dev"
    assert evidence_line(code) == "  [c2] (c2.md | CODE/kotlin) fun main()"
