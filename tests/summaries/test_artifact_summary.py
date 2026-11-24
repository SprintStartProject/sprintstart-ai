from corpus.ids import ArtifactId
from corpus.types import Chunk
from summaries.artifact_summary import (
    ArtifactSummaryAgent,
    SummaryCitation,
    SummaryStage,
    SummaryToken,
)
from tests.stubs.llm import StubLLMClient


def _chunk(artifact_id: str, filename: str, source_url: str | None = None) -> Chunk:
    return Chunk(
        id=f"{artifact_id}-{filename}",
        artifact_id=artifact_id,
        filename=filename,
        text=f"text of {filename}",
        embedding=[1.0, 0.0],
        source_url=source_url,
    )


def test_events_are_the_stages_then_the_tokens_then_the_citations() -> None:
    agent = ArtifactSummaryAgent(StubLLMClient(generate_response="the summary"))

    events = list(
        agent.events(ArtifactId("a1"), [_chunk("a1", "b.md", "https://x/b.md")])
    )

    assert events == [
        SummaryStage("notes", "Extracting notes from the source"),
        SummaryStage("summary", "Generating summary"),
        SummaryToken("the summary"),
        SummaryCitation("a1", "b.md", "https://x/b.md"),
    ]


def test_citations_list_the_current_artifact_then_the_previous_one() -> None:
    agent = ArtifactSummaryAgent(StubLLMClient(generate_response="s"))

    events = list(
        agent.events(
            ArtifactId("new"),
            [_chunk("new", "z.md"), _chunk("new", "a.md")],
            previous_chunks=[_chunk("old", "o.md", "https://x/o.md")],
        )
    )

    assert [e for e in events if isinstance(e, SummaryCitation)] == [
        SummaryCitation("new", "a.md", "/api/v1/vector-db/artifacts/new/chunks"),
        SummaryCitation("new", "z.md", "/api/v1/vector-db/artifacts/new/chunks"),
        SummaryCitation("old", "o.md", "https://x/o.md"),
    ]
