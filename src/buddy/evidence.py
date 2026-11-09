"""What the Buddy may quote: the local search tools and how their results are shown.

Test and fixture material never reaches the model, and each search result is
formatted under the shared evidence budget."""

from buddy.tools.base import ToolRegistry, ToolResult
from buddy.tools.evidence import format_evidence
from buddy.tools.grep import GrepTool
from buddy.tools.retrieve import RetrieveTool
from corpus.source_filter import SourceExclusions
from corpus.types import RetrievalFilters, ScoredChunk
from llm.base import (
    LLMClient,
    ToolSpec,
)
from rag.prompt import chunk_header
from rag.source_kind import is_test_chunk
from store.base import VectorStore

SEARCH_DOCS = "search_docs"

_SEARCH_TOOL: ToolSpec = {
    "name": SEARCH_DOCS,
    "description": (
        "Search the project's indexed documentation, code, issues and pull requests "
        "for grounded evidence. Use this for any question about how the codebase, "
        "product, or process works."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to look for, phrased as a search query.",
            }
        },
        "required": ["query"],
    },
}

_TOP_K = 5
# The retrieval floor and confidence line: `retrieve` drops anything below it, so an
# empty result means nothing indexed answers with confidence.
_MIN_SCORE = 0.3

GREP = "grep"

# The persona's no-evidence path reacts to this wording; keep it.
_NO_MATCH = "No indexed material matched this search."


def drop_test_material(chunks: list[ScoredChunk]) -> list[ScoredChunk]:
    """Remove test, fixture and sample files from what the mentor may quote.

    Dropped rather than labelled, and the difference matters. A label asks the
    model to hold a distinction while it answers, and this is the exact question
    a model is worst at holding it on: a fixture describing a review process
    *reads* like a review process. Told "this is sample data", it produced a
    branch-naming convention, a CI pipeline and a merge workflow that no team had
    ever used — and defended them when challenged. What a model cannot quote, it
    cannot attribute to the team.

    Emptying the result is a *good* outcome, not a degraded one. The buddy's
    no-evidence path drafts the question to ask a colleague instead of answering,
    which is exactly right here: if the only thing matching was a fixture, the
    thing is not documented, and saying so beats reciting an example as policy.

    The label below stays as the second line, for material that is genuinely
    primary but talks about tests.
    """
    return [
        chunk
        for chunk in chunks
        if not is_test_chunk(chunk.filename, chunk.source_url, chunk.source_role)
    ]


def _chunk_header(chunk: ScoredChunk) -> str:
    # Anything reaching here already survived `drop_test_material`, so this only
    # fires for a file no signal marked -- it is the belt to that braces.
    header = chunk_header(chunk)
    if is_test_chunk(chunk.filename, chunk.source_url, chunk.source_role):
        return (
            f"{header} (test/fixture file -- example or sample data, "
            "not the team's real documentation or process)"
        )
    return header


def without_test_material(result: ToolResult) -> ToolResult:
    """``result`` with test material dropped from its chunks *and* its matches.

    Both, so a dropped fixture is neither quoted nor counted as an omitted match
    ("3 further matches omitted") the model would go looking for.
    """
    kept = drop_test_material(result.chunks)
    if len(kept) == len(result.chunks):
        return result
    match_ids = result.match_chunk_ids
    if match_ids is not None:
        match_ids = frozenset(chunk.id for chunk in kept if chunk.id in match_ids)
    return ToolResult(summary=result.summary, chunks=kept, match_chunk_ids=match_ids)


def format_result(result: ToolResult, chunks: list[ScoredChunk]) -> str:
    return format_evidence(result, chunks, header=_chunk_header, empty=_NO_MATCH)


class _SearchDocsTool(RetrieveTool):
    """Chat's ``retrieve`` under the name the buddy's persona and backend know.

    Reused rather than re-implemented so the buddy gets what chat had: each hit
    arrives with its neighbouring chunks, and only the hits are cited.
    """

    name = SEARCH_DOCS
    description = _SEARCH_TOOL["description"]

    def tool_spec(self) -> ToolSpec:
        return _SEARCH_TOOL


def local_tools(
    llm: LLMClient,
    store: VectorStore,
    exclusions: SourceExclusions,
    filters: RetrievalFilters,
    with_grep: bool,
) -> ToolRegistry:
    search = _SearchDocsTool(
        llm,
        store,
        top_k=_TOP_K,
        min_score=_MIN_SCORE,
        exclusions=exclusions,
        filters=filters,
    )
    if not with_grep:
        return ToolRegistry([search])
    return ToolRegistry(
        [search, GrepTool(store, exclusions=exclusions, filters=filters)]
    )
