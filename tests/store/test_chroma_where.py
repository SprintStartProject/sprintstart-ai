from corpus.types import RetrievalFilters
from store.chroma_where import where_filter_for_chroma


def test_where_filter_for_chroma_uses_the_project_marker_key() -> None:
    assert where_filter_for_chroma(RetrievalFilters(project_id="project-1")) == {
        "project:project-1": {"$eq": True}
    }


def test_where_filter_for_chroma_ands_project_with_other_conditions() -> None:
    where = where_filter_for_chroma(
        RetrievalFilters(project_id="project-1", source_systems=["GITHUB"])
    )

    assert where == {
        "$and": [
            {"project:project-1": {"$eq": True}},
            {"source_system": {"$in": ["GITHUB"]}},
        ]
    }
