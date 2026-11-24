from corpus.types import ScoredChunk
from skills.models import SkillCatalogEntry
from skills.suggestion import _build_prompt, _format_catalog, _format_chunks
from tests.support.golden import assert_matches_golden


def _chunk(chunk_id: str, filename: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        id=chunk_id,
        artifact_id=f"art-{chunk_id}",
        filename=filename,
        text=text,
        score=0.9,
    )


def test_skill_prompt_with_catalog_evidence_and_industry_matches_golden() -> None:
    messages = _build_prompt(
        "Frontend Developer",
        "Builds user interfaces",
        "Healthcare",
        [
            SkillCatalogEntry(
                id="s1", name="React", category="Frontend", universal=False
            ),
            SkillCatalogEntry(id="s2", name="Scrum", category=None, universal=True),
        ],
        [_chunk("c1", "App.tsx", "import React from 'react'")],
    )

    assert_matches_golden("skills_full", messages)


def test_skill_prompt_without_optional_context_matches_golden() -> None:
    messages = _build_prompt("Tester", "", None, [], [])

    assert_matches_golden("skills_bare", messages)


def test_catalog_lists_id_name_category_and_universal_marker() -> None:
    catalog = [
        SkillCatalogEntry(id="a", name="Docker", category="DevOps"),
        SkillCatalogEntry(id="b", name="Teamwork", universal=True),
        SkillCatalogEntry(id="c", name="Go", category="Backend", universal=True),
    ]

    assert _format_catalog(catalog) == (
        "- [a] Docker [DevOps]\n"
        "- [b] Teamwork (universal)\n"
        "- [c] Go [Backend] (universal)"
    )


def test_an_empty_catalog_says_so() -> None:
    assert _format_catalog([]) == "(empty catalog)"


def test_the_catalog_stops_at_200_entries() -> None:
    catalog = [SkillCatalogEntry(id=str(i), name="x") for i in range(250)]

    assert len(_format_catalog(catalog).splitlines()) == 200


def test_the_catalog_is_truncated_with_a_marker_when_it_outgrows_its_budget() -> None:
    catalog = [SkillCatalogEntry(id="i", name="n" * 100) for _ in range(100)]

    lines = _format_catalog(catalog).splitlines()

    assert lines[-1] == "- ... (remaining catalog truncated)"
    assert sum(len(line) + 1 for line in lines[:-1]) <= 6000
    assert 55 < len(lines) < 100


def test_evidence_chunks_show_id_filename_and_clipped_text() -> None:
    chunks = [_chunk(f"c{i}", f"f{i}.py", "x" * 1500) for i in range(15)]

    lines = _format_chunks(chunks).splitlines()

    assert len(lines) == 12
    assert lines[0] == "[c0] (f0.py) " + "x" * 1000


def test_no_retrieved_evidence_says_so() -> None:
    assert _format_chunks([]) == "(no project artifacts retrieved)"
