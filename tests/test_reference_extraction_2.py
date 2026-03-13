"""
Phase 1 Tests (Set 2) — Real Thesis References

These tests use a real reference list from a thesis, covering:
- "et al." references (most authors hidden)
- References with BOTH a DOI and an arXiv ID in the same entry
- A web/blog reference with no DOI and no arXiv ID
- A book chapter reference
- A malformed author block [89] — known hard case
- An incomplete self-reference [94] with no year and no venue
- arXiv-only preprints

Run with:
    uv run pytest tests/test_reference_extraction_2.py -v -m "not slow"
    uv run pytest tests/test_reference_extraction_2.py -v -m slow
"""

import pytest
import pathlib
from app.models.schemas import ReferenceList

ASSETS = pathlib.Path("tests/assets")
TOTAL_REFERENCES = 16  # [80] through [95]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_md_text():
    path = ASSETS / "sample_references_2.md"
    return path.read_text(encoding="utf-8")


@pytest.fixture
def extracted(sample_md_text):
    """Run extraction once, reuse result across all tests in this session."""
    from app.agents.reference_extractor import extract_references
    return extract_references(sample_md_text)


# ---------------------------------------------------------------------------
# Unit tests — no LLM, no network
# ---------------------------------------------------------------------------

class TestSampleFixture:

    def test_fixture_file_exists(self):
        path = ASSETS / "sample_references_2.md"
        assert path.exists(), "sample_references_2.md not found in tests/assets/"

    def test_fixture_contains_expected_markers(self):
        path = ASSETS / "sample_references_2.md"
        text = path.read_text()
        for marker in ["[80]", "[89]", "[94]", "[95]"]:
            assert marker in text


# ---------------------------------------------------------------------------
# Integration tests — LLM extraction
# ---------------------------------------------------------------------------

@pytest.mark.slow
class TestRealThesisExtraction:

    def test_extracts_all_references(self, extracted):
        """Should find all 16 references [80]–[95]."""
        assert len(extracted.references) == TOTAL_REFERENCES, (
            f"Expected {TOTAL_REFERENCES}, got {len(extracted.references)}"
        )

    def test_raw_reference_never_empty(self, extracted):
        """raw_reference must be populated for every entry."""
        for ref in extracted.references:
            assert ref.raw_reference and len(ref.raw_reference.strip()) > 0

    def test_no_doi_is_a_url(self, extracted):
        """No DOI field should start with http."""
        for ref in extracted.references:
            if ref.doi:
                assert not ref.doi.startswith("http"), (
                    f"DOI stored as URL: {ref.doi} in: {ref.raw_reference[:60]}"
                )

    def test_no_arxiv_id_has_prefix_or_version(self, extracted):
        """arXiv IDs must be bare numeric form."""
        for ref in extracted.references:
            if ref.arxiv_id:
                assert "arXiv" not in ref.arxiv_id
                assert "arxiv" not in ref.arxiv_id
                assert not ref.arxiv_id[-2:] in ("v1", "v2", "v3", "v4"), (
                    f"arXiv ID has version suffix: {ref.arxiv_id}"
                )

    # --- Specific reference checks ---

    def test_ref80_lightrag_arxiv_only(self, extracted):
        """
        [80] LightRAG — arXiv preprint, no DOI registered.
        Should have arxiv_id but doi may be None.
        """
        ref = next(
            (r for r in extracted.references if r.title and "LightRAG" in r.title),
            None,
        )
        assert ref is not None, "LightRAG reference not found"
        assert ref.arxiv_id == "2410.05779"
        assert ref.year == 2024

    def test_ref83_edge_has_both_doi_and_arxiv(self, extracted):
        """
        [83] Graph RAG — has BOTH an arXiv ID and a DOI in the same entry.
        Both fields should be populated independently.
        This is the doi:10.48550/arXiv.xxxx pattern — doi goes in doi,
        arXiv ID goes in arxiv_id.
        """
        ref = next(
            (r for r in extracted.references if r.title and "Graph RAG" in r.title),
            None,
        )
        assert ref is not None, "Graph RAG reference not found"
        assert ref.arxiv_id == "2404.16130", (
            f"Expected arxiv_id='2404.16130', got: {ref.arxiv_id}"
        )
        assert ref.doi is not None, "DOI should be present"
        assert ref.doi.startswith("10."), f"DOI format wrong: {ref.doi}"

    def test_ref84_kggen_has_both_doi_and_arxiv(self, extracted):
        """
        [84] KGGen — same pattern as [83], both identifiers present.
        """
        ref = next(
            (r for r in extracted.references if r.title and "KGGen" in r.title),
            None,
        )
        assert ref is not None, "KGGen reference not found"
        assert ref.arxiv_id == "2502.09956"
        assert ref.doi is not None
        assert ref.doi.startswith("10.")

    def test_ref87_has_doi(self, extracted):
        """[87] Kartawijaya — has a plain non-arXiv DOI."""
        ref = next(
            (r for r in extracted.references if r.title and "Outline Technique" in r.title),
            None,
        )
        assert ref is not None, "Kartawijaya reference not found"
        assert ref.doi == "10.22216/jcc.2018.v3i3.3429"
        assert ref.year == 2018

    def test_ref89_bert_malformed_authors(self, extracted):
        """
        [89] BERT — author block is malformed in the source:
        'Jacob Devlin Ming-Wei Chang Kenton and Lee Kristina Toutanova'
        The LLM should extract as written, not try to fix or reorder.
        We only check that the title and arXiv ID are correct.
        """
        ref = next(
            (r for r in extracted.references if r.title and "BERT" in r.title),
            None,
        )
        assert ref is not None, "BERT reference not found"
        assert ref.arxiv_id == "1810.04805"
        assert ref.year == 2019
        # We do NOT assert authors here — source is genuinely malformed,
        # any reasonable extraction is acceptable

    def test_ref90_openai_blog_no_doi_no_arxiv(self, extracted):
        """
        [90] OpenAI blog post — web resource, no DOI, no arXiv ID.
        Both fields must be None — not hallucinated.
        """
        ref = next(
            (r for r in extracted.references if r.title and "GPT-4o" in r.title),
            None,
        )
        assert ref is not None, "GPT-4o mini reference not found"
        assert ref.doi is None, f"DOI should be None for blog post, got: {ref.doi}"
        assert ref.arxiv_id is None, (
            f"arXiv ID should be None for blog post, got: {ref.arxiv_id}"
        )
        assert ref.url is not None, "URL should be captured for blog post"

    def test_ref92_book_no_doi_no_arxiv(self, extracted):
        """
        [92] Bruno Latour book — no DOI, no arXiv ID.
        Neither should be hallucinated.
        """
        ref = next(
            (r for r in extracted.references if r.title and "Drawing things" in r.title),
            None,
        )
        assert ref is not None, "Latour book not found"
        assert ref.doi is None, f"Book DOI hallucinated: {ref.doi}"
        assert ref.arxiv_id is None, f"Book arXiv ID hallucinated: {ref.arxiv_id}"
        assert ref.year == 1990

    def test_ref94_incomplete_self_reference(self, extracted):
        """
        [94] User's own incomplete reference — no year, no venue, no DOI.
        The LLM should extract what is there and leave the rest None.
        Critical: must NOT hallucinate a year or venue.
        """
        ref = next(
            (r for r in extracted.references if r.title and "CiteGuard" in r.title),
            None,
        )
        assert ref is not None, "CiteGuard self-reference not found"
        assert ref.year is None, f"Year hallucinated: {ref.year}"
        assert ref.doi is None, f"DOI hallucinated: {ref.doi}"
        assert ref.arxiv_id is None, f"arXiv ID hallucinated: {ref.arxiv_id}"

    def test_ref95_holoeval_arxiv(self, extracted):
        """[95] Holoeval — arXiv preprint with year in text."""
        ref = next(
            (r for r in extracted.references if r.title and "Holoeval" in r.title),
            None,
        )
        assert ref is not None, "Holoeval reference not found"
        assert ref.arxiv_id == "2310.14746"
        assert ref.year == 2023