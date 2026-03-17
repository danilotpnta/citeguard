"""
Phase 1 Tests — Reference Extraction Schema Validation

These tests verify that the LLM extraction agent:
1. Populates the correct fields from different citation styles
2. Never expands or infers information not in the source text
3. Handles identifiers (DOI, arXiv) in the correct format
4. Stores the original reference string in raw_reference unchanged

Run with:
    uv run pytest tests/test_reference_extraction.py -v

For LLM-dependent tests (marked slow):
    uv run pytest tests/test_reference_extraction.py -v -m slow
"""

import pytest
import pathlib
from app.models.schemas import ReferenceResult, ReferenceList


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

ASSETS = pathlib.Path("tests/assets")


@pytest.fixture
def sample_md_text():
    path = ASSETS / "sample_references.md"
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Unit tests — schema validation (no LLM, no network)
# These test that ReferenceResult behaves correctly as a Pydantic model.
# ---------------------------------------------------------------------------


class TestReferenceResultSchema:

    def test_minimal_valid_reference(self):
        """Only raw_reference is required — all other fields can be None."""
        ref = ReferenceResult(raw_reference="Some reference string.")
        assert ref.raw_reference == "Some reference string."
        assert ref.title is None
        assert ref.authors is None
        assert ref.year is None
        assert ref.doi is None
        assert ref.arxiv_id is None

    def test_full_reference(self):
        """All fields populated correctly."""
        ref = ReferenceResult(
            title="Attention is all you need",
            authors=["A. Vaswani", "N. Shazeer", "N. Parmar"],
            year=2017,
            venue="NeurIPS",
            doi="10.48550/arXiv.1706.03762",
            arxiv_id="1706.03762",
            raw_reference="Vaswani et al. (2017). Attention is all you need. NeurIPS.",
        )
        assert ref.year == 2017
        assert len(ref.authors) == 3
        assert ref.authors[0] == "A. Vaswani"

    def test_authors_preserves_initials(self):
        """Authors with initials must be stored as-is — no expansion."""
        ref = ReferenceResult(
            authors=["A. Krizhevsky", "I. Sutskever", "G. E. Hinton"],
            raw_reference="A. Krizhevsky et al.",
        )
        assert ref.authors[0] == "A. Krizhevsky"
        assert "Alex" not in ref.authors[0]  # no expansion

    def test_doi_is_plain_string_not_url(self):
        """DOI must be the bare identifier, not a full URL."""
        ref = ReferenceResult(
            doi="10.1038/nature14539",
            raw_reference="LeCun et al. (2015). Deep learning. Nature.",
        )
        assert not ref.doi.startswith("http")
        assert ref.doi.startswith("10.")

    def test_arxiv_id_no_prefix_no_version(self):
        """arXiv ID must be bare numeric form only."""
        ref = ReferenceResult(
            arxiv_id="2005.14165",
            raw_reference="Brown et al. (2020). Language models are few-shot learners.",
        )
        assert "arXiv" not in ref.arxiv_id
        assert "v" not in ref.arxiv_id
        assert ref.arxiv_id == "2005.14165"

    def test_year_is_integer(self):
        """Year must be an integer, not a string."""
        ref = ReferenceResult(year=2019, raw_reference="Devlin et al. (2019).")
        assert isinstance(ref.year, int)

    def test_reference_list_empty_by_default(self):
        """ReferenceList initialises with empty list."""
        result = ReferenceList()
        assert result.references == []

    def test_reference_list_holds_multiple(self):
        """ReferenceList correctly stores multiple ReferenceResult objects."""
        refs = ReferenceList(
            references=[
                ReferenceResult(title="Paper A", raw_reference="Paper A ref."),
                ReferenceResult(title="Paper B", raw_reference="Paper B ref."),
            ]
        )
        assert len(refs.references) == 2
        assert refs.references[0].title == "Paper A"


# ---------------------------------------------------------------------------
# Integration tests — LLM extraction (require API keys, marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestLLMExtraction:
    """
    These tests call the real LLM agent. Requires:
      - Valid API key in .env
      - sample_references.md in tests/assets/
    """

    @pytest.fixture(autouse=True)
    def setup(self, sample_md_text):
        from app.agents.reference_extractor import extract_references

        self.extract = extract_references
        self.raw_text = sample_md_text

    def test_extracts_nonzero_references(self):
        """LLM should find at least one reference in the sample file."""
        result = self.extract(self.raw_text)
        assert hasattr(result, "references")
        assert len(result.references) > 0

    def test_extracts_expected_count(self):
        """Sample file has 10 references — LLM should find all of them."""
        result = self.extract(self.raw_text)
        assert len(result.references) == 10

    def test_raw_reference_always_populated(self):
        """raw_reference must never be empty on any extracted result."""
        result = self.extract(self.raw_text)
        for ref in result.references:
            assert ref.raw_reference
            assert len(ref.raw_reference.strip()) > 0

    def test_doi_no_url_prefix(self):
        """No DOI should start with http after extraction."""
        result = self.extract(self.raw_text)
        for ref in result.references:
            if ref.doi:
                assert not ref.doi.startswith(
                    "http"
                ), f"DOI should not be a URL, got: {ref.doi}"

    def test_arxiv_id_clean_format(self):
        """No arXiv ID should contain prefix or version suffix."""
        result = self.extract(self.raw_text)
        for ref in result.references:
            if ref.arxiv_id:
                assert "arXiv" not in ref.arxiv_id
                assert "arxiv" not in ref.arxiv_id
                assert not ref.arxiv_id.endswith(
                    ("v1", "v2", "v3")
                ), f"arXiv ID should not have version suffix, got: {ref.arxiv_id}"

    def test_authors_not_expanded(self):
        """
        Reference [8] uses initials: 'A. Krizhevsky'.
        The LLM must not expand this to 'Alex Krizhevsky'.
        """
        result = self.extract(self.raw_text)
        krizhevsky_ref = next(
            (r for r in result.references if r.title and "ImageNet" in r.title),
            None,
        )
        assert krizhevsky_ref is not None, "ImageNet paper not found in results"
        assert krizhevsky_ref.authors is not None
        first_author = krizhevsky_ref.authors[0]
        assert first_author in (
            "A. Krizhevsky",
            "Krizhevsky",
        ), f"Author initial was expanded, got: {first_author}"

    def test_year_extracted_as_integer(self):
        """All extracted years must be integers."""
        result = self.extract(self.raw_text)
        for ref in result.references:
            if ref.year is not None:
                assert isinstance(
                    ref.year, int
                ), f"Year should be int, got {type(ref.year)} for: {ref.raw_reference}"

    def test_vaswani_attention_paper(self):
        """
        Reference [1] is a well-known paper — verify key fields are correct.
        This also tests that a reference with BOTH a DOI and an arXiv ID
        has each stored in its own field.
        """
        result = self.extract(self.raw_text)
        attention_ref = next(
            (r for r in result.references if r.title and "Attention" in r.title),
            None,
        )
        assert attention_ref is not None, "Attention paper not found"
        assert attention_ref.year == 2017
        # DOI and arXiv ID are separate fields — should not bleed into each other
        if attention_ref.doi:
            assert attention_ref.doi.startswith("10.")
        if attention_ref.arxiv_id:
            assert attention_ref.arxiv_id == "1706.03762"

    def test_book_reference_no_doi_no_arxiv(self):
        """
        Reference [6] is a book with no DOI or arXiv ID.
        Both fields should be None — not hallucinated.
        """
        result = self.extract(self.raw_text)
        book_ref = next(
            (
                r
                for r in result.references
                if r.title and "Deep Learning" in r.title and r.year == 2016
            ),
            None,
        )
        assert book_ref is not None, "Deep Learning book not found"
        assert book_ref.doi is None, f"Book has no DOI but got: {book_ref.doi}"
        assert (
            book_ref.arxiv_id is None
        ), f"Book has no arXiv ID but got: {book_ref.arxiv_id}"

    def test_fake_reference_fields_populated_without_verification(self):
        """
        Reference [10] is a deliberately fake paper with a fake DOI.
        The LLM should still extract all fields as written —
        verification happens in Phase 2, not here.
        """
        result = self.extract(self.raw_text)
        fake_ref = next(
            (r for r in result.references if r.title and "Quantum neural" in r.title),
            None,
        )
        assert fake_ref is not None, "Fake reference not extracted"
        assert fake_ref.doi == "10.1038/s42256-023-99999-x"
        assert fake_ref.year == 2023
