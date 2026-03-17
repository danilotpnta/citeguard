"""
Tests for CrossrefVerifier.

Unit tests: mock HTTP, test logic only — no network.
Integration tests (marked slow): hit real Crossref API.

Run unit tests only:
    pytest tests/test_crossref.py -v -m "not slow"

Run all including real API:
    pytest tests/test_crossref.py -v -m slow
"""

import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock

from app.graph.nodes.verification_nodes import classify_references_node
from app.models.schemas import ReferenceResult, VerificationSource, ReferenceList
from app.agents.tools.verifiers.crossref import (
    CrossrefVerifier,
    _normalize_title,
    _extract_lastname,
    _check_author_match,
    _check_retraction,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ref(
    title: str = "Attention is all you need",
    authors: list[str] | None = None,
    year: int | None = 2017,
    doi: str = "10.48550/arXiv.1706.03762",
) -> ReferenceResult:
    return ReferenceResult(
        title=title,
        authors=authors or ["A. Vaswani", "N. Shazeer"],
        year=year,
        doi=doi,
        raw_reference=f"{title} ({year})",
    )


def make_crossref_message(
    title: str = "Attention Is All You Need",
    family_names: list[str] | None = None,
    year: int = 2017,
    retracted: bool = False,
    doi: str = "10.48550/arXiv.1706.03762",
) -> dict:
    message = {
        "title": [title],
        "author": [
            {"family": name} for name in (family_names or ["Vaswani", "Shazeer"])
        ],
        "published": {"date-parts": [[year]]},
        "URL": f"https://doi.org/{doi}",
        "relation": {},
        "update-to": [],
    }
    if retracted:
        message["relation"]["is-retracted-by"] = [{"id": "10.1/retraction"}]
    return message


# ---------------------------------------------------------------------------
# Unit tests — pure logic, no network
# ---------------------------------------------------------------------------


class TestNormalizeTitle:

    def test_lowercases(self):
        assert (
            _normalize_title("Attention Is All You Need") == "attention is all you need"
        )

    def test_strips_punctuation(self):
        assert _normalize_title("BERT: Pre-training") == "bert pre training"

    def test_collapses_whitespace(self):
        result = _normalize_title("Deep   Learning")
        assert "  " not in result


class TestExtractLastname:

    def test_firstname_lastname(self):
        assert _extract_lastname("Ashish Vaswani") == "vaswani"

    def test_initial_lastname(self):
        assert _extract_lastname("A. Vaswani") == "vaswani"

    def test_lastname_initial(self):
        assert _extract_lastname("Vaswani, A.") == "vaswani"

    def test_lastname_only(self):
        assert _extract_lastname("Vaswani") == "vaswani"

    def test_multi_part_lastname(self):
        assert _extract_lastname("G. E. Hinton") == "hinton"


class TestCheckAuthorMatch:

    def test_match_found(self):
        cited = ["A. Vaswani", "N. Shazeer"]
        crossref = [{"family": "Vaswani"}, {"family": "Shazeer"}]
        assert _check_author_match(cited, crossref) is True

    def test_no_match(self):
        cited = ["Y. LeCun"]
        crossref = [{"family": "Vaswani"}, {"family": "Shazeer"}]
        assert _check_author_match(cited, crossref) is False

    def test_partial_match_is_enough(self):
        """Only one author needs to match."""
        cited = ["A. Vaswani", "Y. LeCun"]
        crossref = [{"family": "Vaswani"}]
        assert _check_author_match(cited, crossref) is True

    def test_none_cited_authors(self):
        assert _check_author_match(None, [{"family": "Vaswani"}]) is None

    def test_empty_crossref_authors(self):
        assert _check_author_match(["A. Vaswani"], []) is None

    def test_case_insensitive(self):
        cited = ["vaswani, a."]
        crossref = [{"family": "Vaswani"}]
        assert _check_author_match(cited, crossref) is True


class TestCheckRetraction:

    def test_not_retracted(self):
        message = {"relation": {}, "update-to": []}
        assert _check_retraction(message) is False

    def test_retracted_via_relation(self):
        message = {
            "relation": {"is-retracted-by": [{"id": "10.1/retraction"}]},
            "update-to": [],
        }
        assert _check_retraction(message) is True

    def test_retracted_via_update_to(self):
        message = {
            "relation": {},
            "update-to": [{"type": "retraction", "DOI": "10.1/retraction"}],
        }
        assert _check_retraction(message) is True

    def test_correction_not_retraction(self):
        message = {
            "relation": {},
            "update-to": [{"type": "correction"}],
        }
        assert _check_retraction(message) is False


class TestBuildSourceResult:

    def setup_method(self):
        self.verifier = CrossrefVerifier()

    def test_not_found_returns_found_false(self):
        ref = make_ref()
        result = self.verifier._build_source_result(ref, None)
        assert result.found is False
        assert result.source == VerificationSource.CROSSREF

    def test_high_similarity_match(self):
        ref = make_ref(title="Attention is all you need")
        message = make_crossref_message(title="Attention Is All You Need")
        result = self.verifier._build_source_result(ref, message)
        assert result.found is True
        assert result.title_similarity >= 0.85

    def test_low_similarity_flagged(self):
        ref = make_ref(title="Completely Different Title Here")
        message = make_crossref_message(title="Attention Is All You Need")
        result = self.verifier._build_source_result(ref, message)
        assert result.title_similarity < 0.50

    def test_year_delta_correct(self):
        ref = make_ref(year=2020)
        message = make_crossref_message(year=2017)
        result = self.verifier._build_source_result(ref, message)
        assert result.year_delta == 3

    def test_year_delta_none_when_ref_has_no_year(self):
        ref = make_ref(year=None)
        message = make_crossref_message(year=2017)
        result = self.verifier._build_source_result(ref, message)
        assert result.year_delta is None

    def test_retraction_flag_propagated(self):
        ref = make_ref()
        message = make_crossref_message(retracted=True)
        result = self.verifier._build_source_result(ref, message)
        assert result.retracted is True

    def test_matched_url_populated(self):
        ref = make_ref()
        message = make_crossref_message()
        result = self.verifier._build_source_result(ref, message)
        assert result.matched_url is not None
        assert "doi.org" in result.matched_url


# ---------------------------------------------------------------------------
# Async unit tests — mocked HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestCrossrefVerifierAsync:

    async def test_verify_confirmed_paper(self):
        ref = make_ref()
        message = make_crossref_message()

        verifier = CrossrefVerifier()
        with patch.object(verifier, "_fetch", new=AsyncMock(return_value=message)):
            result = await verifier.verify(ref)

        assert result.reference == ref
        assert VerificationSource.CROSSREF in result.sources_checked
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.author_match is True
        assert sr.year_delta == 0

    async def test_verify_doi_not_found(self):
        ref = make_ref(doi="10.9999/fake.doi")

        verifier = CrossrefVerifier()
        with patch.object(verifier, "_fetch", new=AsyncMock(return_value=None)):
            result = await verifier.verify(ref)

        sr = result.source_results[0]
        assert sr.found is False

    async def test_verify_raises_without_doi(self):
        ref = ReferenceResult(
            title="No DOI Paper",
            raw_reference="No DOI Paper raw",
        )
        verifier = CrossrefVerifier()
        with pytest.raises(ValueError, match="no DOI"):
            await verifier.verify(ref)

    async def test_verify_batch_runs_concurrently(self):
        """Batch of 3 refs — all return confirmed."""
        refs = [make_ref(title=f"Paper {i}", doi=f"10.1/paper{i}") for i in range(3)]
        message = make_crossref_message()

        verifier = CrossrefVerifier()
        with patch.object(verifier, "_fetch", new=AsyncMock(return_value=message)):
            results = await verifier.verify_batch(refs)

        assert len(results) == 3
        assert all(r.source_results[0].found for r in results)

    async def test_fetch_404_returns_none(self):
        verifier = CrossrefVerifier()
        mock_response = MagicMock()
        mock_response.status_code = 404

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        verifier._client = mock_client

        result = await verifier._fetch("10.9999/fake")
        assert result is None

    async def test_fetch_429_returns_none(self):
        verifier = CrossrefVerifier()
        mock_response = MagicMock()
        mock_response.status_code = 429

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        verifier._client = mock_client

        result = await verifier._fetch("10.1/ratelimited")
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests — real Crossref API
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestCrossrefIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verifier = CrossrefVerifier()

    async def test_attention_paper_verified(self):
        """
        10.48550/arXiv.1706.03762 is an arXiv DOI — DataCite, not Crossref.
        Crossref correctly returns not found for these.
        """
        ref = ReferenceResult(
            title="Attention is all you need",
            authors=["A. Vaswani", "N. Shazeer"],
            year=2017,
            doi="10.48550/arXiv.1706.03762",
            raw_reference="Vaswani et al. (2017). Attention is all you need.",
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False  # expected — arXiv DOI

    async def test_fake_doi_not_found(self):
        ref = ReferenceResult(
            title="Fake Paper",
            doi="10.9999/this.doi.does.not.exist.99999",
            raw_reference="Fake Paper raw",
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_nature_deep_learning_paper(self):
        """LeCun et al. 2015 Nature paper."""
        ref = ReferenceResult(
            title="Deep learning",
            authors=["Y. LeCun", "Y. Bengio", "G. Hinton"],
            year=2015,
            doi="10.1038/nature14539",
            raw_reference="LeCun et al. (2015). Deep learning. Nature.",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]

        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.author_match is True
        assert sr.retracted is False
        assert sr.matched_url is not None

    async def test_batch_mixed_real_and_fake(self):
        """Batch with one real DOI and one fake — correct results for each."""
        refs = [
            ReferenceResult(
                title="Deep learning",
                authors=["Y. LeCun"],
                year=2015,
                doi="10.1038/nature14539",
                raw_reference="LeCun et al. (2015).",
            ),
            ReferenceResult(
                title="Quantum Pasta Optimization",
                authors=["A. Smith"],
                year=2023,
                doi="10.9999/fake.99999",
                raw_reference="Smith (2023). Quantum Pasta.",
            ),
        ]
        results = await self.verifier.verify_batch(refs)

        assert results[0].source_results[0].found is True
        assert results[1].source_results[0].found is False

    async def test_arxiv_doi_promoted_to_arxiv_bucket(self):
        """
        A ref with doi=10.48550/arXiv.1706.03762 should go to
        refs_with_arxiv, not refs_with_doi, and get arxiv_id populated.
        """
        ref = make_ref("Attention Paper", doi="10.48550/arXiv.1706.03762")
        state = {"extracted_references": ReferenceList(references=[ref])}
        result = await classify_references_node(state)

        assert len(result["refs_with_doi"]) == 0
        assert len(result["refs_with_arxiv"]) == 1
        assert result["refs_with_arxiv"][0].arxiv_id == "1706.03762"
