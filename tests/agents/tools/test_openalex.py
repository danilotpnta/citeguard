"""
Tests for OpenAlexVerifier.

Unit tests:  mock HTTP, no network.
Integration: real OpenAlex API (marked slow) — no key needed, always runs.

Run unit tests:
    pytest tests/agents/tools/test_openalex.py -v -m "not slow"

Run all:
    pytest tests/agents/tools/test_openalex.py -v -m slow
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.models.schemas import ReferenceResult, VerificationSource
from app.agents.tools.verifiers.openalex import (
    OpenAlexVerifier,
    _normalize_title,
    _extract_lastname,
    _check_author_match,
    _best_match,
    _extract_url,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ref(
    title: str = "Attention is all you need",
    authors: list[str] | None = None,
    year: int | None = 2017,
) -> ReferenceResult:
    return ReferenceResult(
        title=title,
        authors=authors or ["A. Vaswani", "N. Shazeer"],
        year=year,
        raw_reference=f"{title} ({year})",
    )


def make_oa_candidate(
    title: str = "Attention Is All You Need",
    author_names: list[str] | None = None,
    year: int = 2017,
    doi: str | None = "https://doi.org/10.48550/arXiv.1706.03762",
    oa_url: str | None = None,
) -> dict:
    return {
        "title": title,
        "authorships": [
            {"author": {"display_name": name}}
            for name in (author_names or ["Ashish Vaswani", "Noam Shazeer"])
        ],
        "publication_year": year,
        "doi": doi,
        "open_access": {"oa_url": oa_url},
        "primary_location": {"landing_page_url": None},
    }


# ---------------------------------------------------------------------------
# Unit tests — pure logic
# ---------------------------------------------------------------------------


class TestBestMatchOA:

    def test_returns_none_for_empty_candidates(self):
        assert _best_match(make_ref(), []) is None

    def test_returns_none_when_no_title(self):
        ref = ReferenceResult(title=None, raw_reference="raw")
        assert _best_match(ref, [make_oa_candidate()]) is None

    def test_returns_best_above_threshold(self):
        ref = make_ref(title="Attention is all you need")
        result = _best_match(ref, [make_oa_candidate()])
        assert result is not None
        assert result["_title_similarity"] >= 0.85

    def test_returns_none_below_threshold(self):
        ref = make_ref(title="Completely Different Title Here")
        assert _best_match(ref, [make_oa_candidate()]) is None

    def test_picks_best_among_multiple(self):
        ref = make_ref(title="Attention is all you need")
        candidates = [
            make_oa_candidate(title="Something Unrelated"),
            make_oa_candidate(title="Attention Is All You Need"),
        ]
        result = _best_match(ref, candidates)
        assert result["title"] == "Attention Is All You Need"


class TestExtractUrlOA:

    def test_prefers_oa_url(self):
        candidate = make_oa_candidate(oa_url="https://arxiv.org/pdf/1706.03762")
        assert _extract_url(candidate) == "https://arxiv.org/pdf/1706.03762"

    def test_falls_back_to_doi(self):
        candidate = make_oa_candidate(
            oa_url=None, doi="https://doi.org/10.1038/nature14539"
        )
        assert _extract_url(candidate) == "https://doi.org/10.1038/nature14539"

    def test_returns_none_when_nothing(self):
        candidate = {
            "doi": None,
            "open_access": {"oa_url": None},
            "primary_location": {"landing_page_url": None},
        }
        assert _extract_url(candidate) is None


class TestCheckAuthorMatchOA:

    def test_match_found(self):
        cited = ["A. Vaswani"]
        authorships = [
            {"author": {"display_name": "Ashish Vaswani"}},
            {"author": {"display_name": "Noam Shazeer"}},
        ]
        assert _check_author_match(cited, authorships) is True

    def test_no_match(self):
        cited = ["Y. LeCun"]
        authorships = [{"author": {"display_name": "Ashish Vaswani"}}]
        assert _check_author_match(cited, authorships) is False

    def test_et_al_returns_none(self):
        cited = ["Zirui Guo et al."]
        authorships = [{"author": {"display_name": "Zirui Guo"}}]
        assert _check_author_match(cited, authorships) is None

    def test_none_cited_returns_none(self):
        assert (
            _check_author_match(None, [{"author": {"display_name": "Vaswani"}}]) is None
        )

    def test_empty_authorships_returns_none(self):
        assert _check_author_match(["A. Vaswani"], []) is None


class TestBuildSourceResultOA:

    def setup_method(self):
        self.verifier = OpenAlexVerifier()

    def test_not_found(self):
        result = self.verifier._build_source_result(make_ref(), None)
        assert result.found is False
        assert result.source == VerificationSource.OPENALEX

    def test_found_populates_fields(self):
        ref = make_ref(year=2017)
        candidate = make_oa_candidate(year=2017)
        candidate["_title_similarity"] = 0.97
        result = self.verifier._build_source_result(ref, candidate)

        assert result.found is True
        assert result.title_similarity == 0.97
        assert result.year_delta == 0
        assert result.author_match is True

    def test_year_delta_computed(self):
        ref = make_ref(year=2020)
        candidate = make_oa_candidate(year=2017)
        candidate["_title_similarity"] = 0.95
        result = self.verifier._build_source_result(ref, candidate)
        assert result.year_delta == 3


# ---------------------------------------------------------------------------
# Async unit tests — mocked HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestOpenAlexAsync:

    async def test_verify_confirmed_paper(self):
        ref = make_ref()
        verifier = OpenAlexVerifier()
        with patch.object(
            verifier, "_search", new=AsyncMock(return_value=[make_oa_candidate()])
        ):
            result = await verifier.verify(ref)

        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.author_match is True

    async def test_verify_no_candidates(self):
        ref = make_ref(title="Quantum Pasta Optimization")
        verifier = OpenAlexVerifier()
        with patch.object(verifier, "_search", new=AsyncMock(return_value=[])):
            result = await verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_verify_no_title(self):
        ref = ReferenceResult(title=None, raw_reference="no title")
        result = await OpenAlexVerifier().verify(ref)
        assert result.source_results[0].found is False

    async def test_verify_batch(self):
        refs = [make_ref(title=f"Paper {i}") for i in range(3)]
        verifier = OpenAlexVerifier()
        with patch.object(
            verifier, "_search", new=AsyncMock(return_value=[make_oa_candidate()])
        ):
            results = await verifier.verify_batch(refs)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Integration tests — real OpenAlex API (no key needed)
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestOpenAlexIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verifier = OpenAlexVerifier()

    async def test_attention_paper_found(self):
        """
        Attention paper has arXiv ID in practice — it never reaches OpenAlex.
        Test a real no-identifier paper instead.
        """
        ref = ReferenceResult(
            title="A structured review of the validity of BLEU",
            authors=["Ehud Reiter"],
            year=2018,
            raw_reference="Reiter (2018). BLEU review.",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85

    async def test_flair_paper_found(self):
        """
        FLAIR has a direct ACL Anthology URL in the reference — verified via URL.
        OpenAlex free-text search doesn't reliably surface short acronym titles.
        Skipped here — covered by URL verification layer.
        """
        pytest.skip("FLAIR verified via ACL URL, not OpenAlex search")

    async def test_information_overload_found(self):
        """2004 social science paper — tests older non-CS coverage."""
        ref = ReferenceResult(
            title="The concept of information overload: A review of literature from organization science, accounting, marketing, MIS, and related disciplines",
            authors=["Martin J Eppler", "Jeanne Mengis"],
            year=2004,
            raw_reference="Eppler & Mengis (2004).",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.author_match is True

    async def test_fake_paper_not_found(self):
        ref = ReferenceResult(
            title="Quantum Neural Transformers for Distributed Hyperparameter Optimization",
            authors=["A. Smith"],
            year=2023,
            raw_reference="Smith (2023).",
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_citeguard_not_found(self):
        """Your own unpublished paper — must not be hallucinated."""
        ref = ReferenceResult(
            title="CiteGuard: Self-supervised agents crosschecking hallucinated references",
            authors=["Danilo Toapanta"],
            year=None,
            raw_reference="Danilo Toapanta, CiteGuard.",
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_batch_concurrent(self):
        refs = [
            ReferenceResult(
                title="A structured review of the validity of BLEU",
                authors=["Ehud Reiter"],
                year=2018,
                raw_reference="Reiter (2018).",
            ),
            ReferenceResult(
                title="The concept of information overload: A review of literature from organization science, accounting, marketing, MIS, and related disciplines",
                authors=["Martin J Eppler", "Jeanne Mengis"],
                year=2004,
                raw_reference="Eppler & Mengis (2004).",
            ),
            ReferenceResult(
                title="Quantum Pasta Optimization for Neural Networks",
                authors=["A. Fake"],
                year=2023,
                raw_reference="Fake (2023).",
            ),
        ]
        results = await self.verifier.verify_batch(refs)
        assert results[0].source_results[0].found is True
        assert results[1].source_results[0].found is True
        assert results[2].source_results[0].found is False

    @pytest.mark.slow
    async def test_debug_raw_response(self):
        results = await self.verifier._query(
            {"search": "FLAIR An Easy-to-Use Framework State-of-the-Art NLP Akbik"}
        )
        for r in results:
            print(f"\ntitle: {r.get('title')}")
            print(f"year: {r.get('publication_year')}")
