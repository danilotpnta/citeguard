"""
Tests for SemanticScholarVerifier.

Unit tests:  mock HTTP, test logic only — no network.
Integration: hit real SS API (marked slow).

Run unit tests:
    pytest tests/agents/tools/test_semantic_scholar.py -v -m "not slow"

Run all:
    pytest tests/agents/tools/test_semantic_scholar.py -v -m slow
"""

import pytest
from unittest.mock import AsyncMock, patch

from app.models.schemas import ReferenceResult, VerificationSource
from app.agents.tools.verifiers.semantic_scholar import (
    SemanticScholarVerifier,
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


def make_ss_candidate(
    title: str = "Attention Is All You Need",
    author_names: list[str] | None = None,
    year: int = 2017,
    doi: str | None = "10.48550/arXiv.1706.03762",
    arxiv_id: str | None = "1706.03762",
    oa_pdf: str | None = None,
) -> dict:
    return {
        "title": title,
        "authors": [
            {"name": n} for n in (author_names or ["Ashish Vaswani", "Noam Shazeer"])
        ],
        "year": year,
        "externalIds": {k: v for k, v in {"DOI": doi, "ArXiv": arxiv_id}.items() if v},
        "openAccessPdf": {"url": oa_pdf} if oa_pdf else None,
        "publicationVenue": None,
    }


# ---------------------------------------------------------------------------
# Unit tests — pure logic
# ---------------------------------------------------------------------------


class TestBestMatch:

    def test_returns_none_for_empty_candidates(self):
        ref = make_ref()
        assert _best_match(ref, []) is None

    def test_returns_none_when_no_title_on_ref(self):
        ref = ReferenceResult(title=None, raw_reference="raw")
        assert _best_match(ref, [make_ss_candidate()]) is None

    def test_returns_best_above_threshold(self):
        ref = make_ref(title="Attention is all you need")
        candidates = [make_ss_candidate(title="Attention Is All You Need")]
        result = _best_match(ref, candidates)
        assert result is not None
        assert result["_title_similarity"] >= 0.85

    def test_returns_none_below_threshold(self):
        ref = make_ref(title="Completely Different Title")
        candidates = [make_ss_candidate(title="Attention Is All You Need")]
        assert _best_match(ref, candidates) is None

    def test_picks_best_among_multiple_candidates(self):
        ref = make_ref(title="Attention is all you need")
        candidates = [
            make_ss_candidate(title="Something Unrelated"),
            make_ss_candidate(title="Attention Is All You Need"),
            make_ss_candidate(title="Attention Mechanism Survey"),
        ]
        result = _best_match(ref, candidates)
        assert result is not None
        assert result["title"] == "Attention Is All You Need"


class TestExtractUrl:

    def test_prefers_open_access_pdf(self):
        candidate = make_ss_candidate(oa_pdf="https://arxiv.org/pdf/1706.03762")
        assert _extract_url(candidate) == "https://arxiv.org/pdf/1706.03762"

    def test_falls_back_to_doi(self):
        candidate = make_ss_candidate(
            doi="10.1038/nature14539", arxiv_id=None, oa_pdf=None
        )
        assert _extract_url(candidate) == "https://doi.org/10.1038/nature14539"

    def test_falls_back_to_arxiv(self):
        candidate = make_ss_candidate(doi=None, arxiv_id="1706.03762", oa_pdf=None)
        assert _extract_url(candidate) == "https://arxiv.org/abs/1706.03762"

    def test_returns_none_when_nothing_available(self):
        candidate = {"title": "Some paper", "externalIds": {}, "openAccessPdf": None}
        assert _extract_url(candidate) is None


class TestCheckAuthorMatchSS:

    def test_match_found(self):
        cited = ["A. Vaswani"]
        ss = [{"name": "Ashish Vaswani"}, {"name": "Noam Shazeer"}]
        assert _check_author_match(cited, ss) is True

    def test_no_match(self):
        cited = ["Y. LeCun"]
        ss = [{"name": "Ashish Vaswani"}]
        assert _check_author_match(cited, ss) is False

    def test_et_al_returns_none(self):
        cited = ["Zirui Guo et al."]
        ss = [{"name": "Zirui Guo"}]
        assert _check_author_match(cited, ss) is None

    def test_none_cited_returns_none(self):
        assert _check_author_match(None, [{"name": "Vaswani"}]) is None

    def test_empty_ss_authors_returns_none(self):
        assert _check_author_match(["A. Vaswani"], []) is None


class TestBuildSourceResultSS:

    def setup_method(self):
        self.verifier = SemanticScholarVerifier()

    def test_not_found_returns_found_false(self):
        ref = make_ref()
        result = self.verifier._build_source_result(ref, None)
        assert result.found is False
        assert result.source == VerificationSource.SEMANTIC_SCHOLAR

    def test_found_populates_fields(self):
        ref = make_ref(title="Attention is all you need", year=2017)
        candidate = make_ss_candidate()
        candidate["_title_similarity"] = 0.97
        result = self.verifier._build_source_result(ref, candidate)

        assert result.found is True
        assert result.title_similarity == 0.97
        assert result.year_delta == 0
        assert result.author_match is True

    def test_year_delta_computed(self):
        ref = make_ref(year=2020)
        candidate = make_ss_candidate(year=2017)
        candidate["_title_similarity"] = 0.95
        result = self.verifier._build_source_result(ref, candidate)
        assert result.year_delta == 3

    def test_matched_url_from_oa_pdf(self):
        ref = make_ref()
        candidate = make_ss_candidate(oa_pdf="https://arxiv.org/pdf/1706.03762")
        candidate["_title_similarity"] = 0.95
        result = self.verifier._build_source_result(ref, candidate)
        assert result.matched_url == "https://arxiv.org/pdf/1706.03762"


# ---------------------------------------------------------------------------
# Async unit tests — mocked HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSemanticScholarAsync:

    async def test_verify_confirmed_paper(self):
        ref = make_ref()
        candidates = [make_ss_candidate()]

        verifier = SemanticScholarVerifier()
        with patch.object(verifier, "_search", new=AsyncMock(return_value=candidates)):
            result = await verifier.verify(ref)

        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.author_match is True

    async def test_verify_no_candidates_found(self):
        ref = make_ref(title="Quantum Pasta Optimization")

        verifier = SemanticScholarVerifier()
        with patch.object(verifier, "_search", new=AsyncMock(return_value=[])):
            result = await verifier.verify(ref)

        assert result.source_results[0].found is False

    async def test_verify_candidates_but_no_title_match(self):
        ref = make_ref(title="Completely Different Title")
        candidates = [make_ss_candidate(title="Attention Is All You Need")]

        verifier = SemanticScholarVerifier()
        with patch.object(verifier, "_search", new=AsyncMock(return_value=candidates)):
            result = await verifier.verify(ref)

        assert result.source_results[0].found is False

    async def test_verify_no_title_on_ref(self):
        ref = ReferenceResult(title=None, raw_reference="No title ref")

        verifier = SemanticScholarVerifier()
        result = await verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_verify_batch(self):
        refs = [make_ref(title=f"Paper {i}") for i in range(3)]
        candidates = [make_ss_candidate()]

        verifier = SemanticScholarVerifier()
        with patch.object(verifier, "_search", new=AsyncMock(return_value=candidates)):
            results = await verifier.verify_batch(refs)

        assert len(results) == 3


# ---------------------------------------------------------------------------
# Integration tests — real Semantic Scholar API
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestSemanticScholarIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verifier = SemanticScholarVerifier()
        yield
        import time

        time.sleep(3)  # wait between tests to avoid 429

    async def test_attention_paper_found(self):
        ref = ReferenceResult(
            title="Attention is all you need",
            authors=["A. Vaswani", "N. Shazeer"],
            year=2017,
            raw_reference="Vaswani et al. (2017).",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]

        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.author_match is True
        assert sr.year_delta == 0

    async def test_flair_nlp_paper_found(self):
        """FLAIR — no DOI, no arXiv, ACL paper — SS should find it."""
        ref = ReferenceResult(
            title="FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP",
            authors=["Alan Akbik et al."],
            year=2019,
            raw_reference="Alan Akbik et al. (2019). FLAIR.",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]

        assert sr.found is True
        assert sr.title_similarity >= 0.85

    async def test_information_overload_paper_found(self):
        """Old non-CS paper from 2004 — tests broader coverage."""
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
            raw_reference="Smith (2023). Quantum Neural Transformers.",
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_citeguard_not_found(self):
        """
        [94] CiteGuard — your own unpublished paper.
        Should not be found anywhere. Key anti-hallucination test.
        """
        ref = ReferenceResult(
            title="CiteGuard: Self-supervised agents crosschecking hallucinated references",
            authors=["Danilo Toapanta"],
            year=None,
            raw_reference="Danilo Toapanta, CiteGuard.",
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_batch_mixed(self):
        refs = [
            ReferenceResult(
                title="Attention is all you need",
                authors=["A. Vaswani"],
                year=2017,
                raw_reference="Vaswani et al.",
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
        assert results[1].source_results[0].found is False

    @pytest.mark.slow
    async def test_debug_rate_limit(self):
        """
        Fires one request with a long wait before it.
        If it passes after a fresh window, rate limiting is confirmed as the only issue.
        """
        import asyncio

        # Wait long enough for SS to reset its window
        await asyncio.sleep(10)

        ref = ReferenceResult(
            title="FLAIR: An Easy-to-Use Framework for State-of-the-Art NLP",
            authors=["Alan Akbik"],
            year=2019,
            raw_reference="Alan Akbik et al. (2019). FLAIR.",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]

        print(f"\nfound: {sr.found}")
        print(f"title_sim: {sr.title_similarity}")
        print(f"matched_title: {sr.matched_title}")

        assert sr.found is True

    @pytest.mark.slow
    async def test_debug_ss_raw(self):
        """Shows raw SS response to distinguish 429 from empty results."""
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": "FLAIR easy-to-use framework state-of-the-art NLP",
                    "fields": "title,authors,year",
                    "limit": 3,
                },
            )

        print(f"\nStatus: {response.status_code}")
        print(f"Body: {response.text[:500]}")

    @pytest.mark.slow
    async def test_debug_find_minimum_wait(self):
        """
        Fires requests with increasing wait times to find the minimum
        that SS allows without a 429.
        """
        import asyncio
        import httpx

        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        params = {
            "query": "Attention is all you need",
            "fields": "title,year",
            "limit": 1,
        }

        async with httpx.AsyncClient() as client:
            # Fire first request to trigger rate limit
            r1 = await client.get(url, params=params)
            print(f"\nRequest 1: {r1.status_code}")

            for wait in [1, 2, 3, 5, 8, 10]:
                await asyncio.sleep(wait)
                r = await client.get(url, params=params)
                print(f"After {wait}s wait: {r.status_code}")
                if r.status_code == 200:
                    print(f"  → Minimum wait found: {wait}s")
                    break
                elif r.status_code == 429:
                    retry_after = r.headers.get("Retry-After", "not set")
                    print(f"  → Still 429, Retry-After header: {retry_after}")

    @pytest.mark.slow
    async def test_debug_search_pipeline(self):
        import asyncio
        import httpx

        test_cases = [
            ("InPars: Unsupervised dataset generation for information retrieval", 2022),
            (
                "Efficient Memory Management for Large Language Model Serving with PagedAttention",
                2023,
            ),
            (
                "PAQ: 65 million probably-asked questions and what you can do with them",
                2021,
            ),
            ("Finetuned language models are zero-shot learners", 2021),
            ("Fact or Fiction: Verifying Scientific Claims", 2020),
        ]

        async with httpx.AsyncClient() as client:
            for title, year in test_cases:
                await asyncio.sleep(10)
                print(f"\n{'='*60}")
                print(f"Title: {title[:60]}")

                response = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": title,
                        "fields": "title,authors,year",
                        "limit": 3,
                    },
                )

                print(f"Status: {response.status_code}")

                if response.status_code == 200:
                    data = response.json()
                    total = data.get("total", 0)
                    results = data.get("data", [])
                    print(f"Total results: {total}")
                    for i, r in enumerate(results):
                        print(f"  [{i}] {r.get('title')}")
                        print(f"       year: {r.get('year')}")
                        print(
                            f"       authors: {[a.get('name') for a in r.get('authors', [])][:2]}"
                        )
                elif response.status_code == 429:
                    print("429 — still rate limited despite 10s wait")
