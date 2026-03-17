"""
Tests for ArxivVerifier.

Unit tests:  mock HTTP, test logic only — no network.
Integration: hit real arXiv API (marked slow).

Run unit tests:
    pytest tests/agents/tools/test_arxiv.py -v -m "not slow"

Run all:
    pytest tests/agents/tools/test_arxiv.py -v -m slow
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from app.models.schemas import ReferenceResult, VerificationSource
from app.agents.tools.verifiers.arxiv import (
    ArxivVerifier,
    _normalize_arxiv_id,
    _normalize_title,
    _check_author_match,
    _parse_arxiv_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_ref(
    title: str = "Attention is all you need",
    authors: list[str] | None = None,
    year: int | None = 2017,
    arxiv_id: str = "1706.03762",
) -> ReferenceResult:
    return ReferenceResult(
        title=title,
        authors=authors or ["A. Vaswani", "N. Shazeer"],
        year=year,
        arxiv_id=arxiv_id,
        raw_reference=f"{title} ({year})",
    )


SAMPLE_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <author><name>Niki Parmar</name></author>
    <published>2017-06-12T17:57:34Z</published>
    <link type="text/html" href="https://arxiv.org/abs/1706.03762"/>
  </entry>
</feed>"""

EMPTY_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
</feed>"""

ERROR_ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Error</title>
  </entry>
</feed>"""


# ---------------------------------------------------------------------------
# Unit tests — pure logic
# ---------------------------------------------------------------------------


class TestNormalizeArxivId:

    def test_strips_version_suffix(self):
        assert _normalize_arxiv_id("1706.03762v2") == "1706.03762"

    def test_strips_arxiv_prefix(self):
        assert _normalize_arxiv_id("arXiv:1706.03762") == "1706.03762"

    def test_strips_prefix_case_insensitive(self):
        assert _normalize_arxiv_id("ARXIV:1706.03762") == "1706.03762"

    def test_clean_id_unchanged(self):
        assert _normalize_arxiv_id("1706.03762") == "1706.03762"

    def test_strips_version_v3(self):
        assert _normalize_arxiv_id("2310.14746v3") == "2310.14746"


class TestParseArxivResponse:

    def test_parses_valid_response(self):
        result = _parse_arxiv_response(SAMPLE_ARXIV_XML)
        assert result is not None
        assert result["title"] == "Attention Is All You Need"
        assert len(result["authors"]) == 3
        assert "Ashish Vaswani" in result["authors"]
        assert result["year"] == 2017
        assert result["url"] == "https://arxiv.org/abs/1706.03762"

    def test_returns_none_for_empty_feed(self):
        result = _parse_arxiv_response(EMPTY_ARXIV_XML)
        assert result is None

    def test_returns_none_for_error_entry(self):
        result = _parse_arxiv_response(ERROR_ARXIV_XML)
        assert result is None

    def test_returns_none_for_invalid_xml(self):
        result = _parse_arxiv_response("not xml at all")
        assert result is None


class TestCheckAuthorMatchArxiv:

    def test_match_found(self):
        cited = ["A. Vaswani", "N. Shazeer"]
        arxiv = ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
        assert _check_author_match(cited, arxiv) is True

    def test_no_match(self):
        cited = ["Y. LeCun"]
        arxiv = ["Ashish Vaswani", "Noam Shazeer"]
        assert _check_author_match(cited, arxiv) is False

    def test_et_al_returns_none(self):
        """'et al.' only — no real names to compare, skip match."""
        cited = ["Zirui Guo et al."]
        arxiv = ["Zirui Guo", "Someone Else"]
        assert _check_author_match(cited, arxiv) is None

    def test_mixed_et_al_and_real(self):
        """Has one real name alongside et al. — use real name."""
        cited = ["Darren Edge", "et al."]
        arxiv = ["Darren Edge", "Someone Else"]
        assert _check_author_match(cited, arxiv) is True

    def test_none_cited_returns_none(self):
        assert _check_author_match(None, ["Ashish Vaswani"]) is None

    def test_empty_arxiv_returns_none(self):
        assert _check_author_match(["A. Vaswani"], []) is None


class TestBuildSourceResultArxiv:

    def setup_method(self):
        self.verifier = ArxivVerifier()

    def test_not_found_returns_found_false(self):
        ref = make_ref()
        result = self.verifier._build_source_result(ref, None)
        assert result.found is False
        assert result.source == VerificationSource.ARXIV

    def test_high_similarity_match(self):
        ref = make_ref(title="Attention is all you need")
        parsed = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "url": "https://arxiv.org/abs/1706.03762",
        }
        result = self.verifier._build_source_result(ref, parsed)
        assert result.found is True
        assert result.title_similarity >= 0.85

    def test_year_delta_correct(self):
        ref = make_ref(year=2019)
        parsed = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani"],
            "year": 2017,
            "url": None,
        }
        result = self.verifier._build_source_result(ref, parsed)
        assert result.year_delta == 2

    def test_year_delta_none_when_no_ref_year(self):
        ref = make_ref(year=None)
        parsed = {"title": "Some Title", "authors": [], "year": 2017, "url": None}
        result = self.verifier._build_source_result(ref, parsed)
        assert result.year_delta is None

    def test_matched_url_populated(self):
        ref = make_ref()
        parsed = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani"],
            "year": 2017,
            "url": "https://arxiv.org/abs/1706.03762",
        }
        result = self.verifier._build_source_result(ref, parsed)
        assert result.matched_url == "https://arxiv.org/abs/1706.03762"

    def test_et_al_author_skips_match(self):
        """et al. only → author_match is None, not False."""
        ref = make_ref(authors=["Zirui Guo et al."])
        parsed = {
            "title": "LightRAG",
            "authors": ["Zirui Guo", "Someone Else"],
            "year": 2024,
            "url": None,
        }
        result = self.verifier._build_source_result(ref, parsed)
        assert result.author_match is None


# ---------------------------------------------------------------------------
# Async unit tests — mocked HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestArxivVerifierAsync:

    async def test_verify_confirmed_paper(self):
        ref = make_ref()
        parsed = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": 2017,
            "url": "https://arxiv.org/abs/1706.03762",
        }
        verifier = ArxivVerifier()
        with patch.object(verifier, "_fetch", new=AsyncMock(return_value=parsed)):
            result = await verifier.verify(ref)

        assert VerificationSource.ARXIV in result.sources_checked
        sr = result.source_results[0]
        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.year_delta == 0

    async def test_verify_not_found(self):
        ref = make_ref(arxiv_id="9999.99999")
        verifier = ArxivVerifier()
        with patch.object(verifier, "_fetch", new=AsyncMock(return_value=None)):
            result = await verifier.verify(ref)

        assert result.source_results[0].found is False

    async def test_verify_raises_without_arxiv_id(self):
        ref = ReferenceResult(
            title="No arXiv ID",
            raw_reference="No arXiv ID raw",
        )
        verifier = ArxivVerifier()
        with pytest.raises(ValueError, match="no arxiv_id"):
            await verifier.verify(ref)

    async def test_verify_batch(self):
        refs = [
            make_ref(title=f"Paper {i}", arxiv_id=f"1706.0376{i}") for i in range(3)
        ]
        parsed = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani"],
            "year": 2017,
            "url": None,
        }
        verifier = ArxivVerifier()
        with patch.object(verifier, "_fetch", new=AsyncMock(return_value=parsed)):
            results = await verifier.verify_batch(refs)

        assert len(results) == 3
        assert all(r.source_results[0].found for r in results)

    async def test_fetch_non_200_returns_none(self):
        verifier = ArxivVerifier()
        mock_response = MagicMock()
        mock_response.status_code = 503
        mock_response.text = ""

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client.is_closed = False
        verifier._client = mock_client

        result = await verifier._fetch("1706.03762")
        assert result is None


# ---------------------------------------------------------------------------
# Integration tests — real arXiv API
# ---------------------------------------------------------------------------


@pytest.mark.slow
class TestArxivIntegration:

    @pytest.fixture(autouse=True)
    def setup(self):
        self.verifier = ArxivVerifier()

    async def test_attention_paper_verified(self):
        """Attention is All You Need — should always resolve."""
        ref = ReferenceResult(
            title="Attention is all you need",
            authors=["A. Vaswani", "N. Shazeer"],
            year=2017,
            arxiv_id="1706.03762",
            raw_reference="Vaswani et al. (2017).",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]

        assert sr.found is True
        assert sr.title_similarity >= 0.85
        assert sr.author_match is True
        assert sr.year_delta == 0
        assert sr.matched_url is not None

    async def test_fake_arxiv_id_not_found(self):
        ref = ReferenceResult(
            title="Fake Preprint",
            arxiv_id="9999.99999",
            raw_reference="Fake preprint raw",
        )
        result = await self.verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_promoted_arxiv_doi(self):
        """
        [83] Graph RAG — originally had doi:10.48550/arXiv.2404.16130.
        classify_references_node extracted arxiv_id='2404.16130'.
        arXiv should find it directly.
        """
        ref = ReferenceResult(
            title="From Local to Global: A Graph RAG Approach to Query-Focused Summarization",
            authors=["Darren Edge et al."],
            year=2025,
            arxiv_id="2404.16130",
            raw_reference="Darren Edge et al. (2025). Graph RAG.",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]

        assert sr.found is True
        assert sr.title_similarity >= 0.80

    async def test_et_al_author_match_skipped_gracefully(self):
        """
        LightRAG has 'Zirui Guo et al.' — author match should be None not False.
        Paper should still be found.
        """
        ref = ReferenceResult(
            title="LightRAG: Simple and Fast Retrieval-Augmented Generation",
            authors=["Zirui Guo et al."],
            year=2024,
            arxiv_id="2410.05779",
            raw_reference="Zirui Guo et al. (2024). LightRAG.",
        )
        result = await self.verifier.verify(ref)
        sr = result.source_results[0]

        assert sr.found is True
        assert sr.author_match is None  # et al. — skipped, not False
        assert sr.title_similarity >= 0.85

    async def test_batch_real_and_fake(self):
        refs = [
            ReferenceResult(
                title="Attention is all you need",
                authors=["A. Vaswani"],
                year=2017,
                arxiv_id="1706.03762",
                raw_reference="Vaswani et al.",
            ),
            ReferenceResult(
                title="Fake Preprint",
                arxiv_id="9999.99999",
                raw_reference="Fake raw",
            ),
        ]
        results = await self.verifier.verify_batch(refs)
        assert results[0].source_results[0].found is True
        assert results[1].source_results[0].found is False
