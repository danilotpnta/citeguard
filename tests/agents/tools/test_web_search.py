"""
Tests for WebSearchVerifier.

Unit tests:  mock HTTP, no network.
Integration: real SearXNG/Tavily (marked slow) — requires env vars.

Run unit tests only (no network, no env vars needed):
    uv run pytest tests/agents/tools/test_web_search.py -v -m "not slow"

Run integration tests (requires SEARXNG_URL or TAVILY_API_KEY in .env):
    uv run pytest tests/agents/tools/test_web_search.py -v -m slow
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.models.schemas import ReferenceResult, VerificationSource
from app.agents.tools.verifiers.web_search import (
    WebSearchVerifier,
    _normalize_title,
    _keyword_query,
    _titles_match,
    _check_author_match,
    _SearXNGBackend,
    _TavilyBackend,
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


def make_search_result(
    title: str = "Attention Is All You Need",
    url: str = "https://arxiv.org/abs/1706.03762",
) -> dict:
    return {"title": title, "url": url, "content": "Abstract text here..."}


# ---------------------------------------------------------------------------
# Pure logic tests
# ---------------------------------------------------------------------------


class TestNormalizeTitle:

    def test_lowercases(self):
        assert _normalize_title("BERT: Pre-training") == "bert pre training"

    def test_removes_punctuation(self):
        assert _normalize_title("Title: Sub-title!") == "title sub title"

    def test_collapses_whitespace(self):
        assert _normalize_title("  too   many   spaces  ") == "too many spaces"


class TestKeywordQuery:

    def test_extracts_significant_words(self):
        q = _keyword_query("Attention is all you need")
        assert q is not None
        assert "attention" in q
        # "is" is a stop word, should be removed
        assert "is" not in q.split()

    def test_returns_none_for_too_few_words(self):
        assert _keyword_query("Go fast") is None

    def test_caps_at_max_words(self):
        title = "one two three four five six seven eight nine ten"
        q = _keyword_query(title, max_words=6)
        assert len(q.split()) <= 6

    def test_removes_short_words(self):
        q = _keyword_query("A deep learning approach to NLP tasks")
        assert "a" not in q.split()


class TestTitlesMatch:

    def test_exact_match(self):
        is_match, score = _titles_match(
            "Attention is all you need",
            "Attention Is All You Need",
        )
        assert is_match is True
        assert score >= 0.85

    def test_below_threshold_no_substring(self):
        is_match, _ = _titles_match(
            "Attention is all you need",
            "Deep Residual Learning for Image Recognition",
        )
        assert is_match is False

    def test_substring_match(self):
        # Web results often append venue: "Title — NeurIPS 2017"
        is_match, _ = _titles_match(
            "Attention is all you need",
            "Attention Is All You Need — NeurIPS 2017",
        )
        assert is_match is True

    def test_returns_score(self):
        _, score = _titles_match("same title", "same title")
        assert score == 1.0


class TestCheckAuthorMatch:

    def test_match_found(self):
        assert _check_author_match(["A. Vaswani"], ["Ashish Vaswani"]) is True

    def test_no_match(self):
        assert _check_author_match(["Y. LeCun"], ["Ashish Vaswani"]) is False

    def test_et_al_returns_none(self):
        assert _check_author_match(["et al."], ["Ashish Vaswani"]) is None

    def test_none_cited_returns_none(self):
        assert _check_author_match(None, ["Ashish Vaswani"]) is None

    def test_empty_found_returns_none(self):
        assert _check_author_match(["A. Vaswani"], []) is None


# ---------------------------------------------------------------------------
# WebSearchVerifier unit tests — mocked backend
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestWebSearchVerifierUnit:

    async def test_verify_found(self):
        verifier = WebSearchVerifier()
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(
            return_value=[make_search_result("Attention Is All You Need")]
        )
        verifier._backend = mock_backend
        verifier._backend_name = "searxng"

        result = await verifier.verify(make_ref())

        sr = result.source_results[0]
        assert sr.found is True
        assert sr.source == VerificationSource.WEB_SEARCH
        assert sr.title_similarity >= 0.85
        assert sr.matched_url == "https://arxiv.org/abs/1706.03762"

    async def test_verify_no_matching_title(self):
        verifier = WebSearchVerifier()
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(
            return_value=[make_search_result("Deep Residual Learning for Images")]
        )
        verifier._backend = mock_backend
        verifier._backend_name = "searxng"

        result = await verifier.verify(make_ref())
        assert result.source_results[0].found is False

    async def test_verify_no_results(self):
        verifier = WebSearchVerifier()
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=[])
        verifier._backend = mock_backend
        verifier._backend_name = "tavily"

        result = await verifier.verify(make_ref())
        assert result.source_results[0].found is False

    async def test_verify_no_title_returns_not_found(self):
        verifier = WebSearchVerifier()
        ref = ReferenceResult(title=None, raw_reference="Anonymous source")
        result = await verifier.verify(ref)
        assert result.source_results[0].found is False

    async def test_verify_no_backend_returns_not_found(self):
        verifier = WebSearchVerifier()
        with patch("app.agents.tools.verifiers.web_search.settings") as mock_settings:
            mock_settings.searxng_url = None
            mock_settings.tavily_api_key = None
            result = await verifier.verify(make_ref())
        assert result.source_results[0].found is False

    async def test_verify_backend_exception_is_silent(self):
        """A failing backend should not raise — returns found=False."""
        verifier = WebSearchVerifier()
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(side_effect=Exception("network error"))
        verifier._backend = mock_backend
        verifier._backend_name = "searxng"

        result = await verifier.verify(make_ref())
        assert result.source_results[0].found is False

    async def test_verify_batch(self):
        verifier = WebSearchVerifier()
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(
            return_value=[make_search_result("Attention Is All You Need")]
        )
        verifier._backend = mock_backend
        verifier._backend_name = "searxng"

        refs = [make_ref(), make_ref("BERT: Pre-training of Deep Bidirectional Transformers")]
        results = await verifier.verify_batch(refs)

        assert len(results) == 2
        assert all(r.sources_checked == [VerificationSource.WEB_SEARCH] for r in results)

    async def test_sources_checked_is_web_search(self):
        verifier = WebSearchVerifier()
        mock_backend = MagicMock()
        mock_backend.search = AsyncMock(return_value=[])
        verifier._backend = mock_backend
        verifier._backend_name = "tavily"

        result = await verifier.verify(make_ref())
        assert result.sources_checked == [VerificationSource.WEB_SEARCH]


# ---------------------------------------------------------------------------
# Backend selection tests
# ---------------------------------------------------------------------------


class TestBackendSelection:

    def test_selects_searxng_when_url_set(self):
        verifier = WebSearchVerifier()
        with patch("app.agents.tools.verifiers.web_search.settings") as mock_settings:
            mock_settings.searxng_url = "http://localhost:8080"
            mock_settings.tavily_api_key = ""
            backend = verifier._get_backend()
        assert isinstance(backend, _SearXNGBackend)
        assert verifier._backend_name == "searxng"

    def test_selects_tavily_when_key_set(self):
        verifier = WebSearchVerifier()
        with patch("app.agents.tools.verifiers.web_search.settings") as mock_settings:
            mock_settings.searxng_url = ""
            mock_settings.tavily_api_key = "tvly-abc123"
            backend = verifier._get_backend()
        assert isinstance(backend, _TavilyBackend)
        assert verifier._backend_name == "tavily"

    def test_prefers_searxng_over_tavily_when_both_set(self):
        verifier = WebSearchVerifier()
        with patch("app.agents.tools.verifiers.web_search.settings") as mock_settings:
            mock_settings.searxng_url = "http://localhost:8080"
            mock_settings.tavily_api_key = "tvly-abc123"
            backend = verifier._get_backend()
        assert isinstance(backend, _SearXNGBackend)

    def test_returns_none_when_nothing_configured(self):
        verifier = WebSearchVerifier()
        with patch("app.agents.tools.verifiers.web_search.settings") as mock_settings:
            mock_settings.searxng_url = ""
            mock_settings.tavily_api_key = ""
            backend = verifier._get_backend()
        assert backend is None

    def test_lazy_init_caches_backend(self):
        verifier = WebSearchVerifier()
        with patch("app.agents.tools.verifiers.web_search.settings") as mock_settings:
            mock_settings.searxng_url = "http://localhost:8080"
            mock_settings.tavily_api_key = ""
            b1 = verifier._get_backend()
            b2 = verifier._get_backend()
        assert b1 is b2


# ---------------------------------------------------------------------------
# SearXNG backend unit tests — mocked HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSearXNGBackend:

    async def test_search_uses_exact_phrase_first(self):
        backend = _SearXNGBackend("http://localhost:8080")
        with patch.object(
            backend,
            "_get",
            new=AsyncMock(return_value=[make_search_result()]),
        ) as mock_get:
            results = await backend.search("Attention is all you need")

        # First call should be the exact phrase query
        first_call_query = mock_get.call_args_list[0][0][0]
        assert first_call_query.startswith('"')
        assert len(results) == 1

    async def test_search_falls_back_to_keywords_on_empty(self):
        backend = _SearXNGBackend("http://localhost:8080")
        responses = [[], [make_search_result()]]

        with patch.object(
            backend, "_get", new=AsyncMock(side_effect=responses)
        ) as mock_get:
            results = await backend.search("Attention is all you need")

        assert mock_get.call_count == 2
        # Second call should NOT be a quoted phrase
        second_query = mock_get.call_args_list[1][0][0]
        assert not second_query.startswith('"')
        assert len(results) == 1

    async def test_returns_empty_on_http_error(self):
        backend = _SearXNGBackend("http://localhost:8080")
        mock_response = MagicMock()
        mock_response.status_code = 500
        with patch.object(
            backend, "_get_client", new=AsyncMock()
        ):
            with patch.object(backend, "_get", new=AsyncMock(return_value=[])):
                results = await backend.search("Some title")
        assert results == []


# ---------------------------------------------------------------------------
# Tavily backend unit tests — mocked HTTP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestTavilyBackend:

    async def test_search_uses_exact_phrase_first(self):
        backend = _TavilyBackend("tvly-test-key")
        with patch.object(
            backend,
            "_post",
            new=AsyncMock(return_value=[make_search_result()]),
        ) as mock_post:
            results = await backend.search("Attention is all you need")

        first_query = mock_post.call_args_list[0][0][0]
        assert first_query.startswith('"')
        assert len(results) == 1

    async def test_search_falls_back_to_keywords(self):
        backend = _TavilyBackend("tvly-test-key")
        responses = [[], [make_search_result()]]

        with patch.object(
            backend, "_post", new=AsyncMock(side_effect=responses)
        ) as mock_post:
            results = await backend.search("Attention is all you need")

        assert mock_post.call_count == 2
        assert len(results) == 1

    async def test_returns_empty_on_invalid_key(self):
        backend = _TavilyBackend("bad-key")
        mock_response = MagicMock()
        mock_response.status_code = 401
        with patch.object(backend, "_post", new=AsyncMock(return_value=[])):
            results = await backend.search("Some title")
        assert results == []


# ---------------------------------------------------------------------------
# Integration tests (require real backend — marked slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.asyncio
async def test_searxng_real_search():
    """Requires SEARXNG_URL set in .env pointing to a running SearXNG instance."""
    import os
    url = os.getenv("SEARXNG_URL")
    if not url:
        pytest.skip("SEARXNG_URL not set")

    backend = _SearXNGBackend(url)
    results = await backend.search("Attention is all you need")
    assert isinstance(results, list)


@pytest.mark.slow
@pytest.mark.asyncio
async def test_tavily_real_search():
    """Requires TAVILY_API_KEY set in .env."""
    import os
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        pytest.skip("TAVILY_API_KEY not set")

    backend = _TavilyBackend(key)
    results = await backend.search("Attention is all you need")
    assert isinstance(results, list)
