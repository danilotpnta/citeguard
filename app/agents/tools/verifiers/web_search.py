"""
Web search fallback verifier.

Last-resort verification for references that all academic databases failed to find.
Supports two backends, selected by which env var is present:

  SearXNG (preferred — self-hosted, targets Google Scholar + academic engines):
      Set SEARXNG_URL=http://localhost:8080
      See docker/searxng/ for a one-command setup.

  Tavily (cloud alternative — no infrastructure needed):
      Set TAVILY_API_KEY=tvly-xxxxxxxxxx
      Sign up at https://tavily.com (free tier: 1 000 req/month).

Results are tagged VerificationSource.WEB_SEARCH and scored as LIKELY_REAL at
best (never VERIFIED), reflecting the weaker signal from a web search match.

Search strategy (same for both backends):
  1. Exact phrase:  "Full Title As Written"
  2. Keyword query: top-6 significant words joined by spaces (fallback)

Title matching uses the same 0.85 rapidfuzz threshold as all other verifiers,
plus a substring containment check to handle "Title — Conference Name" suffixes.
"""

import asyncio
import re

import httpx
from rapidfuzz import fuzz
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from app.core.logging import get_logger
from app.models.schemas import (
    ReferenceResult,
    SourceResult,
    VerificationResult,
    VerificationSource,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TITLE_MATCH_THRESHOLD = 0.85
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_CONCURRENT = 5

# Words excluded when building the keyword fallback query
_STOP_WORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "is", "are", "was", "were", "be", "been",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _keyword_query(title: str, max_words: int = 6) -> str | None:
    """Return top-N significant words from the title, or None if too few."""
    words = [
        w for w in re.split(r"\W+", title.lower())
        if w and w not in _STOP_WORDS and len(w) > 2
    ]
    significant = words[:max_words]
    return " ".join(significant) if len(significant) >= 3 else None


def _titles_match(cited: str, found: str) -> tuple[bool, float]:
    """
    Returns (is_match, similarity_score).
    Match if rapidfuzz ratio >= threshold OR found title contains cited as substring
    (handles 'Title — Venue' suffixes common in web results).
    """
    norm_cited = _normalize_title(cited)
    norm_found = _normalize_title(found)
    score = fuzz.ratio(norm_cited, norm_found) / 100.0
    if score >= TITLE_MATCH_THRESHOLD:
        return True, score
    if norm_cited in norm_found:
        return True, score
    return False, score


def _extract_lastname(author: str) -> str:
    author = author.strip().rstrip(".")
    if "," in author:
        return author.split(",")[0].strip().lower()
    parts = author.split()
    return parts[-1].lower() if parts else author.lower()


def _check_author_match(
    cited_authors: list[str] | None,
    found_author_strings: list[str],
) -> bool | None:
    """Standard last-name matching used across all verifiers."""
    if not cited_authors or not found_author_strings:
        return None
    real_cited = [a for a in cited_authors if "et al" not in a.lower()]
    if not real_cited:
        return None
    cited_lastnames = {_extract_lastname(a) for a in real_cited}
    found_lastnames = {_extract_lastname(a) for a in found_author_strings}
    if not found_lastnames:
        return None
    return bool(cited_lastnames & found_lastnames)


# ---------------------------------------------------------------------------
# SearXNG backend
# ---------------------------------------------------------------------------


class _SearXNGBackend:
    """
    Queries a self-hosted SearXNG instance.
    JSON format must be enabled in SearXNG settings (see docker/searxng/settings.yml).
    """

    def __init__(self, base_url: str):
        self._base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "citeguard/1.0"},
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
    )
    async def _get(self, query: str) -> list[dict]:
        client = await self._get_client()
        try:
            response = await client.get(
                f"{self._base_url}/search",
                params={"q": query, "format": "json", "categories": "science"},
            )
        except (httpx.TimeoutException, httpx.ConnectError):
            raise
        except Exception as e:
            logger.warning("SearXNG unexpected error: %s", e)
            return []

        if response.status_code != 200:
            logger.warning("SearXNG returned HTTP %d", response.status_code)
            return []

        return response.json().get("results", [])

    async def search(self, title: str) -> list[dict]:
        """Try exact phrase first, fall back to keywords."""
        results = await self._get(f'"{title}"')
        if results:
            return results
        kw = _keyword_query(title)
        if kw:
            return await self._get(kw)
        return []


# ---------------------------------------------------------------------------
# Tavily backend
# ---------------------------------------------------------------------------


class _TavilyBackend:
    """
    Queries the Tavily Search API (https://tavily.com).
    Requires TAVILY_API_KEY. Free tier: 1 000 req/month.
    """

    _BASE_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: str):
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "citeguard/1.0"},
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    @retry(
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError)),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        stop=stop_after_attempt(3),
    )
    async def _post(self, query: str) -> list[dict]:
        client = await self._get_client()
        try:
            response = await client.post(
                self._BASE_URL,
                json={
                    "api_key": self._api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": 5,
                    "include_domains": [
                        "scholar.google.com",
                        "semanticscholar.org",
                        "arxiv.org",
                        "doi.org",
                        "pubmed.ncbi.nlm.nih.gov",
                    ],
                },
            )
        except (httpx.TimeoutException, httpx.ConnectError):
            raise
        except Exception as e:
            logger.warning("Tavily unexpected error: %s", e)
            return []

        if response.status_code == 401:
            logger.error("Tavily: invalid API key")
            return []
        if response.status_code == 429:
            logger.warning("Tavily: rate limited")
            return []
        if response.status_code != 200:
            logger.warning("Tavily returned HTTP %d", response.status_code)
            return []

        return response.json().get("results", [])

    async def search(self, title: str) -> list[dict]:
        """Try exact phrase first, fall back to keywords."""
        results = await self._post(f'"{title}"')
        if results:
            return results
        kw = _keyword_query(title)
        if kw:
            return await self._post(kw)
        return []


# ---------------------------------------------------------------------------
# WebSearchVerifier
# ---------------------------------------------------------------------------


class WebSearchVerifier:
    """
    Web search fallback verifier. Uses SearXNG if SEARXNG_URL is set,
    otherwise falls back to Tavily if TAVILY_API_KEY is set.
    If neither is configured, verify() returns found=False immediately.
    """

    def __init__(self):
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._backend: _SearXNGBackend | _TavilyBackend | None = None
        self._backend_name: str = "none"

    def _get_backend(self) -> "_SearXNGBackend | _TavilyBackend | None":
        """Lazy-initialise on first use so settings are read at runtime."""
        if self._backend is not None:
            return self._backend
        from app.core.config import settings
        if settings.searxng_url:
            self._backend = _SearXNGBackend(settings.searxng_url)
            self._backend_name = "searxng"
        elif settings.tavily_api_key:
            self._backend = _TavilyBackend(settings.tavily_api_key)
            self._backend_name = "tavily"
        return self._backend

    async def close(self):
        if self._backend:
            await self._backend.close()

    def _build_source_result(
        self,
        ref: ReferenceResult,
        result: dict | None,
    ) -> SourceResult:
        if result is None:
            return SourceResult(source=VerificationSource.WEB_SEARCH, found=False)

        title = result.get("title", "") or ""
        _, similarity = _titles_match(ref.title or "", title)

        # Web results rarely expose structured author data — treat as unknown
        author_match = _check_author_match(ref.authors, [])

        return SourceResult(
            source=VerificationSource.WEB_SEARCH,
            found=True,
            title_similarity=similarity,
            author_match=author_match,
            matched_title=title or None,
            matched_url=result.get("url"),
        )

    async def verify(self, ref: ReferenceResult) -> VerificationResult:
        backend = self._get_backend()

        if not ref.title or backend is None:
            return VerificationResult(
                reference=ref,
                sources_checked=[VerificationSource.WEB_SEARCH],
                source_results=[
                    SourceResult(source=VerificationSource.WEB_SEARCH, found=False)
                ],
            )

        async with self._semaphore:
            try:
                raw_results = await backend.search(ref.title)
            except Exception as e:
                logger.warning(
                    "Web search failed for '%s' (%s): %s",
                    ref.title[:60],
                    self._backend_name,
                    e,
                )
                raw_results = []

        # Find the first result whose title matches
        best: dict | None = None
        for r in raw_results:
            found_title = r.get("title", "") or ""
            is_match, _ = _titles_match(ref.title, found_title)
            if is_match:
                best = r
                break

        source_result = self._build_source_result(ref, best)

        logger.info(
            "WebSearch [%s] backend=%s found=%s title_sim=%s",
            (ref.title or "?")[:50],
            self._backend_name,
            source_result.found,
            (
                f"{source_result.title_similarity:.2f}"
                if source_result.title_similarity is not None
                else "N/A"
            ),
        )

        return VerificationResult(
            reference=ref,
            sources_checked=[VerificationSource.WEB_SEARCH],
            source_results=[source_result],
        )

    async def verify_batch(
        self,
        refs: list[ReferenceResult],
    ) -> list[VerificationResult]:
        tasks = [self.verify(ref) for ref in refs]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

web_search_verifier = WebSearchVerifier()
