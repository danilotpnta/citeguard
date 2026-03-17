"""
Semantic Scholar verifier.

Given a ReferenceResult with a title, searches Semantic Scholar and returns
a VerificationResult with title similarity, author match, and year delta.

Uses the Graph API search endpoint:
    GET https://api.semanticscholar.org/graph/v1/paper/search
    Fields: title, authors, year, externalIds, publicationVenue, openAccessPdf

Rate limits:
    Without key: 1 req/sec
    With free key: 10 req/sec  ← strongly recommended
    Set SEMANTIC_SCHOLAR_API_KEY in .env
"""

import asyncio
import os
import re

from tenacity import retry_if_exception_type, retry_if_exception

import httpx
from rapidfuzz import fuzz
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
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

SS_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
SS_FIELDS = "title,authors,year,externalIds,publicationVenue,openAccessPdf"
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
TITLE_MATCH_THRESHOLD = 0.85
MAX_RESULTS_PER_QUERY = 5  # top 5 candidates, we pick the best match


def _get_max_concurrent() -> int:
    return 5 if os.getenv("SEMANTIC_SCHOLAR_API_KEY") else 1


# ---------------------------------------------------------------------------
# Helpers  (shared with crossref/arxiv for consistency)
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _extract_lastname(author: str) -> str:
    author = author.strip().rstrip(".")
    if "," in author:
        return author.split(",")[0].strip().lower()
    parts = author.split()
    return parts[-1].lower() if parts else author.lower()


def _check_author_match(
    cited_authors: list[str] | None,
    ss_authors: list[dict],
) -> bool | None:
    """
    Returns True if at least one cited last name appears in SS authors.
    Returns None if no usable author data on either side.
    """
    if not ss_authors:
        return None

    if not cited_authors:
        return None

    real_cited = [a for a in cited_authors if "et al" not in a.lower()]
    if not real_cited:
        return None

    cited_lastnames = {_extract_lastname(a) for a in real_cited}

    ss_lastnames = set()
    for author in ss_authors:
        name = author.get("name", "")
        if name:
            ss_lastnames.add(_extract_lastname(name))

    if not ss_lastnames:
        return None

    return bool(cited_lastnames & ss_lastnames)


def _best_match(
    ref: ReferenceResult,
    candidates: list[dict],
) -> dict | None:
    """
    From a list of SS candidate papers, return the one with the highest
    title similarity to the cited reference.
    Returns None if no candidate exceeds TITLE_MATCH_THRESHOLD.
    """
    if not candidates or not ref.title:
        return None

    norm_cited = _normalize_title(ref.title)
    best_score = 0.0
    best_candidate = None

    for candidate in candidates:
        candidate_title = candidate.get("title", "")
        if not candidate_title:
            continue
        score = fuzz.ratio(_normalize_title(candidate_title), norm_cited) / 100.0
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_score < TITLE_MATCH_THRESHOLD:
        return None

    # Attach score for use in build_source_result
    if best_candidate:
        best_candidate["_title_similarity"] = best_score

    return best_candidate


def _extract_url(candidate: dict) -> str | None:
    """Extract best available URL from a SS paper record."""
    # Prefer open access PDF
    oa = candidate.get("openAccessPdf")
    if oa and oa.get("url"):
        return oa["url"]

    # Fall back to external IDs
    ext = candidate.get("externalIds", {})
    doi = ext.get("DOI")
    if doi:
        return f"https://doi.org/{doi}"

    arxiv = ext.get("ArXiv")
    if arxiv:
        return f"https://arxiv.org/abs/{arxiv}"

    return None


# ---------------------------------------------------------------------------
# SemanticScholarVerifier
# ---------------------------------------------------------------------------


class SemanticScholarVerifier:
    """
    Async Semantic Scholar verifier. Create one instance and reuse it.
    """

    def __init__(self):
        self._api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        self._semaphore = asyncio.Semaphore(_get_max_concurrent())
        self._client: httpx.AsyncClient | None = None

        if self._api_key:
            logger.info("Semantic Scholar: using authenticated requests (10 req/sec)")
        else:
            logger.warning(
                "Semantic Scholar: no API key found — limited to 1 req/sec. "
                "Set SEMANTIC_SCHOLAR_API_KEY in .env for better performance."
            )

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            headers = {"User-Agent": "citeguard/1.0 (academic reference verifier)"}
            if self._api_key:
                headers["x-api-key"] = self._api_key
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers=headers,
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def _is_retryable(exc) -> bool:
        return isinstance(
            exc, (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPStatusError)
        )

    @retry(
        retry=retry_if_exception(_is_retryable),
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(3),
    )
    async def _search(self, title: str) -> list[dict]:
        """
        Search SS for a title string. Returns list of candidate paper dicts.
        Empty list if nothing found or on error.
        """
        client = await self._get_client()

        async with self._semaphore:
            # Respect rate limit for unauthenticated requests
            if not self._api_key:
                await asyncio.sleep(3.5)

            try:
                response = await client.get(
                    SS_SEARCH_URL,
                    params={
                        "query": title,
                        "fields": SS_FIELDS,
                        "limit": MAX_RESULTS_PER_QUERY,
                    },
                )
            except (httpx.TimeoutException, httpx.ConnectError):
                raise  # tenacity retries
            except Exception as e:
                logger.warning("Semantic Scholar unexpected error: %s", e)
                return []

        if response.status_code == 429:
            wait_time = int(response.headers.get("Retry-After", 15))
            await asyncio.sleep(wait_time)
            raise httpx.HTTPStatusError(
                "429 rate limited", request=response.request, response=response
            )

        if response.status_code == 200:
            data = response.json()
            return data.get("data", [])

        logger.warning(
            "Semantic Scholar returned %d for query: %s",
            response.status_code,
            title[:60],
        )
        return []

    def _build_source_result(
        self,
        ref: ReferenceResult,
        candidate: dict | None,
    ) -> SourceResult:
        if candidate is None:
            return SourceResult(
                source=VerificationSource.SEMANTIC_SCHOLAR,
                found=False,
            )

        title_similarity = candidate.get("_title_similarity")

        # Author match
        ss_authors = candidate.get("authors", [])
        author_match = _check_author_match(ref.authors, ss_authors)

        # Year delta
        year_delta: int | None = None
        ss_year = candidate.get("year")
        if ref.year and ss_year:
            year_delta = abs(ref.year - ss_year)

        # URL
        matched_url = _extract_url(candidate)

        return SourceResult(
            source=VerificationSource.SEMANTIC_SCHOLAR,
            found=True,
            title_similarity=title_similarity,
            author_match=author_match,
            year_delta=year_delta,
            matched_title=candidate.get("title"),
            matched_url=matched_url,
        )

    async def verify(self, ref: ReferenceResult) -> VerificationResult:
        """Verify a single reference via title search on Semantic Scholar."""
        if not ref.title:
            logger.warning(
                "SemanticScholarVerifier: no title for ref: %s",
                ref.raw_reference[:60],
            )
            return VerificationResult(
                reference=ref,
                sources_checked=[VerificationSource.SEMANTIC_SCHOLAR],
                source_results=[
                    SourceResult(
                        source=VerificationSource.SEMANTIC_SCHOLAR,
                        found=False,
                    )
                ],
            )

        candidates = await self._search(ref.title)
        best = _best_match(ref, candidates)
        source_result = self._build_source_result(ref, best)

        logger.info(
            "SemanticScholar [%s] found=%s title_sim=%s author=%s",
            (ref.title or "?")[:50],
            source_result.found,
            (
                f"{source_result.title_similarity:.2f}"
                if source_result.title_similarity
                else "N/A"
            ),
            source_result.author_match,
        )

        return VerificationResult(
            reference=ref,
            sources_checked=[VerificationSource.SEMANTIC_SCHOLAR],
            source_results=[source_result],
        )

    async def verify_batch(
        self,
        refs: list[ReferenceResult],
    ) -> list[VerificationResult]:
        """Verify multiple references concurrently."""
        tasks = [self.verify(ref) for ref in refs]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

semantic_scholar_verifier = SemanticScholarVerifier()
