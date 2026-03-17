"""
OpenAlex verifier.

Given a ReferenceResult with a title, searches OpenAlex and returns
a VerificationResult with title similarity, author match, and year delta.

API used:
    GET https://api.openalex.org/works?search={title}&mailto={email}
    No key required. Email in mailto param → polite pool (10 req/sec, 100k/day).
    Set OPENALEX_MAILTO in .env

Docs: https://docs.openalex.org/
"""

import asyncio
import os
import re

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

OPENALEX_BASE = "https://api.openalex.org/works"
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
TITLE_MATCH_THRESHOLD = 0.85
MAX_CONCURRENT = 10  # polite pool allows this comfortably
MAX_RESULTS_PER_QUERY = 10


def _get_mailto() -> str:
    return os.getenv("OPENALEX_MAILTO", "citeguard@example.com")


# ---------------------------------------------------------------------------
# Helpers
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
    oa_authorships: list[dict],
) -> bool | None:
    """
    OpenAlex returns authors under 'authorships' key, each with
    'author': {'display_name': 'Firstname Lastname'}.
    Returns True if at least one cited last name matches.
    Returns None if no usable data on either side.
    """
    if not oa_authorships or not cited_authors:
        return None

    real_cited = [a for a in cited_authors if "et al" not in a.lower()]
    if not real_cited:
        return None

    cited_lastnames = {_extract_lastname(a) for a in real_cited}

    oa_lastnames = set()
    for authorship in oa_authorships:
        author = authorship.get("author", {})
        name = author.get("display_name", "")
        if name:
            oa_lastnames.add(_extract_lastname(name))

    if not oa_lastnames:
        return None

    return bool(cited_lastnames & oa_lastnames)


def _best_match(
    ref: ReferenceResult,
    candidates: list[dict],
) -> dict | None:
    if not candidates or not ref.title:
        return None

    norm_cited = _normalize_title(ref.title)

    # Step 1: collect all candidates above title threshold
    matches = []
    for candidate in candidates:
        title = candidate.get("title", "") or ""
        if not title:
            continue
        score = fuzz.ratio(_normalize_title(title), norm_cited) / 100.0
        if score >= TITLE_MATCH_THRESHOLD:
            candidate["_title_similarity"] = score
            matches.append(candidate)

    if not matches:
        return None

    # Step 2: among title matches, prefer the one closest to cited year
    if ref.year:
        matches.sort(key=lambda c: abs(ref.year - (c.get("publication_year") or 0)))

    return matches[0]


def _extract_url(candidate: dict) -> str | None:
    """
    Extract best available URL from an OpenAlex work record.
    Priority: open access URL → DOI → landing page URL
    """
    # Open access best URL
    oa = candidate.get("open_access", {})
    oa_url = oa.get("oa_url")
    if oa_url:
        return oa_url

    # DOI
    doi = candidate.get("doi")
    if doi:
        # OA returns full DOI URL already: "https://doi.org/10.xxx/xxx"
        return doi

    # Landing page
    return candidate.get("primary_location", {}).get("landing_page_url")


# ---------------------------------------------------------------------------
# OpenAlexVerifier
# ---------------------------------------------------------------------------


class OpenAlexVerifier:
    """
    Async OpenAlex verifier. Create one instance and reuse it.
    """

    def __init__(self):
        self._mailto = _get_mailto()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": f"citeguard/1.0 (mailto:{self._mailto})"},
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
    async def _query(self, params: dict) -> list[dict]:
        """Single HTTP call to OpenAlex with given params."""
        client = await self._get_client()

        async with self._semaphore:
            try:
                response = await client.get(
                    OPENALEX_BASE,
                    params={
                        **params,
                        "per_page": MAX_RESULTS_PER_QUERY,
                        "mailto": self._mailto,
                        "select": "title,authorships,publication_year,doi,open_access,primary_location",
                    },
                )
            except (httpx.TimeoutException, httpx.ConnectError):
                raise
            except Exception as e:
                logger.warning("OpenAlex unexpected error: %s", e)
                return []

        if response.status_code == 429:
            logger.warning("OpenAlex rate limited — backing off")
            await asyncio.sleep(5)
            return []

        if response.status_code == 200:
            return response.json().get("results", [])

        logger.warning("OpenAlex returned %d", response.status_code)
        return []

    # async def _search(self, title: str, authors: list[str] | None = None) -> list[dict]:
    #     # Build richer query with first author lastname if available
    #     query = title
    #     if authors:
    #         real_authors = [a for a in authors if "et al" not in a.lower()]
    #         if real_authors:
    #             lastname = _extract_lastname(real_authors[0])
    #             query = f"{title} {lastname}"

    #     search_title = query.split(":")[0].strip()

    #     results = await self._query({"filter": f"title.search:{search_title}"})
    #     if results:
    #         return results
    #     return await self._query({"search": query})
    async def _search(self, title: str, authors: list[str] | None = None) -> list[dict]:
        # Never add author to query — diagnostic proved it breaks ranking
        # Authors are used only for post-retrieval matching in _check_author_match
        search_title = title.split(":")[0].strip()

        results = await self._query({"filter": f"title.search:{search_title}"})
        if results:
            return results
        return await self._query({"search": title})

    def _build_source_result(
        self,
        ref: ReferenceResult,
        candidate: dict | None,
    ) -> SourceResult:
        if candidate is None:
            return SourceResult(
                source=VerificationSource.OPENALEX,
                found=False,
            )

        title_similarity = candidate.get("_title_similarity")

        # Author match — OpenAlex uses 'authorships'
        authorships = candidate.get("authorships", [])
        author_match = _check_author_match(ref.authors, authorships)

        # Year delta — OpenAlex uses 'publication_year'
        year_delta: int | None = None
        oa_year = candidate.get("publication_year")
        if ref.year and oa_year:
            year_delta = abs(ref.year - oa_year)

        matched_url = _extract_url(candidate)

        return SourceResult(
            source=VerificationSource.OPENALEX,
            found=True,
            title_similarity=title_similarity,
            author_match=author_match,
            year_delta=year_delta,
            matched_title=candidate.get("title"),
            matched_url=matched_url,
        )

    async def verify(self, ref: ReferenceResult) -> VerificationResult:
        """Verify a single reference via title search on OpenAlex."""
        if not ref.title:
            logger.warning(
                "OpenAlexVerifier: no title for ref: %s",
                ref.raw_reference[:60],
            )
            return VerificationResult(
                reference=ref,
                sources_checked=[VerificationSource.OPENALEX],
                source_results=[
                    SourceResult(
                        source=VerificationSource.OPENALEX,
                        found=False,
                    )
                ],
            )

        candidates = await self._search(ref.title, ref.authors)
        best = _best_match(ref, candidates)
        source_result = self._build_source_result(ref, best)

        logger.info(
            "OpenAlex [%s] found=%s title_sim=%s author=%s",
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
            sources_checked=[VerificationSource.OPENALEX],
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

openalex_verifier = OpenAlexVerifier()
