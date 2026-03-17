"""
OpenLibrary verifier.

Given a ReferenceResult, searches OpenLibrary and returns a VerificationResult.
Primary use case: books, technical reports, and grey literature that arXiv,
Crossref, OpenAlex, and DBLP don't index.

Two search strategies (tried in order):
  1. title + first author lastname  (more specific)
  2. title only                     (fallback)

If ref.isbn is present, tries direct ISBN lookup first — highest confidence.

API used:
    GET https://openlibrary.org/search.json
    GET https://openlibrary.org/isbn/{isbn}.json
    No key required. Add User-Agent with contact email for polite access.

Docs: https://openlibrary.org/developers/api
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

OL_SEARCH_URL = "https://openlibrary.org/search.json"
OL_ISBN_URL = "https://openlibrary.org/isbn/{isbn}.json"
OL_WORK_BASE = "https://openlibrary.org"

TIMEOUT = httpx.Timeout(15.0, connect=5.0)
TITLE_MATCH_THRESHOLD = 0.85
MAX_CONCURRENT = 5  # conservative — no documented rate limit
MAX_RESULTS = 10

SEARCH_FIELDS = "key,title,author_name,first_publish_year,publisher,isbn"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    from app.agents.tools.parser.extractor import LIGATURE_MAP

    title = title.translate(LIGATURE_MAP)
    title = re.sub(r"(\w)- ([a-z])", r"\1\2", title)  # line-break hyphen artifacts
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _extract_lastname(author: str) -> str:
    author = re.sub(r"\s+et\.?\s+al\.?.*", "", author, flags=re.IGNORECASE).strip()
    author = author.rstrip(".")
    if "," in author:
        return author.split(",")[0].strip().lower()
    parts = author.split()
    return parts[-1].lower() if parts else author.lower()


def _check_author_match(
    cited_authors: list[str] | None,
    ol_author_names: list[str] | None,
) -> bool | None:
    """
    OpenLibrary returns author_name as a flat list of strings.
    Returns True if at least one cited lastname matches.
    Returns None if et al. or no usable data on either side.
    """
    if not ol_author_names or not cited_authors:
        return None

    has_et_al = any("et al" in a.lower() for a in cited_authors)
    real_cited = [a for a in cited_authors if "et al" not in a.lower()]

    if not real_cited:
        return None

    cited_lastnames = {_extract_lastname(a) for a in real_cited}
    ol_lastnames = {_extract_lastname(n) for n in ol_author_names if n}

    if not ol_lastnames:
        return None

    match = bool(cited_lastnames & ol_lastnames)

    if has_et_al:
        # Only first author was listed — if it matches, return None (partial)
        # rather than True, to signal we couldn't confirm the full list
        first_lastname = _extract_lastname(real_cited[0])
        if first_lastname in ol_lastnames:
            return None
        return False

    return match


def _best_match(
    ref: ReferenceResult,
    candidates: list[dict],
) -> dict | None:
    if not candidates or not ref.title:
        return None

    norm_cited = _normalize_title(ref.title)
    matches = []

    for candidate in candidates:
        title = candidate.get("title") or ""
        if not title:
            continue
        score = fuzz.ratio(_normalize_title(title), norm_cited) / 100.0
        if score >= TITLE_MATCH_THRESHOLD:
            candidate["_title_similarity"] = score
            matches.append(candidate)

    if not matches:
        return None

    # Among title matches, prefer the one closest to cited year
    if ref.year:
        matches.sort(
            key=lambda c: abs(ref.year - (c.get("first_publish_year") or 0))
        )

    return matches[0]


def _build_source_result(
    ref: ReferenceResult,
    candidate: dict | None,
    *,
    found: bool = False,
) -> SourceResult:
    if candidate is None or not found:
        return SourceResult(
            source=VerificationSource.OPENLIBRARY,
            found=False,
        )

    title_similarity = candidate.get("_title_similarity")
    author_match = _check_author_match(
        ref.authors, candidate.get("author_name")
    )

    year_delta: int | None = None
    ol_year = candidate.get("first_publish_year")
    if ref.year and ol_year:
        year_delta = abs(ref.year - ol_year)

    work_key = candidate.get("key", "")
    matched_url = f"{OL_WORK_BASE}{work_key}" if work_key else None

    return SourceResult(
        source=VerificationSource.OPENLIBRARY,
        found=True,
        title_similarity=title_similarity,
        author_match=author_match,
        year_delta=year_delta,
        matched_title=candidate.get("title"),
        matched_url=matched_url,
    )


# ---------------------------------------------------------------------------
# OpenLibraryVerifier
# ---------------------------------------------------------------------------


class OpenLibraryVerifier:
    """
    Async OpenLibrary verifier. Create one instance and reuse it.
    Mirrors the pattern of OpenAlexVerifier.
    """

    def __init__(self):
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers={
                    "User-Agent": "citeguard/1.0 (mailto:citeguard@example.com)"
                },
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
    async def _get(self, url: str, params: dict | None = None) -> dict | None:
        client = await self._get_client()
        async with self._semaphore:
            try:
                response = await client.get(url, params=params)
            except (httpx.TimeoutException, httpx.ConnectError):
                raise
            except Exception as e:
                logger.warning("OpenLibrary unexpected error: %s", e)
                return None

        if response.status_code == 429:
            logger.warning("OpenLibrary rate limited — backing off")
            await asyncio.sleep(5)
            return None

        if response.status_code == 200:
            return response.json()

        if response.status_code != 404:
            logger.warning("OpenLibrary returned %d", response.status_code)
        return None

    async def _lookup_isbn(self, isbn: str) -> dict | None:
        """
        Direct ISBN lookup — highest confidence path.
        Returns a normalised candidate dict or None.
        """
        clean_isbn = re.sub(r"[\s\-]", "", isbn)
        data = await self._get(OL_ISBN_URL.format(isbn=clean_isbn))
        if not data:
            return None

        # ISBN endpoint returns edition data — normalise to search-like shape
        return {
            "title": data.get("title"),
            "author_name": [
                a.get("key", "").split("/")[-1]  # fallback — OL author key
                for a in data.get("authors", [])
            ],
            "first_publish_year": _parse_year(data.get("publish_date", "")),
            "key": data.get("works", [{}])[0].get("key", ""),
        }

    async def _search(
        self, title: str, authors: list[str] | None = None
    ) -> list[dict]:
        """
        Two-pass search:
          1. title + first author lastname
          2. title only (fallback)
        """
        norm_title = _normalize_title(title)
        if not norm_title:
            return []

        params: dict = {
            "fields": SEARCH_FIELDS,
            "limit": MAX_RESULTS,
        }

        # Pass 1 — title + author
        if authors:
            real_authors = [a for a in authors if "et al" not in a.lower()]
            if real_authors:
                lastname = _extract_lastname(real_authors[0])
                params["title"] = title
                params["author"] = lastname
                data = await self._get(OL_SEARCH_URL, params=params)
                results = (data or {}).get("docs", [])
                if results:
                    return results

        # Pass 2 — title only
        params.pop("author", None)
        params["title"] = title
        data = await self._get(OL_SEARCH_URL, params=params)
        return (data or {}).get("docs", [])

    async def verify(self, ref: ReferenceResult) -> VerificationResult:
        """Verify a single reference via OpenLibrary."""

        def _not_found() -> VerificationResult:
            return VerificationResult(
                reference=ref,
                sources_checked=[VerificationSource.OPENLIBRARY],
                source_results=[
                    SourceResult(source=VerificationSource.OPENLIBRARY, found=False)
                ],
            )

        if not ref.title:
            logger.warning(
                "OpenLibraryVerifier: no title for ref: %s",
                ref.raw_reference[:60],
            )
            return _not_found()

        candidate: dict | None = None

        # Fast path — ISBN lookup
        isbn = getattr(ref, "isbn", None)
        if isbn:
            candidate = await self._lookup_isbn(isbn)
            if candidate:
                candidate["_title_similarity"] = (
                    fuzz.ratio(
                        _normalize_title(candidate.get("title") or ""),
                        _normalize_title(ref.title),
                    )
                    / 100.0
                )

        # Slow path — title search
        if candidate is None:
            results = await self._search(ref.title, ref.authors)
            candidate = _best_match(ref, results)

        found = candidate is not None
        source_result = _build_source_result(ref, candidate, found=found)

        logger.debug(
            "OpenLibrary [%s] found=%s title_sim=%s author=%s",
            (ref.title or "?")[:50],
            source_result.found,
            (
                f"{source_result.title_similarity:.2f}"
                if source_result.title_similarity is not None
                else "N/A"
            ),
            source_result.author_match,
        )

        return VerificationResult(
            reference=ref,
            sources_checked=[VerificationSource.OPENLIBRARY],
            source_results=[source_result],
        )

    async def verify_batch(
        self, refs: list[ReferenceResult]
    ) -> list[VerificationResult]:
        tasks = [self.verify(ref) for ref in refs]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_year(publish_date: str) -> int | None:
    """Extract a 4-digit year from strings like '2006', 'January 2006', '2006-01'."""
    m = re.search(r"\b(1[89]\d{2}|20\d{2})\b", publish_date)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

openlibrary_verifier = OpenLibraryVerifier()