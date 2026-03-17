"""
Crossref verifier.

Given a ReferenceResult with a DOI, queries the Crossref API and returns
a VerificationResult with title similarity, author match, year delta,
and retraction flag.

API used:
    GET https://api.crossref.org/works/{doi}
    No key required. Email in User-Agent header → polite pool (higher limits).
"""

import asyncio
import re

import httpx
from rapidfuzz import fuzz
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.core.config import config
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

CROSSREF_BASE = "https://api.crossref.org/works"
TIMEOUT = httpx.Timeout(10.0, connect=5.0)
MAX_CONCURRENT = 10  # polite pool allows ~50/sec — we stay conservative
TITLE_MATCH_THRESHOLD = 0.85  # minimum similarity to consider a title match


# ---------------------------------------------------------------------------
# Text normalization helpers
# ---------------------------------------------------------------------------


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation and extra whitespace."""
    title = title.lower()
    title = re.sub(r"[^\w\s]", " ", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def _extract_lastname(author: str) -> str:
    """
    Extract last name from any common format:
      'Ashish Vaswani'   → 'vaswani'
      'A. Vaswani'       → 'vaswani'
      'Vaswani, A.'      → 'vaswani'
      'Vaswani'          → 'vaswani'
    """
    author = author.strip().rstrip(".")
    if "," in author:
        # 'Lastname, F.' format
        return author.split(",")[0].strip().lower()
    parts = author.split()
    return parts[-1].lower() if parts else author.lower()


def _check_author_match(
    cited_authors: list[str] | None,
    crossref_authors: list[dict],
) -> bool | None:
    """
    Returns True if at least one cited last name appears in Crossref authors.
    Returns None if we have no author data to compare.
    """
    if not cited_authors or not crossref_authors:
        return None

    cited_lastnames = {_extract_lastname(a) for a in cited_authors}

    crossref_lastnames = set()
    for author in crossref_authors:
        family = author.get("family", "")
        if family:
            crossref_lastnames.add(family.lower())

    if not crossref_lastnames:
        return None

    return bool(cited_lastnames & crossref_lastnames)


def _check_retraction(data: dict) -> bool:
    """
    Check Crossref retraction signals (Retraction Watch integrated Jan 2025).
    Looks at 'relation' field and 'update-to' field.
    """
    relation = data.get("relation", {})
    if relation.get("is-retracted-by"):
        return True

    for update in data.get("update-to", []):
        if update.get("type") == "retraction":
            return True

    return False


# ---------------------------------------------------------------------------
# CrossrefVerifier
# ---------------------------------------------------------------------------


class CrossrefVerifier:
    """
    Async Crossref verifier. Create one instance and reuse it —
    it manages a shared httpx client and semaphore.
    """

    def __init__(self, mailto: str | None = None):
        self.mailto = (
            mailto
            or getattr(config, "crossref_mailto", None)
            or "citeguard@example.com"
        )
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT,
                headers={"User-Agent": f"citeguard/1.0 (mailto:{self.mailto})"},
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
    async def _fetch(self, doi: str) -> dict | None:
        """
        Fetch a single DOI from Crossref.
        Returns the 'message' dict or None if not found.
        """
        url = f"{CROSSREF_BASE}/{doi}"
        client = await self._get_client()

        async with self._semaphore:
            try:
                response = await client.get(url)
            except (httpx.TimeoutException, httpx.ConnectError):
                raise  # tenacity will retry
            except Exception as e:
                logger.warning("Crossref unexpected error for DOI %s: %s", doi, e)
                return None

        if response.status_code == 404:
            logger.debug("Crossref 404 for DOI: %s", doi)
            return None

        if response.status_code == 429:
            logger.warning("Crossref rate limited — backing off")
            await asyncio.sleep(2)
            return None

        if response.status_code != 200:
            logger.warning(
                "Crossref returned %d for DOI: %s", response.status_code, doi
            )
            return None

        body = response.json()
        return body.get("message")

    def _build_source_result(
        self,
        ref: ReferenceResult,
        message: dict | None,
    ) -> SourceResult:
        """
        Compare Crossref message against cited reference fields.
        Returns a SourceResult with all signals populated.
        """
        if message is None:
            return SourceResult(
                source=VerificationSource.CROSSREF,
                found=False,
            )

        # --- Title similarity ---
        crossref_titles = message.get("title", [])
        crossref_title = crossref_titles[0] if crossref_titles else ""
        title_similarity: float | None = None

        if ref.title and crossref_title:
            norm_cited = _normalize_title(ref.title)
            norm_found = _normalize_title(crossref_title)
            title_similarity = fuzz.ratio(norm_cited, norm_found) / 100.0

        # --- Author match ---
        crossref_authors = message.get("author", [])
        author_match = _check_author_match(ref.authors, crossref_authors)

        # --- Year delta ---
        year_delta: int | None = None
        date_parts = message.get("published", {}).get("date-parts", [[]])[0]
        if date_parts and ref.year:
            crossref_year = date_parts[0]
            year_delta = abs(ref.year - crossref_year)

        # --- Retraction ---
        retracted = _check_retraction(message)

        # --- Matched URL ---
        matched_url = message.get("URL") or f"https://doi.org/{ref.doi}"

        return SourceResult(
            source=VerificationSource.CROSSREF,
            found=True,
            title_similarity=title_similarity,
            author_match=author_match,
            year_delta=year_delta,
            retracted=retracted,
            matched_title=crossref_title or None,
            matched_url=matched_url,
        )

    async def verify(self, ref: ReferenceResult) -> VerificationResult:
        """
        Verify a single reference against Crossref.
        ref.doi must be present.
        """
        if not ref.doi:
            raise ValueError(
                f"CrossrefVerifier.verify called with no DOI: {ref.raw_reference}"
            )

        message = await self._fetch(ref.doi)
        source_result = self._build_source_result(ref, message)

        logger.info(
            "Crossref [%s] doi=%s found=%s title_sim=%.2f author=%s retracted=%s",
            ref.title[:50] if ref.title else "?",
            ref.doi,
            source_result.found,
            source_result.title_similarity or 0.0,
            source_result.author_match,
            source_result.retracted,
        )

        return VerificationResult(
            reference=ref,
            sources_checked=[VerificationSource.CROSSREF],
            source_results=[source_result],
        )

    async def verify_batch(
        self,
        refs: list[ReferenceResult],
    ) -> list[VerificationResult]:
        """
        Verify multiple references concurrently.
        Semaphore inside _fetch controls actual concurrency.
        """
        tasks = [self.verify(ref) for ref in refs]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

crossref_verifier = CrossrefVerifier()
