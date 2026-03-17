"""
arXiv verifier.

Given a ReferenceResult with an arxiv_id, queries the arXiv API and returns
a VerificationResult with title similarity, author match, and year delta.

Two verification paths:
  Path A — arxiv_id present: fetch directly by ID (authoritative)
  Path B — no arxiv_id but ref landed here via 10.48550 DOI promotion:
            arxiv_id was extracted by classify_references_node, so Path A always applies

API used:
    GET http://export.arxiv.org/api/query?id_list={id}
    No key required. Rate limit: 3 req/sec — enforced via semaphore.
"""

import asyncio
import re
import xml.etree.ElementTree as ET

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

ARXIV_API_BASE = "https://export.arxiv.org/api/query"
ARXIV_NS = "http://www.w3.org/2005/Atom"
TIMEOUT = httpx.Timeout(15.0, connect=5.0)
MAX_CONCURRENT = 3  # arXiv asks for max 3 req/sec
TITLE_MATCH_THRESHOLD = 0.85


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_arxiv_id(arxiv_id: str) -> str:
    """
    Strip version suffix and any prefix.
    '1706.03762v2' → '1706.03762'
    'arXiv:1706.03762' → '1706.03762'
    """
    arxiv_id = arxiv_id.strip()
    arxiv_id = re.sub(r"(?i)^arxiv:", "", arxiv_id)
    arxiv_id = re.sub(r"v\d+$", "", arxiv_id)
    return arxiv_id


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


def _parse_arxiv_response(xml_text: str) -> dict | None:
    """
    Parse arXiv Atom XML response.
    Returns a dict with title, authors, year or None if no entry found.
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as e:
        logger.warning("arXiv XML parse error: %s", e)
        return None

    ns = {"atom": ARXIV_NS}
    entries = root.findall("atom:entry", ns)

    if not entries:
        return None

    entry = entries[0]

    # Check for arXiv error entry
    title_el = entry.find("atom:title", ns)
    if title_el is not None and title_el.text and "Error" in title_el.text:
        return None

    title = title_el.text.strip() if title_el is not None and title_el.text else ""

    # Authors
    authors = []
    for author_el in entry.findall("atom:author", ns):
        name_el = author_el.find("atom:name", ns)
        if name_el is not None and name_el.text:
            authors.append(name_el.text.strip())

    # Published year
    published_el = entry.find("atom:published", ns)
    year = None
    if published_el is not None and published_el.text:
        try:
            year = int(published_el.text[:4])
        except ValueError:
            pass

    # arXiv abstract URL
    url = None
    for link_el in entry.findall("atom:link", ns):
        if link_el.get("type") == "text/html":
            url = link_el.get("href")
            break

    return {
        "title": title,
        "authors": authors,  # list of "Firstname Lastname" strings
        "year": year,
        "url": url,
    }


def _check_author_match(
    cited_authors: list[str] | None,
    arxiv_authors: list[str],
) -> bool | None:
    """
    Returns True if at least one cited last name appears in arXiv authors.
    Returns None if no author data available for comparison.
    """
    if not cited_authors or not arxiv_authors:
        return None

    # Handle "et al." — if cited authors only has et al. entries, skip match
    real_cited = [a for a in cited_authors if "et al" not in a.lower()]
    if not real_cited:
        return None

    cited_lastnames = {_extract_lastname(a) for a in real_cited}
    arxiv_lastnames = {_extract_lastname(a) for a in arxiv_authors}

    return bool(cited_lastnames & arxiv_lastnames)


# ---------------------------------------------------------------------------
# ArxivVerifier
# ---------------------------------------------------------------------------


class ArxivVerifier:
    """
    Async arXiv verifier. Create one instance and reuse it.
    """

    def __init__(self):
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=TIMEOUT,
                follow_redirects=True,
                headers={"User-Agent": "citeguard/1.0 (academic reference verifier)"},
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
    async def _fetch(self, arxiv_id: str) -> dict | None:
        """
        Fetch a single arXiv entry by ID.
        Returns parsed dict or None if not found.
        """
        clean_id = _normalize_arxiv_id(arxiv_id)
        client = await self._get_client()

        async with self._semaphore:
            try:
                response = await client.get(
                    ARXIV_API_BASE,
                    params={"id_list": clean_id, "max_results": 1},
                )
            except (httpx.TimeoutException, httpx.ConnectError):
                raise  # tenacity retries
            except Exception as e:
                logger.warning("arXiv unexpected error for ID %s: %s", arxiv_id, e)
                return None

        if response.status_code != 200:
            logger.warning(
                "arXiv returned %d for ID: %s", response.status_code, arxiv_id
            )
            return None

        return _parse_arxiv_response(response.text)

    def _build_source_result(
        self,
        ref: ReferenceResult,
        parsed: dict | None,
    ) -> SourceResult:
        if parsed is None:
            return SourceResult(
                source=VerificationSource.ARXIV,
                found=False,
            )

        # Title similarity
        title_similarity: float | None = None
        if ref.title and parsed["title"]:
            norm_cited = _normalize_title(ref.title)
            norm_found = _normalize_title(parsed["title"])
            title_similarity = fuzz.ratio(norm_cited, norm_found) / 100.0

        # Author match
        author_match = _check_author_match(ref.authors, parsed["authors"])

        # Year delta
        year_delta: int | None = None
        if ref.year and parsed["year"]:
            year_delta = abs(ref.year - parsed["year"])

        return SourceResult(
            source=VerificationSource.ARXIV,
            found=True,
            title_similarity=title_similarity,
            author_match=author_match,
            year_delta=year_delta,
            matched_title=parsed["title"] or None,
            matched_url=parsed["url"],
        )

    async def verify(self, ref: ReferenceResult) -> VerificationResult:
        """Verify a single reference against arXiv."""
        if not ref.arxiv_id:
            raise ValueError(
                f"ArxivVerifier.verify called with no arxiv_id: {ref.raw_reference}"
            )

        parsed = await self._fetch(ref.arxiv_id)
        source_result = self._build_source_result(ref, parsed)

        logger.info(
            "arXiv [%s] id=%s found=%s title_sim=%s author=%s",
            (ref.title or "?")[:50],
            ref.arxiv_id,
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
            sources_checked=[VerificationSource.ARXIV],
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

arxiv_verifier = ArxivVerifier()
