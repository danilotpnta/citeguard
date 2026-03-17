"""
DBLP verifier.

Queries a local SQLite FTS5 database built from the DBLP XML dump.
Used as a parallel search source alongside OpenAlex for CS conference
papers that have no DOI and no arXiv ID.

Requires:
    DBLP_DB_PATH set in .env AND the database built via:
    uv run python scripts/build_dblp_index.py

No network calls — queries are local.
"""

import re
import sqlite3
from pathlib import Path

from rapidfuzz import fuzz

from app.core.config import settings
from app.core.logging import get_logger
from app.models.schemas import (
    ReferenceResult,
    SourceResult,
    VerificationResult,
    VerificationSource,
)

logger = get_logger(__name__)

TITLE_MATCH_THRESHOLD = 0.85


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


def _build_fts_query(title: str) -> str:
    """
    Build a safe FTS5 query from a title string.
    Strips FTS5 special characters and takes the first 8 significant words
    to avoid overly-specific queries that return nothing.
    """
    # Remove FTS5 special chars
    # clean = re.sub(r'["\(\)\*\:\^\-]', " ", title)
    clean = re.sub(r'["\(\)\*\:\^\-\#\!\.\,\?\/\\]', " ", title)
    clean = re.sub(r"\s+", " ", clean).strip().lower()

    # Take first 8 words — enough to be specific, not so many that FTS fails
    words = clean.split()[:8]
    return " ".join(words)


def _check_author_match(
    cited_authors: list[str] | None,
    dblp_authors_str: str,
) -> bool | None:
    """
    DBLP stores authors as pipe-separated string: "Author A | Author B | ..."
    Returns True if at least one cited last name appears in DBLP authors.
    Returns None if no usable data.
    """
    if not cited_authors or not dblp_authors_str:
        return None

    real_cited = [a for a in cited_authors if "et al" not in a.lower()]
    if not real_cited:
        return None

    cited_lastnames = {_extract_lastname(a) for a in real_cited}

    dblp_lastnames = set()
    for author in dblp_authors_str.split("|"):
        author = author.strip()
        if author:
            dblp_lastnames.add(_extract_lastname(author))

    if not dblp_lastnames:
        return None

    return bool(cited_lastnames & dblp_lastnames)


# ---------------------------------------------------------------------------
# DBLPVerifier
# ---------------------------------------------------------------------------


class DBLPVerifier:
    """
    Local DBLP verifier. Queries an SQLite FTS5 database.
    Instantiation is cheap — the database connection is opened per query.
    """

    def __init__(self, db_path: str | None = None):
        self._db_path = Path(db_path or settings.dblp_db_path)
        self._available: bool | None = None  # cached after first check

    @property
    def available(self) -> bool:
        if self._available is None:
            self._available = self._db_path.exists()
            if not self._available:
                logger.info(
                    "DBLP database not found at %s — DBLP verification disabled. "
                    "Build it with: uv run python scripts/build_dblp_index.py",
                    self._db_path,
                )
        return self._available

    def _search(self, title: str, limit: int = 10) -> list[dict]:
        """
        Full-text search against the local DBLP database.
        Returns list of raw row dicts.
        """
        if not self.available:
            return []

        fts_query = _build_fts_query(title)
        if not fts_query:
            return []

        try:
            conn = sqlite3.connect(str(self._db_path))
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            cursor.execute(
                """
                SELECT dblp_key, title, authors, venue, year, url
                FROM papers
                WHERE papers MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            )
            rows = [dict(row) for row in cursor.fetchall()]
            conn.close()
            return rows

        except sqlite3.Error as e:
            logger.warning("DBLP query error: %s", e)
            return []

    def _best_match(
        self,
        ref: ReferenceResult,
        candidates: list[dict],
    ) -> dict | None:
        """
        Find the best matching candidate by title similarity.
        Applies year proximity preference among equally good title matches.
        """
        if not candidates or not ref.title:
            return None

        norm_cited = _normalize_title(ref.title)
        matches = []

        for candidate in candidates:
            dblp_title = candidate.get("title", "") or ""
            if not dblp_title:
                continue

            score = fuzz.ratio(_normalize_title(dblp_title), norm_cited) / 100.0
            if score >= TITLE_MATCH_THRESHOLD:
                candidate["_title_similarity"] = score
                matches.append(candidate)

        if not matches:
            return None

        # Prefer candidate whose year is closest to cited year
        if ref.year:
            matches.sort(
                key=lambda c: (
                    abs(ref.year - int(c.get("year") or 0)) if c.get("year") else 999
                )
            )

        return matches[0]

    def _build_source_result(
        self,
        ref: ReferenceResult,
        candidate: dict | None,
    ) -> SourceResult:
        if candidate is None:
            return SourceResult(
                source=VerificationSource.DBLP,
                found=False,
            )

        title_similarity = candidate.get("_title_similarity")
        author_match = _check_author_match(ref.authors, candidate.get("authors", ""))

        year_delta: int | None = None
        dblp_year = candidate.get("year")
        if ref.year and dblp_year:
            try:
                year_delta = abs(ref.year - int(dblp_year))
            except (ValueError, TypeError):
                pass

        # Build matched URL — prefer ee link, fall back to DBLP page
        dblp_key = candidate.get("dblp_key", "")
        matched_url = candidate.get("url") or (
            f"https://dblp.org/rec/{dblp_key}" if dblp_key else None
        )

        return SourceResult(
            source=VerificationSource.DBLP,
            found=True,
            title_similarity=title_similarity,
            author_match=author_match,
            year_delta=year_delta,
            matched_title=candidate.get("title"),
            matched_url=matched_url,
        )

    def verify(self, ref: ReferenceResult) -> VerificationResult:
        """
        Verify a single reference against the local DBLP database.
        Synchronous — no network calls.
        """
        if not self.available:
            return VerificationResult(
                reference=ref,
                sources_checked=[],
                source_results=[],
            )

        if not ref.title:
            return VerificationResult(
                reference=ref,
                sources_checked=[VerificationSource.DBLP],
                source_results=[
                    SourceResult(source=VerificationSource.DBLP, found=False)
                ],
            )

        candidates = self._search(ref.title)
        best = self._best_match(ref, candidates)
        source_result = self._build_source_result(ref, best)

        logger.debug(
            "DBLP [%s] found=%s title_sim=%s author=%s",
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
            sources_checked=[VerificationSource.DBLP],
            source_results=[source_result],
        )

    def verify_batch(
        self,
        refs: list[ReferenceResult],
    ) -> list[VerificationResult]:
        """
        Verify multiple references. Synchronous — local DB queries are fast.
        No async needed here since there's no network I/O.
        """
        return [self.verify(ref) for ref in refs]


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

dblp_verifier = DBLPVerifier()
