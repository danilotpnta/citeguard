"""
merge_results_node  — combines doi_results, arxiv_results, search_results
                      into a single deduplicated list, merging source signals
                      for references that went through multiple paths.

score_node          — applies rule-based verdict logic to each merged result.
                      No numeric scoring — explicit rules based on signal combinations.

Verdicts:
    VERIFIED            — found by identifier or found by search with author confirmation
    LIKELY_REAL         — found by search, title matches, but author unknown (et al.)
    NEEDS_REVIEW        — found but metadata mismatch (title or author conflict)
    UNVERIFIABLE        — not found but reference has a URL or is clearly non-academic
    LIKELY_HALLUCINATED — not found anywhere, looks like an academic paper
    RETRACTED           — found but flagged as retracted (overrides all)
"""

from langfuse import observe

from app.graph.state import WorkflowState
from app.models.schemas import (
    ReferenceResult,
    VerificationResult,
    SourceResult,
    VerificationSource,
)
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Verdict constants
# ---------------------------------------------------------------------------

VERIFIED = "VERIFIED"
LIKELY_REAL = "LIKELY_REAL"
NEEDS_REVIEW = "NEEDS_REVIEW"
UNVERIFIABLE = "UNVERIFIABLE"
LIKELY_HALLUCINATED = "LIKELY_HALLUCINATED"
RETRACTED = "RETRACTED"

TITLE_THRESHOLD = 0.85

# Venue/publisher keywords that suggest non-academic sources
NON_ACADEMIC_SIGNALS = {
    "blog",
    "github",
    "openai",
    "google",
    "microsoft",
    "anthropic",
    "mit press",
    "oxford university press",
    "cambridge university press",
    "springer",
    "wiley",
    "elsevier",  # books — verifiable by ISBN
    "accessed",
    "retrieved",  # web resources
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_identifier_source(source: VerificationSource) -> bool:
    return source in (VerificationSource.CROSSREF, VerificationSource.ARXIV)


def _is_search_source(source: VerificationSource) -> bool:
    return source in (
        VerificationSource.SEMANTIC_SCHOLAR,
        VerificationSource.OPENALEX,
        VerificationSource.DBLP,
        VerificationSource.OPENLIBRARY,
    )


def _looks_non_academic(ref: ReferenceResult) -> bool:
    """
    Returns True if the reference appears to be a non-academic source
    (book, blog, web resource) that we don't expect to find in academic DBs.
    """
    # Has a URL but no academic identifiers
    if ref.url and not ref.doi and not ref.arxiv_id:
        url_lower = ref.url.lower()
        if any(
            sig in url_lower
            for sig in ("github.com", "openai.com", "blog.", "medium.com")
        ):
            return True

    # Venue suggests a book or non-academic publisher
    if ref.venue:
        venue_lower = ref.venue.lower()
        if any(sig in venue_lower for sig in NON_ACADEMIC_SIGNALS):
            return True

    return False


def _has_url(ref: ReferenceResult) -> bool:
    return bool(ref.url)


# ---------------------------------------------------------------------------
# Core verdict logic
# ---------------------------------------------------------------------------


def _determine_verdict(result: VerificationResult) -> str:
    """
    Rule-based verdict. First matching rule wins.
    No numeric scoring — each rule maps directly to an outcome.
    """
    ref = result.reference
    source_results = result.source_results

    # Rule 1 — RETRACTED (highest priority, overrides everything)
    if any(sr.retracted for sr in source_results):
        return RETRACTED

    # Partition results into found vs not found
    found_results = [sr for sr in source_results if sr.found]
    not_found_results = [sr for sr in source_results if not sr.found]

    # Rule 2 — VERIFIED via identifier (Crossref or arXiv)
    for sr in found_results:
        if _is_identifier_source(sr.source):
            sim = sr.title_similarity or 0.0
            if sim >= TITLE_THRESHOLD:
                return VERIFIED

    # Rule 3 — NEEDS_REVIEW: identifier found but title mismatch
    for sr in found_results:
        if _is_identifier_source(sr.source):
            sim = sr.title_similarity or 0.0
            if sim < TITLE_THRESHOLD:
                return NEEDS_REVIEW

    # Rule 4 — VERIFIED via search with author confirmation
    for sr in found_results:
        if _is_search_source(sr.source):
            sim = sr.title_similarity or 0.0
            if sim >= TITLE_THRESHOLD and sr.author_match is True:
                return VERIFIED

    # Rule 5 — LIKELY_REAL: search found, title matches, author unknown (et al.)
    for sr in found_results:
        if _is_search_source(sr.source):
            sim = sr.title_similarity or 0.0
            if sim >= TITLE_THRESHOLD and sr.author_match is None:
                return LIKELY_REAL

    # Rule 6 — NEEDS_REVIEW: search found, title matches, but author mismatch
    for sr in found_results:
        if _is_search_source(sr.source):
            sim = sr.title_similarity or 0.0
            if sim >= TITLE_THRESHOLD and sr.author_match is False:
                return NEEDS_REVIEW

    # Rule 7 — NEEDS_REVIEW: search found but title similarity below threshold
    for sr in found_results:
        if _is_search_source(sr.source):
            sim = sr.title_similarity or 0.0
            if 0 < sim < TITLE_THRESHOLD:
                return NEEDS_REVIEW

    # Nothing found anywhere — distinguish UNVERIFIABLE from LIKELY_HALLUCINATED
    if not found_results:

        # Rule 8 — UNVERIFIABLE: has a direct URL (blog, PDF, GitHub)
        if _has_url(ref):
            return UNVERIFIABLE

        # Rule 9 — UNVERIFIABLE: looks like a book or non-academic source
        if _looks_non_academic(ref):
            return UNVERIFIABLE

        # Rule 10 — LIKELY_HALLUCINATED: academic-looking, nothing found
        return LIKELY_HALLUCINATED

    # Fallback — should not normally reach here
    return NEEDS_REVIEW


# ---------------------------------------------------------------------------
# merge_results_node
# ---------------------------------------------------------------------------


@observe(name="merge_results_node")
async def merge_results_node(state: WorkflowState) -> dict:
    """
    Combines doi_results, arxiv_results, search_results into one list.
    Deduplicates by raw_reference — if a paper went through multiple paths
    (e.g. arXiv failed → search ran too), merge their source_results together.

    Called after: verify_search_node (or needs_search_node if skipped)
    Writes to:    merged_results
    """
    doi_results: list[VerificationResult] = state.get("doi_results", [])
    arxiv_results: list[VerificationResult] = state.get("arxiv_results", [])
    search_results: list[VerificationResult] = state.get("search_results", [])
    dblp_results: list[VerificationResult] = state.get("dblp_results", [])

    # Index by raw_reference for deduplication
    merged: dict[str, VerificationResult] = {}

    for result in doi_results + arxiv_results + search_results + dblp_results:
        key = result.reference.raw_reference

        if key not in merged:
            merged[key] = result
        else:
            # Paper appeared in multiple paths — merge source signals
            existing = merged[key]
            existing.sources_checked.extend(
                s for s in result.sources_checked if s not in existing.sources_checked
            )
            existing.source_results.extend(result.source_results)

    all_results = list(merged.values())

    logger.info(
        "merge_results_node: %d total results "
        "(doi=%d arxiv=%d search=%d dblp=%d deduplicated=%d)",
        len(all_results),
        len(doi_results),
        len(arxiv_results),
        len(search_results),
        len(dblp_results),
        len(doi_results)
        + len(arxiv_results)
        + len(search_results)
        + len(dblp_results)
        - len(all_results),
    )

    return {"merged_results": all_results}


# ---------------------------------------------------------------------------
# score_node
# ---------------------------------------------------------------------------


# @observe(name="score_node")
# async def score_node(state: WorkflowState) -> dict:
#     """
#     Applies rule-based verdict to each merged VerificationResult.
#     Mutates verdict field in place, returns scored_references.

#     Called after: merge_results_node
#     Writes to:    scored_references
#     """
#     merged_results: list[VerificationResult] = state.get("merged_results", [])

#     verdict_counts: dict[str, int] = {}

#     for result in merged_results:
#         verdict = _determine_verdict(result)
#         result.verdict = verdict
#         verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

#         logger.info(
#             "score_node [%s] → %s (sources: %s)",
#             (result.reference.title or "?")[:50],
#             verdict,
#             [s.value for s in result.sources_checked],
#         )

#     logger.info(
#         "score_node complete: %d references scored — %s",
#         len(merged_results),
#         verdict_counts,
#     )

#     return {"scored_references": merged_results}


@observe(name="score_node")
async def score_node(state: WorkflowState) -> dict:
    from app.models.schemas import (
        ReferenceVerdict,
        VerifySummary,
        VerifyResponse,
    )

    merged_results: list[VerificationResult] = state.get("merged_results", [])

    verdicts: list[ReferenceVerdict] = []
    verdict_counts: dict[str, int] = {
        "VERIFIED": 0,
        "LIKELY_REAL": 0,
        "NEEDS_REVIEW": 0,
        "UNVERIFIABLE": 0,
        "LIKELY_HALLUCINATED": 0,
        "RETRACTED": 0,
    }

    for result in merged_results:
        verdict = _determine_verdict(result)
        result.verdict = verdict
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

        # Find best match across all sources (highest title_similarity)
        best_sr = None
        for sr in result.source_results:
            if sr.found:
                if best_sr is None:
                    best_sr = sr
                elif (sr.title_similarity or 0.0) > (best_sr.title_similarity or 0.0):
                    best_sr = sr

        ref = result.reference
        verdicts.append(
            ReferenceVerdict(
                raw_reference=ref.raw_reference,
                title=ref.title,
                authors=ref.authors,
                year=ref.year,
                doi=ref.doi,
                arxiv_id=ref.arxiv_id,
                verdict=verdict,
                sources_checked=[s.value for s in result.sources_checked],
                matched_title=best_sr.matched_title if best_sr else None,
                matched_url=best_sr.matched_url if best_sr else None,
                title_similarity=best_sr.title_similarity if best_sr else None,
                author_match=best_sr.author_match if best_sr else None,
                year_delta=best_sr.year_delta if best_sr else None,
            )
        )

        logger.info(
            "score_node [%s] → %s (sources: %s)",
            (ref.title or "?")[:50],
            verdict,
            [s.value for s in result.sources_checked],
        )

    summary = VerifySummary(
        total=len(verdicts),
        verified=verdict_counts.get("VERIFIED", 0),
        likely_real=verdict_counts.get("LIKELY_REAL", 0),
        needs_review=verdict_counts.get("NEEDS_REVIEW", 0),
        unverifiable=verdict_counts.get("UNVERIFIABLE", 0),
        likely_hallucinated=verdict_counts.get("LIKELY_HALLUCINATED", 0),
        retracted=verdict_counts.get("RETRACTED", 0),
    )

    verify_response = VerifyResponse(summary=summary, references=verdicts)

    logger.info(
        "score_node complete: %d references — %s",
        len(verdicts),
        verdict_counts,
    )

    return {
        "scored_references": merged_results,  # keep raw for debugging
        "verify_response": verify_response,  # this goes to the API
    }
