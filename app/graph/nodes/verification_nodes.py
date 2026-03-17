# app/graph/nodes/verification_nodes.py
import re
from typing import Literal

from langfuse import observe

from app.graph.state import WorkflowState
from app.models.schemas import ReferenceResult, VerificationResult
from app.core.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_arxiv_doi(doi: str) -> bool:
    """10.48550 is arXiv's DOI prefix — registered with DataCite not Crossref."""
    return doi.startswith("10.48550/")


def _extract_arxiv_id_from_doi(doi: str) -> str | None:
    """
    '10.48550/arXiv.1706.03762' → '1706.03762'
    Returns None if pattern not found.
    """
    parts = doi.split("arXiv.")
    return parts[1] if len(parts) == 2 else None


# ---------------------------------------------------------------------------
# classify_references_node
# ---------------------------------------------------------------------------

ARXIV_PATTERN = re.compile(r"arXiv[:\s]+(\d{4}\.\d{4,5})", re.IGNORECASE)
DOI_PATTERN = re.compile(r"\b(10\.\d{4,}/\S+)", re.IGNORECASE)


def _enrich_reference(ref: ReferenceResult) -> ReferenceResult:
    """
    Recover missing identifiers from raw_reference text.
    Returns an enriched copy — never mutates the original.
    """
    updates = {}

    # Recover DOI the LLM missed
    if not ref.doi:
        m = DOI_PATTERN.search(ref.raw_reference)
        if m:
            updates["doi"] = m.group(1).rstrip(".,)")

    # Recover arXiv ID the LLM missed
    if not ref.arxiv_id and not updates.get("doi"):
        m = ARXIV_PATTERN.search(ref.raw_reference)
        if m:
            updates["arxiv_id"] = m.group(1)

    # Promote arXiv DOI → arxiv_id
    doi = updates.get("doi") or ref.doi
    if doi and _is_arxiv_doi(doi) and not ref.arxiv_id:
        extracted = _extract_arxiv_id_from_doi(doi)
        if extracted:
            updates["arxiv_id"] = extracted

    if updates:
        logger.debug("Enriched ref '%s': %s", ref.title or "?", updates)
        return ref.model_copy(update=updates)
    return ref


def _classify_reference(ref: ReferenceResult) -> str:
    """Return bucket name: 'doi' | 'arxiv' | 'neither'."""
    if ref.doi and not _is_arxiv_doi(ref.doi):
        return "doi"
    if ref.arxiv_id or (ref.doi and _is_arxiv_doi(ref.doi)):
        return "arxiv"
    return "neither"


@observe(name="classify_references_node")
async def classify_references_node(state: WorkflowState) -> dict:
    references: list[ReferenceResult] = state["extracted_references"].references

    refs_with_doi: list[ReferenceResult] = []
    refs_with_arxiv: list[ReferenceResult] = []
    refs_with_neither: list[ReferenceResult] = []

    for ref in references:
        ref = _enrich_reference(ref)

        bucket = _classify_reference(ref)
        if bucket == "doi":
            refs_with_doi.append(ref)
        elif bucket == "arxiv":
            refs_with_arxiv.append(ref)
        else:
            refs_with_neither.append(ref)

    logger.info(
        "Classified %d references → doi=%d arxiv=%d neither=%d",
        len(references),
        len(refs_with_doi),
        len(refs_with_arxiv),
        len(refs_with_neither),
    )

    return {
        "refs_with_doi": refs_with_doi,
        "refs_with_arxiv": refs_with_arxiv,
        "refs_with_neither": refs_with_neither,
    }


def _needs_title_search(result: VerificationResult) -> bool:
    """
    Returns True if the result from the fast pass (DOI or arXiv check)
    was not conclusive enough to skip title search.

    A result is conclusive if:
      - found=True AND title_similarity >= 0.85 AND author_match=True
    Anything else falls through to the search layer.
    """
    if not result.source_results:
        return True

    # Take the single source result from the fast pass
    sr = result.source_results[0]

    if not sr.found:
        return True

    title_ok = (sr.title_similarity or 0.0) >= 0.85

    # None means no author data (et al.) — don't penalize, rely on title only
    if sr.author_match is None:
        return not title_ok

    author_ok = sr.author_match is True

    return not (title_ok and author_ok)


@observe(name="needs_search_node")
async def needs_search_node(state: WorkflowState) -> dict:
    """
    Decides which references still need a title search after the fast pass.

    Collects:
      1. refs_with_neither  — never had an identifier
      2. Failed/inconclusive doi_results
      3. Failed/inconclusive arxiv_results

    Called after: verify_doi_node + verify_arxiv_node (barrier join)
    Writes to:    refs_needing_search
    """
    doi_results: list[VerificationResult] = state.get("doi_results", [])
    arxiv_results: list[VerificationResult] = state.get("arxiv_results", [])
    refs_with_neither: list[ReferenceResult] = state.get("refs_with_neither", [])

    needs_search: list[ReferenceResult] = list(refs_with_neither)

    for result in doi_results:
        if _needs_title_search(result):
            needs_search.append(result.reference)
            logger.debug(
                "DOI check inconclusive for '%s' → queuing for title search",
                result.reference.title or result.reference.raw_reference[:60],
            )

    for result in arxiv_results:
        if _needs_title_search(result):
            needs_search.append(result.reference)
            logger.debug(
                "arXiv check inconclusive for '%s' → queuing for title search",
                result.reference.title or result.reference.raw_reference[:60],
            )

    logger.info(
        "needs_search_node: %d refs queued for title search "
        "(neither=%d, doi_fallback=%d, arxiv_fallback=%d)",
        len(needs_search),
        len(refs_with_neither),
        sum(1 for r in doi_results if _needs_title_search(r)),
        sum(1 for r in arxiv_results if _needs_title_search(r)),
    )

    return {"refs_needing_search": needs_search}


@observe(name="verify_doi_node")
async def verify_doi_node(state: WorkflowState) -> dict:
    refs_with_doi: list[ReferenceResult] = state.get("refs_with_doi", [])

    if not refs_with_doi:
        logger.info("verify_doi_node: no DOI references to verify")
        return {"doi_results": []}

    from app.agents.tools.verifiers.crossref import crossref_verifier

    logger.info("verify_doi_node: verifying %d DOI references", len(refs_with_doi))
    results = await crossref_verifier.verify_batch(refs_with_doi)

    return {"doi_results": results}


@observe(name="verify_arxiv_node")
async def verify_arxiv_node(state: WorkflowState) -> dict:
    refs_with_arxiv: list[ReferenceResult] = state.get("refs_with_arxiv", [])

    if not refs_with_arxiv:
        logger.info("verify_arxiv_node: no arXiv references to verify")
        return {"arxiv_results": []}

    from app.agents.tools.verifiers.arxiv import arxiv_verifier

    logger.info(
        "verify_arxiv_node: verifying %d arXiv references", len(refs_with_arxiv)
    )
    results = await arxiv_verifier.verify_batch(refs_with_arxiv)

    return {"arxiv_results": results}


@observe(name="verify_search_node")
async def verify_search_node(state: WorkflowState) -> dict:
    refs = state["refs_needing_search"]

    from app.agents.tools.verifiers.openalex import openalex_verifier

    oa_results = await openalex_verifier.verify_batch(refs)

    # Collect refs OpenAlex didn't find — candidates for DBLP
    refs_needing_dblp = [
        result.reference
        for result in oa_results
        if not result.source_results or not result.source_results[0].found
    ]

    logger.info(
        "verify_search_node: %d refs checked, %d not found → queuing for DBLP",
        len(refs),
        len(refs_needing_dblp),
    )

    return {
        "search_results": oa_results,
        "refs_needing_dblp": refs_needing_dblp,
    }


@observe(name="verify_dblp_node")
async def verify_dblp_node(state: WorkflowState) -> dict:
    refs = state.get("refs_needing_dblp", [])

    if not refs:
        logger.info("verify_dblp_node: no refs to check")
        return {"dblp_results": [], "refs_needing_openlibrary": []}

    from app.agents.tools.verifiers.dblp import dblp_verifier

    if not dblp_verifier.available:
        logger.info("verify_dblp_node: DBLP database not available — skipping")
        return {"dblp_results": [], "refs_needing_openlibrary": refs}

    results = dblp_verifier.verify_batch(refs)

    found = sum(1 for r in results if r.source_results and r.source_results[0].found)
    logger.info(
        "verify_dblp_node: %d refs checked, %d found",
        len(refs),
        found,
    )

    # Refs DBLP didn't find → candidates for OpenLibrary
    refs_needing_openlibrary = [
        r.reference
        for r in results
        if not r.source_results or not r.source_results[0].found
    ]

    return {
        "dblp_results": results,
        "refs_needing_openlibrary": refs_needing_openlibrary,
    }


@observe(name="verify_openlibrary_node")
async def verify_openlibrary_node(state: WorkflowState) -> dict:
    refs = state.get("refs_needing_openlibrary", [])

    if not refs:
        logger.info("verify_openlibrary_node: no refs to check")
        return {"openlibrary_results": []}

    from app.agents.tools.verifiers.openlibrary import openlibrary_verifier

    results = await openlibrary_verifier.verify_batch(refs)

    found = sum(1 for r in results if r.source_results and r.source_results[0].found)
    logger.info(
        "verify_openlibrary_node: %d refs checked, %d found",
        len(refs),
        found,
    )

    return {"openlibrary_results": results}
