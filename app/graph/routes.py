# app/graph/routes.py
from typing import Literal
from app.graph.state import WorkflowState
from app.core.logging import get_logger

logger = get_logger(__name__)


def decide_content_type(state: WorkflowState) -> Literal["text", "file"]:
    """
    Called after: router_input_node
    Checks:       state["content_type"]
    Returns:
        "text" → goes to gather_all_info_node
        "file" → goes to parse_content_from_file_node
    """
    return "text" if state["content_type"] == "text" else "file"


def decide_needs_search(state: WorkflowState) -> Literal["search", "merge"]:
    """
    Called after: needs_search_node
    If any refs still need title search → verify_search_node.
    If all refs resolved by fast pass   → skip straight to merge.
    """
    has_refs = bool(state.get("refs_needing_search"))
    decision = "search" if has_refs else "merge"

    logger.info(
        "decide_needs_search: %d refs needing search → routing to '%s'",
        len(state.get("refs_needing_search", [])),
        decision,
    )

    return decision


def decide_needs_dblp(state: WorkflowState) -> Literal["dblp", "merge"]:
    """
    Called after: verify_search_node
    If DBLP is available and there are unresolved refs → verify_dblp_node
    Otherwise → merge_results_node
    """
    from app.core.config import settings

    refs = state.get("refs_needing_dblp", [])

    if not refs:
        logger.info("decide_needs_dblp: no unresolved refs → merge")
        return "merge"

    if not settings.dblp_available:
        logger.info("decide_needs_dblp: DBLP not available → merge")
        return "merge"

    logger.info(
        "decide_needs_dblp: %d unresolved refs → dblp",
        len(refs),
    )
    return "dblp"


def decide_needs_openlibrary(state: WorkflowState) -> Literal["openlibrary", "merge"]:
    """
    Called after: verify_dblp_node
    If there are unresolved refs after DBLP → verify_openlibrary_node
    Otherwise → merge_results_node
    """
    refs = state.get("refs_needing_openlibrary", [])

    if not refs:
        logger.info("decide_needs_openlibrary: no unresolved refs → merge")
        return "merge"

    logger.info(
        "decide_needs_openlibrary: %d unresolved refs → openlibrary",
        len(refs),
    )
    return "openlibrary"
