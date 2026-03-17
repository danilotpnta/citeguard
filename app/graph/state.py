from typing import TypedDict, Optional, List, Literal
from app.models.schemas import ReferenceResult, VerificationResult, VerifyResponse


class WorkflowState(TypedDict):
    """State that flows through the citeguard LangGraph pipeline"""

    # ═══════════════════════════════════════
    # Input
    # ═══════════════════════════════════════
    user_id: str
    input_type: Literal["text", "file"]

    # -- Input --
    raw_input: str | bytes
    content_type: str  # "text", "pdf", "docx", "md"
    filename: str | None

    # -- After parser node --
    text: str

    # -- After extractor node --
    extracted_references: list[dict]
    # Each dict: {raw_reference, title, authors, year, doi, journal}

    # classifier outputs
    refs_with_doi: list[ReferenceResult]
    refs_with_arxiv: list[ReferenceResult]
    refs_with_neither: list[ReferenceResult]

    # refs_with_neither ──────────────────────────────────┐
    #                                                     ▼
    # doi_results (failed ones) ──────────► needs_search_node → refs_needing_search
    # arxiv_results (failed ones) ────────┘

    # fast pass results — list of VerificationResult
    doi_results: list[VerificationResult]
    arxiv_results: list[VerificationResult]

    # search pass
    refs_needing_search: list[ReferenceResult]
    search_results: list[VerificationResult]

    # dblp pass (refs OpenAlex didn't find)
    refs_needing_dblp: list[ReferenceResult]
    dblp_results: list[VerificationResult]

    # openlibrary pass (refs DBLP didn't find)
    refs_needing_openlibrary: list[ReferenceResult]
    openlibrary_results: list[VerificationResult]

    # final
    merged_results: list[VerificationResult]  # all results combined
    verify_response: VerifyResponse

    # # -- After verifier node --
    # verification_results: list[dict]
    # # Each dict: extracted fields + {matches: [...], sources_checked: [...]}

    # # -- After scorer node (final output) --
    # references: list[ReferenceResult]
