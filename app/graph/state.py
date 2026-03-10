from typing import TypedDict, Optional, List, Literal
from app.models.schemas import ReferenceResult

class WorkflowState(TypedDict):
    """State that flows through the citeguard LangGraph pipeline"""

    # ═══════════════════════════════════════
    # Input
    # ═══════════════════════════════════════
    user_id: str
    input_type: Literal["text", "file"]
    doc_text: Optional[str]

    # -- Input --
    raw_input: str | bytes
    content_type: str  # "text/plain", "application/pdf", etc.
    filename: str | None

    # -- After parser node --
    text: str

    # -- After extractor node --
    extracted_references: list[dict]
    # Each dict: {raw_reference, title, authors, year, doi, journal}

    # -- After verifier node --
    verification_results: list[dict]
    # Each dict: extracted fields + {matches: [...], sources_checked: [...]}

    # -- After scorer node (final output) --
    references: list[ReferenceResult]
