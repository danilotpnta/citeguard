from typing import TypedDict, Optional, List, Literal


class WorkflowState(TypedDict):
    """State that flows through the citeguard processing graph"""

    # ═══════════════════════════════════════
    # Input
    # ═══════════════════════════════════════
    user_id: str
    input_type: Literal["text", "file"]
    doc_text: Optional[str]

    # ═══════════════════════════════════════
    # Extracted metadata
    # ═══════════════════════════════════════
