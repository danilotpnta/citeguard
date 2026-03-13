from enum import Enum
from typing import List, Optional
from datetime import datetime

from pydantic import BaseModel, Field


class VerificationStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ReferenceStatus(str, Enum):
    VERIFIED = "verified"
    SUSPICIOUS = "suspicious"
    HALLUCINATED = "hallucinated"
    UNRESOLVED = "unresolved"


# class ReferenceResult(BaseModel):
#     # --- LLM should extract ---
#     title: Optional[str]  # Phase 3, 4, 5 — most important field
#     authors: Optional[List[str]]  # Phase 2, 3, 5 — list of "Lastname, F." strings
#     year: Optional[int]  # Phase 2, 3, 5 — just the integer year
#     venue: Optional[str]  # Phase 5 — journal or conference name

#     # --- Identifiers (extract if present, None if not) ---
#     doi: Optional[str]  # Phase 2 — "10.xxxx/xxxxx" only, no URL prefix
#     arxiv_id: Optional[str]  # Phase 4 — "1706.03762" only, no "arXiv:" prefix
#     url: Optional[str]  # fallback signal

#     # --- Provenance ---
#     raw_text: str  # the original reference string, always keep this


class ReferenceResult(BaseModel):

    title: Optional[str] = None
    authors: Optional[list[str]] = Field(
        default=None,
        # description="List of author names exactly as written in the reference. Do not reorder or reformat names.",
        description="Each author string copied verbatim from the text.",
    )
    year: Optional[int] = None
    venue: Optional[str] = Field(
        default=None,
        description="Journal, conference, or workshop name as written in the reference.",
    )
    doi: Optional[str] = Field(
        default=None,
        description="DOI string only, no URL prefix. e.g. '10.xxxx/xxxxx'",
    )
    arxiv_id: Optional[str] = Field(
        default=None,
        description="arXiv ID only, no prefix or version suffix. e.g. '1706.03762'",
    )
    url: Optional[str] = None
    raw_reference: str = Field(
        ..., description="The original reference string as extracted"
    )


class ReferenceList(BaseModel):
    references: list[ReferenceResult] = Field(default_factory=list)


class ReferenceResult_(BaseModel):
    """Verification result for a single reference."""

    raw_reference: str = Field(
        ..., description="The original reference string as extracted"
    )
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    doi: str | None = None
    journal: str | None = None

    status: ReferenceStatus = ReferenceStatus.UNRESOLVED
    confidence: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Confidence score 0-1"
    )
    sources_checked: list[str] = Field(
        default_factory=list, description="APIs queried during verification"
    )
    match_details: str | None = Field(
        None, description="Explanation of why this status was assigned"
    )


# ============================================================================
# REQUEST MODELS
# ============================================================================


class VerifyRequest(BaseModel):
    """Request body for the /verify endpoint when sending raw text."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=50000,
        description="Text containing references to verify",
    )


# ============================================================================
# RESPONSE MODELS
# ============================================================================


class VerifyResponse(BaseModel):
    """Response from the /verify endpoint."""

    job_id: str
    status: VerificationStatus
    created_at: datetime
    token_id: str
    total_references: int = 0
    verified: int = 0
    suspicious: int = 0
    hallucinated: int = 0
    unresolved: int = 0
    references: list[ReferenceResult] = Field(default_factory=list)


class TokenUsageResponse(BaseModel):
    """Response for admin usage endpoint."""

    token_id: str
    company: str
    status: str
    used_requests: int
    max_requests: int
    remaining_requests: int
    total_logged: int
    unique_ips: int
    last_used: datetime | None


class AdminTokenListResponse(BaseModel):
    """Response for admin token list endpoint."""

    tokens: list[TokenUsageResponse]
    total: int
