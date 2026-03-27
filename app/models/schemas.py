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


class ReferenceResult(BaseModel):

    title: Optional[str] = None
    # authors: Optional[list[str]] = Field(
    #     default=None,
    #     description="Each author string copied verbatim from the text.",
    # )
    authors: Optional[list[str]] = Field(
        default=None,
        description=(
            "List of author names. Split into one name per list element. "
            "Copy each name exactly as written. "
        ),
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


class VerificationSource(str, Enum):
    CROSSREF = "crossref"
    ARXIV = "arxiv"
    SEMANTIC_SCHOLAR = "semantic_scholar"
    OPENALEX = "openalex"
    DBLP = "dblp"
    OPENLIBRARY = "openlibrary"
    WEB_SEARCH = "web_search"


class SourceResult(BaseModel):
    """Result from a single verification source."""

    source: VerificationSource
    found: bool
    title_similarity: float | None = None  # -------- 0.0–1.0, rapidfuzz score
    author_match: bool | None = None  # ------------- at least one last name matches
    year_delta: int | None = None  # ---------------- abs(cited_year - found_year)
    venue_match: bool | None = None  # -------------- for DBLP
    retracted: bool = False  # ---------------------- Crossref retraction flag
    matched_title: str | None = None  # ------------- what the source actually has
    matched_url: str | None = None  # --------------- link to the real paper


class VerificationResult(BaseModel):
    """Aggregated verification outcome for one reference."""

    reference: ReferenceResult  # ------------------- original extracted reference
    sources_checked: list[VerificationSource]
    source_results: list[SourceResult]  # ----------- one per source that ran
    # populated by score_node
    confidence: float | None = None  # -------------- 0.0–1.0
    verdict: str | None = (
        None  # VERIFIED / NEEDS_REVIEW / LIKELY_HALLUCINATED / RETRACTED
    )


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


class ReferenceVerdict(BaseModel):
    """
    The fully resolved result for a single reference — what the API returns.
    Flattens VerificationResult into a user-facing shape.
    """

    # Original reference
    raw_reference: str
    title: str | None = None
    authors: list[str] | None = None
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None

    # Verdict
    verdict: str

    # Sources
    sources_checked: list[str]  # list of source name strings e.g. ["arxiv", "openalex"]

    # Best match across all sources (highest title_similarity)
    matched_title: str | None = None
    matched_url: str | None = None
    title_similarity: float | None = None
    author_match: bool | None = None
    year_delta: int | None = None


class VerifySummary(BaseModel):
    total: int
    verified: int
    likely_real: int
    needs_review: int
    unverifiable: int
    likely_hallucinated: int
    retracted: int


class VerifyResponse(BaseModel):
    summary: VerifySummary
    references: list[ReferenceVerdict]
