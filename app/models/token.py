from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, Field


class TokenStatus(str, Enum):
    ACTIVE = "active"
    REVOKED = "revoked"
    EXPIRED = "expired"


class Token(BaseModel):
    token_id: str
    company: str
    created_at: datetime
    expires_at: datetime
    max_requests: int
    used_requests: int = 0
    status: TokenStatus = TokenStatus.ACTIVE

    @property
    def is_valid(self) -> bool:
        if self.status != TokenStatus.ACTIVE:
            return False
        if datetime.now(UTC) > self.expires_at:
            return False
        if self.used_requests >= self.max_requests:
            return False
        return True

    @property
    def remaining_requests(self) -> int:
        return max(0, self.max_requests - self.used_requests)


class TokenCreate(BaseModel):
    company: str = Field(..., description="Company name this token is for")
    max_requests: int = Field(default=50, ge=1, le=1000)
    expires_in_days: int = Field(default=30, ge=1, le=365)


class TokenInfo(BaseModel):
    """Public-facing token information (no internal IDs)."""
    token_id: str
    company: str
    created_at: datetime
    expires_at: datetime
    max_requests: int
    used_requests: int
    remaining_requests: int
    status: TokenStatus