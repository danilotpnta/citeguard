from fastapi import Depends, HTTPException, status

from app.db.tokens import increment_usage
from app.models.token import Token
from app.api.dependencies import get_current_token


async def check_rate_limit(
    token: Token = Depends(get_current_token),
) -> Token:
    """
    Check that the token has remaining requests, then increment usage.

    This runs after get_current_token (which already validates the token),
    so we only need to increment and handle the edge case where another
    request used the last slot between validation and here.
    """
    if token.remaining_requests <= 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Request limit reached for this token.",
        )

    await increment_usage(token.token_id)
    return token