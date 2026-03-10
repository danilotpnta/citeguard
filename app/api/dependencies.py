from fastapi import Depends, Header, HTTPException, Query, Request, status

from app.config import Settings, get_settings
from app.db.tokens import get_token
from app.models.token import Token


async def get_current_token(
    request: Request,
    token: str | None = Query(None, alias="token"),
    x_api_key: str | None = Header(None),
) -> Token:
    """
    Extract and validate the access token from the request.

    Accepts the token from:
      1. Query parameter: ?token=abc123
      2. Header: X-API-Key: abc123

    The query parameter is the primary method (used in demo links).
    The header is an alternative for programmatic access.
    """
    token_id = token or x_api_key

    if not token_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing access token. Provide ?token= or X-API-Key header.",
        )

    token_obj = await get_token(token_id)

    if token_obj is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token.",
        )

    if not token_obj.is_valid:
        # Give a specific message depending on the reason
        if token_obj.status.value == "revoked":
            detail = "This access token has been revoked."
        elif token_obj.remaining_requests <= 0:
            detail = "This access token has exhausted its request limit."
        else:
            detail = "This access token has expired."

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )

    # Attach token to request state for downstream use (logging, etc.)
    request.state.token = token_obj
    return token_obj


async def require_admin(
    x_admin_key: str | None = Header(None),
    settings: Settings = Depends(get_settings),
) -> bool:
    """
    Validate the admin API key for protected admin endpoints.
    """
    if not x_admin_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin key. Provide X-Admin-Key header.",
        )

    if x_admin_key != settings.admin_api_key:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid admin key.",
        )

    return True