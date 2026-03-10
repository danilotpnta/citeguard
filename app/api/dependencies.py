from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, get_settings
from app.db.tokens import get_token
from app.models.token import Token


async def get_current_token(
    request: Request,
    x_api_key: str | None = Header(None),
) -> Token:
    """
    Extract and validate the access token from X-API-Key header.

    The frontend reads the token from the demo URL and sends it
    as this header on every API call.
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing access token. Provide X-API-Key header.",
        )

    token_obj = await get_token(x_api_key)

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
