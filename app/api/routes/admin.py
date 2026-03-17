from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import require_admin
from app.db.tokens import create_token, get_token, list_tokens, revoke_token
from app.db.usage import get_usage_summary
from app.models.schemas import AdminTokenListResponse, TokenUsageResponse
from app.models.token import TokenCreate, TokenInfo

router = APIRouter(
    prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)]
)


@router.post("/tokens", response_model=TokenInfo)
async def admin_create_token(body: TokenCreate):
    """Create a new access token for a company."""
    token = await create_token(
        company=body.company,
        max_requests=body.max_requests,
        expires_in_days=body.expires_in_days,
    )
    return TokenInfo(
        token_id=token.token_id,
        company=token.company,
        created_at=token.created_at,
        expires_at=token.expires_at,
        max_requests=token.max_requests,
        used_requests=token.used_requests,
        remaining_requests=token.remaining_requests,
        status=token.status,
    )


@router.get("/tokens", response_model=AdminTokenListResponse)
async def admin_list_tokens(include_revoked: bool = False):
    """List all tokens with usage summaries."""
    tokens = await list_tokens(include_revoked=include_revoked)
    items = []
    for t in tokens:
        summary = await get_usage_summary(t.token_id)
        items.append(
            TokenUsageResponse(
                token_id=t.token_id,
                company=t.company,
                status=t.status.value,
                used_requests=t.used_requests,
                max_requests=t.max_requests,
                remaining_requests=t.remaining_requests,
                total_logged=summary.total_requests,
                unique_ips=summary.unique_ips,
                last_used=summary.last_used,
            )
        )
    return AdminTokenListResponse(tokens=items, total=len(items))


@router.post("/revoke/{token_id}")
async def admin_revoke_token(token_id: str):
    """Revoke an access token."""
    token = await get_token(token_id)
    if token is None:
        raise HTTPException(status_code=404, detail=f"Token '{token_id}' not found.")

    await revoke_token(token_id)
    return {"detail": f"Token '{token_id}' ({token.company}) has been revoked."}


@router.get("/usage/{token_id}", response_model=TokenUsageResponse)
async def admin_token_usage(token_id: str):
    """Get usage stats for a specific token."""
    token = await get_token(token_id)
    if token is None:
        raise HTTPException(status_code=404, detail=f"Token '{token_id}' not found.")

    summary = await get_usage_summary(token_id)
    return TokenUsageResponse(
        token_id=token.token_id,
        company=token.company,
        status=token.status.value,
        used_requests=token.used_requests,
        max_requests=token.max_requests,
        remaining_requests=token.remaining_requests,
        total_logged=summary.total_requests,
        unique_ips=summary.unique_ips,
        last_used=summary.last_used,
    )
