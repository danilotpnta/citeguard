from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """Public health check endpoint. Not behind auth or rate limiting."""
    return {"status": "ok", "service": "citeguard"}