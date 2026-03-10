import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI, Depends

from app.api.middleware.rate_limit import check_rate_limit
from app.db.tokens import create_token, get_token
from app.models.token import Token


def _create_app() -> FastAPI:
    """Create a minimal app with a rate-limited route."""
    app = FastAPI()

    @app.post("/verify")
    async def verify(token: Token = Depends(check_rate_limit)):
        return {
            "token_id": token.token_id,
            "remaining": token.remaining_requests,
        }

    return app


def _auth(token) -> dict:
    return {"X-API-Key": token.token_id}


@pytest.mark.asyncio
async def test_rate_limit_increments_usage(setup_test_db):
    """Each request should increment used_requests by 1."""
    token = await create_token(company="RateCorp", max_requests=10)
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.post("/verify", headers=_auth(token))

    assert resp.status_code == 200

    updated = await get_token(token.token_id)
    assert updated.used_requests == 1


@pytest.mark.asyncio
async def test_rate_limit_allows_up_to_max(setup_test_db):
    """Should allow exactly max_requests successful calls."""
    token = await create_token(company="MaxCorp", max_requests=3)
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        for i in range(3):
            resp = await client.post("/verify", headers=_auth(token))
            assert resp.status_code == 200

        # The 4th request: token is now exhausted, get_current_token rejects it
        resp = await client.post("/verify", headers=_auth(token))
        assert resp.status_code in (403, 429)


@pytest.mark.asyncio
async def test_rate_limit_different_tokens_independent(setup_test_db):
    """Rate limits should be independent per token."""
    t1 = await create_token(company="Corp1", max_requests=2)
    t2 = await create_token(company="Corp2", max_requests=2)
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Use up all of t1
        await client.post("/verify", headers=_auth(t1))
        await client.post("/verify", headers=_auth(t1))

        # t2 should still work
        resp = await client.post("/verify", headers=_auth(t2))
        assert resp.status_code == 200