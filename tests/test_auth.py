import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI, Depends

from app.api.dependencies import get_current_token, require_admin
from app.config import Settings, get_settings
from app.db.tokens import create_token, revoke_token, increment_usage
from app.models.token import Token


def _create_app() -> FastAPI:
    """Create a minimal FastAPI app with auth-protected routes for testing."""
    app = FastAPI()

    # Override settings to use a known admin key
    def _test_settings():
        return Settings(_env_file=None, admin_api_key="test-admin-secret")

    app.dependency_overrides[get_settings] = _test_settings

    @app.get("/protected")
    async def protected_route(token: Token = Depends(get_current_token)):
        return {"token_id": token.token_id, "company": token.company}

    @app.get("/admin/status")
    async def admin_route(is_admin: bool = Depends(require_admin)):
        return {"admin": True}

    return app


@pytest.mark.asyncio
async def test_auth_with_query_param(setup_test_db):
    """Token passed as ?token= query param should authenticate."""
    token = await create_token(company="QueryCorp")
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/protected?token={token.token_id}")

    assert resp.status_code == 200
    assert resp.json()["company"] == "QueryCorp"


@pytest.mark.asyncio
async def test_auth_with_header(setup_test_db):
    """Token passed as X-API-Key header should authenticate."""
    token = await create_token(company="HeaderCorp")
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/protected", headers={"X-API-Key": token.token_id})

    assert resp.status_code == 200
    assert resp.json()["company"] == "HeaderCorp"


@pytest.mark.asyncio
async def test_auth_missing_token(setup_test_db):
    """Request without any token should return 401."""
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/protected")

    assert resp.status_code == 401
    assert "Missing access token" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_invalid_token(setup_test_db):
    """Request with a non-existent token should return 401."""
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/protected?token=bogus999999")

    assert resp.status_code == 401
    assert "Invalid" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_revoked_token(setup_test_db):
    """Request with a revoked token should return 403."""
    token = await create_token(company="RevokedCorp")
    await revoke_token(token.token_id)
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/protected?token={token.token_id}")

    assert resp.status_code == 403
    assert "revoked" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_auth_exhausted_token(setup_test_db):
    """Request with an exhausted token should return 403."""
    token = await create_token(company="ExhaustedCorp", max_requests=1)
    await increment_usage(token.token_id)
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(f"/protected?token={token.token_id}")

    assert resp.status_code == 403
    assert "exhausted" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_admin_valid_key(setup_test_db):
    """Admin endpoint with correct key should return 200."""
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/status", headers={"X-Admin-Key": "test-admin-secret"})

    assert resp.status_code == 200
    assert resp.json()["admin"] is True


@pytest.mark.asyncio
async def test_admin_missing_key(setup_test_db):
    """Admin endpoint without key should return 401."""
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/status")

    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_wrong_key(setup_test_db):
    """Admin endpoint with wrong key should return 403."""
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/status", headers={"X-Admin-Key": "wrong-key"})

    assert resp.status_code == 403