import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from app.api.middleware.admin_protection import AdminProtectionMiddleware


def _create_app(block_via_proxy: bool = True) -> FastAPI:
    app = FastAPI()
    app.add_middleware(AdminProtectionMiddleware, block_via_proxy=block_via_proxy)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/admin/stats")
    async def admin_stats():
        return {"stats": "data"}

    return app


@pytest.mark.asyncio
async def test_proxied_admin_request_returns_404():
    """CF-Connecting-IP present → /admin returns 404."""
    app = _create_app(block_via_proxy=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/stats", headers={"cf-connecting-ip": "1.2.3.4"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_direct_admin_request_passes_through():
    """No CF-Connecting-IP → /admin route accessible."""
    app = _create_app(block_via_proxy=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/stats")
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_proxied_non_admin_request_passes_through():
    """CF-Connecting-IP present but path is not /admin → allowed."""
    app = _create_app(block_via_proxy=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health", headers={"cf-connecting-ip": "1.2.3.4"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_block_disabled_allows_proxied_admin():
    """block_via_proxy=False → proxied /admin requests are allowed."""
    app = _create_app(block_via_proxy=False)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/stats", headers={"cf-connecting-ip": "1.2.3.4"})
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_404_hides_route_not_403():
    """Returns 404 not 403 — route appears not to exist to external callers."""
    app = _create_app(block_via_proxy=True)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/admin/stats", headers={"cf-connecting-ip": "5.6.7.8"})
    assert resp.status_code == 404
    assert resp.status_code != 403
