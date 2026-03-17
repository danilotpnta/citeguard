import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI
from pathlib import Path

from app.api.middleware.killswitch import KillswitchMiddleware, set_killswitch_path


def _create_app(killswitch_path: Path) -> FastAPI:
    """Create a minimal app with killswitch middleware."""
    set_killswitch_path(killswitch_path)

    app = FastAPI()
    app.add_middleware(KillswitchMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/api/test")
    async def test_route():
        return {"data": "hello"}

    return app


@pytest.mark.asyncio
async def test_killswitch_off_allows_requests(tmp_path):
    """When killswitch file does not exist, requests pass through."""
    ks_path = tmp_path / "KILLSWITCH"
    app = _create_app(ks_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/test")

    assert resp.status_code == 200
    assert resp.json()["data"] == "hello"


@pytest.mark.asyncio
async def test_killswitch_on_blocks_requests(tmp_path):
    """When killswitch file exists, requests get 503."""
    ks_path = tmp_path / "KILLSWITCH"
    ks_path.touch()
    app = _create_app(ks_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/test")

    assert resp.status_code == 503
    assert "unavailable" in resp.json()["detail"].lower()


@pytest.mark.asyncio
async def test_killswitch_allows_health_check(tmp_path):
    """Even with killswitch on, /health should respond."""
    ks_path = tmp_path / "KILLSWITCH"
    ks_path.touch()
    app = _create_app(ks_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")

    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_killswitch_toggle(tmp_path):
    """Killswitch should respond to file being created/deleted at runtime."""
    ks_path = tmp_path / "KILLSWITCH"
    app = _create_app(ks_path)

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        # Off - should work
        resp = await client.get("/api/test")
        assert resp.status_code == 200

        # Turn on
        ks_path.touch()
        resp = await client.get("/api/test")
        assert resp.status_code == 503

        # Turn off again
        ks_path.unlink()
        resp = await client.get("/api/test")
        assert resp.status_code == 200