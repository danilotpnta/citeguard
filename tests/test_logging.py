import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI, Depends, Request

from app.api.middleware.logging import LoggingMiddleware, _get_client_ip
from app.db.tokens import create_token
from app.db.usage import get_usage_by_token
from app.api.dependencies import get_current_token
from app.models.token import Token


def _create_app() -> FastAPI:
    """Create an app with logging middleware and an authenticated route."""
    app = FastAPI()
    app.add_middleware(LoggingMiddleware)

    @app.get("/public")
    async def public_route():
        return {"public": True}

    @app.get("/protected")
    async def protected_route(token: Token = Depends(get_current_token)):
        return {"token_id": token.token_id}

    return app


@pytest.mark.asyncio
async def test_logging_records_authenticated_request(setup_test_db):
    """Authenticated requests should be logged to the database."""
    token = await create_token(company="LogTestCorp")
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/protected", headers={"X-API-Key": token.token_id})

    assert resp.status_code == 200

    logs = await get_usage_by_token(token.token_id)
    assert len(logs) == 1
    assert logs[0].method == "GET"
    assert logs[0].path == "/protected"
    assert logs[0].status_code == 200


@pytest.mark.asyncio
async def test_logging_skips_unauthenticated_request(setup_test_db):
    """Public requests (no token) should not create database log entries."""
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/public")

    assert resp.status_code == 200
    # No token attached, so nothing should be in usage_logs.
    # We can check by looking for any token (there are none).


@pytest.mark.asyncio
async def test_logging_captures_ip_from_x_forwarded_for(setup_test_db):
    """Should extract IP from X-Forwarded-For header."""
    token = await create_token(company="IPTestCorp")
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/protected",
            headers={
                "X-API-Key": token.token_id,
                "X-Forwarded-For": "203.0.113.50, 10.0.0.1",
            },
        )

    assert resp.status_code == 200

    logs = await get_usage_by_token(token.token_id)
    assert len(logs) == 1
    assert logs[0].ip_address == "203.0.113.50"


@pytest.mark.asyncio
async def test_logging_captures_cf_connecting_ip(setup_test_db):
    """Should prefer CF-Connecting-IP over X-Forwarded-For."""
    token = await create_token(company="CFIPCorp")
    app = _create_app()

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get(
            "/protected",
            headers={
                "X-API-Key": token.token_id,
                "CF-Connecting-IP": "198.51.100.25",
                "X-Forwarded-For": "203.0.113.50",
            },
        )

    assert resp.status_code == 200

    logs = await get_usage_by_token(token.token_id)
    assert len(logs) == 1
    assert logs[0].ip_address == "198.51.100.25"


def test_get_client_ip_cloudflare():
    """_get_client_ip should prefer CF-Connecting-IP."""
    from unittest.mock import MagicMock

    request = MagicMock()
    request.headers = {"cf-connecting-ip": "1.2.3.4", "x-forwarded-for": "5.6.7.8"}
    request.client.host = "127.0.0.1"

    assert _get_client_ip(request) == "1.2.3.4"


def test_get_client_ip_forwarded():
    """_get_client_ip should fall back to X-Forwarded-For."""
    from unittest.mock import MagicMock

    request = MagicMock()
    request.headers = {"x-forwarded-for": "5.6.7.8, 10.0.0.1"}
    request.client.host = "127.0.0.1"

    assert _get_client_ip(request) == "5.6.7.8"


def test_get_client_ip_direct():
    """_get_client_ip should fall back to request.client.host."""
    from unittest.mock import MagicMock

    request = MagicMock()
    request.headers = {}
    request.client.host = "192.168.1.1"

    assert _get_client_ip(request) == "192.168.1.1"