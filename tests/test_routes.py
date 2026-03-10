import io

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from app.core.config import Settings, get_settings
from app.db.database import init_db, set_db_path
from app.db.tokens import create_token
from app.main import create_app


def _auth(token) -> dict:
    """Helper to build auth header from a token."""
    return {"X-API-Key": token.token_id}


@pytest_asyncio.fixture
async def app(setup_test_db):
    """Create a test app with overridden settings."""
    application = create_app()

    def _test_settings():
        return Settings(
            _env_file=None,
            admin_api_key="test-admin-secret",
            database_path=setup_test_db,
        )

    application.dependency_overrides[get_settings] = _test_settings
    return application


@pytest_asyncio.fixture
async def client(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


@pytest_asyncio.fixture
async def token(setup_test_db):
    return await create_token(company="TestCorp", max_requests=100)


# -- Health --


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["service"] == "citeguard"


# -- Verify text --


@pytest.mark.asyncio
async def test_verify_text(client, token):
    """POST /verify with valid text should return a stub result."""
    resp = await client.post(
        "/verify",
        json={"text": "See Smith et al. (2023) for details."},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["token_id"] == token.token_id
    assert data["total_references"] >= 1
    assert len(data["references"]) >= 1
    assert "job_id" in data


@pytest.mark.asyncio
async def test_verify_text_empty(client, token):
    """POST /verify with empty text should be rejected by validation."""
    resp = await client.post(
        "/verify",
        json={"text": ""},
        headers=_auth(token),
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_verify_text_no_auth(client):
    """POST /verify without token should return 401."""
    resp = await client.post("/verify", json={"text": "some text"})
    assert resp.status_code == 401


# -- Verify file upload --


@pytest.mark.asyncio
async def test_verify_upload_txt(client, token):
    """POST /verify/upload with a .txt file should work."""
    content = b"References:\nSmith, J. (2023). On Testing. Journal of Tests, 1(1), 1-10."
    resp = await client.post(
        "/verify/upload",
        files={"file": ("refs.txt", io.BytesIO(content), "text/plain")},
        headers=_auth(token),
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["total_references"] >= 1


@pytest.mark.asyncio
async def test_verify_upload_unsupported_type(client, token):
    """POST /verify/upload with an unsupported file type should return 400."""
    resp = await client.post(
        "/verify/upload",
        files={"file": ("data.csv", io.BytesIO(b"a,b,c"), "text/csv")},
        headers=_auth(token),
    )
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


# -- Get result by job_id --


@pytest.mark.asyncio
async def test_get_result(client, token):
    """GET /verify/{job_id} should return the previous result."""
    resp = await client.post(
        "/verify",
        json={"text": "Test reference for retrieval."},
        headers=_auth(token),
    )
    job_id = resp.json()["job_id"]

    resp = await client.get(f"/verify/{job_id}", headers=_auth(token))
    assert resp.status_code == 200
    assert resp.json()["job_id"] == job_id


@pytest.mark.asyncio
async def test_get_result_not_found(client, token):
    """GET /verify/{job_id} with unknown ID should return 404."""
    resp = await client.get("/verify/nonexistent123", headers=_auth(token))
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_get_result_wrong_token(client, token, setup_test_db):
    """GET /verify/{job_id} with a different token should return 403."""
    resp = await client.post(
        "/verify",
        json={"text": "Test reference for isolation."},
        headers=_auth(token),
    )
    job_id = resp.json()["job_id"]

    other_token = await create_token(company="OtherCorp", max_requests=100)
    resp = await client.get(f"/verify/{job_id}", headers=_auth(other_token))
    assert resp.status_code == 403


# -- Admin endpoints --


@pytest.mark.asyncio
async def test_admin_create_token(client):
    """POST /admin/tokens should create a token."""
    resp = await client.post(
        "/admin/tokens",
        json={"company": "NewCorp", "max_requests": 25, "expires_in_days": 7},
        headers={"X-Admin-Key": "test-admin-secret"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["company"] == "NewCorp"
    assert data["max_requests"] == 25
    assert data["remaining_requests"] == 25
    assert data["status"] == "active"


@pytest.mark.asyncio
async def test_admin_list_tokens(client, token):
    """GET /admin/tokens should list all tokens."""
    resp = await client.get(
        "/admin/tokens",
        headers={"X-Admin-Key": "test-admin-secret"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    ids = [t["token_id"] for t in data["tokens"]]
    assert token.token_id in ids


@pytest.mark.asyncio
async def test_admin_revoke_token(client, token):
    """POST /admin/revoke/{token_id} should revoke the token."""
    resp = await client.post(
        f"/admin/revoke/{token.token_id}",
        headers={"X-Admin-Key": "test-admin-secret"},
    )
    assert resp.status_code == 200
    assert "revoked" in resp.json()["detail"]

    # Verify the token is now rejected
    resp = await client.post(
        "/verify",
        json={"text": "test"},
        headers=_auth(token),
    )
    assert resp.status_code == 403


@pytest.mark.asyncio
async def test_admin_revoke_not_found(client):
    """POST /admin/revoke/{token_id} with unknown ID should return 404."""
    resp = await client.post(
        "/admin/revoke/nonexistent",
        headers={"X-Admin-Key": "test-admin-secret"},
    )
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_admin_usage(client, token):
    """GET /admin/usage/{token_id} should return usage stats."""
    await client.post(
        "/verify",
        json={"text": "Generate usage data."},
        headers=_auth(token),
    )

    resp = await client.get(
        f"/admin/usage/{token.token_id}",
        headers={"X-Admin-Key": "test-admin-secret"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["token_id"] == token.token_id
    assert data["company"] == "TestCorp"
    assert data["used_requests"] >= 1


@pytest.mark.asyncio
async def test_admin_no_key(client):
    """Admin endpoints without key should return 401."""
    resp = await client.get("/admin/tokens")
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_admin_wrong_key(client):
    """Admin endpoints with wrong key should return 403."""
    resp = await client.get(
        "/admin/tokens",
        headers={"X-Admin-Key": "wrong-key"},
    )
    assert resp.status_code == 403