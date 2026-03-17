from datetime import datetime, timedelta

import pytest

from app.db.tokens import create_token, get_token, increment_usage, list_tokens, revoke_token
from app.models.token import TokenStatus


@pytest.mark.asyncio
async def test_create_token(setup_test_db):
    """Creating a token should return a valid token with correct fields."""
    token = await create_token(company="TestCorp", max_requests=100, expires_in_days=14)

    assert token.token_id is not None
    assert len(token.token_id) == 12
    assert token.company == "TestCorp"
    assert token.max_requests == 100
    assert token.used_requests == 0
    assert token.status == TokenStatus.ACTIVE
    assert token.is_valid is True
    # Should expire roughly 14 days from now
    delta = token.expires_at - token.created_at
    assert 13 <= delta.days <= 14


@pytest.mark.asyncio
async def test_get_token(setup_test_db):
    """Should retrieve a token by its ID."""
    created = await create_token(company="FetchCorp")
    fetched = await get_token(created.token_id)

    assert fetched is not None
    assert fetched.token_id == created.token_id
    assert fetched.company == "FetchCorp"


@pytest.mark.asyncio
async def test_get_token_not_found(setup_test_db):
    """Should return None for a non-existent token."""
    result = await get_token("nonexistent123")
    assert result is None


@pytest.mark.asyncio
async def test_revoke_token(setup_test_db):
    """Revoking a token should set its status to revoked."""
    token = await create_token(company="RevokeCorp")
    success = await revoke_token(token.token_id)
    assert success is True

    fetched = await get_token(token.token_id)
    assert fetched is not None
    assert fetched.status == TokenStatus.REVOKED
    assert fetched.is_valid is False


@pytest.mark.asyncio
async def test_revoke_nonexistent_token(setup_test_db):
    """Revoking a non-existent token should return False."""
    success = await revoke_token("doesnotexist")
    assert success is False


@pytest.mark.asyncio
async def test_increment_usage(setup_test_db):
    """Incrementing usage should increase used_requests by 1."""
    token = await create_token(company="UsageCorp", max_requests=5)

    await increment_usage(token.token_id)
    await increment_usage(token.token_id)

    fetched = await get_token(token.token_id)
    assert fetched.used_requests == 2
    assert fetched.remaining_requests == 3


@pytest.mark.asyncio
async def test_token_exhausted(setup_test_db):
    """A token that has used all its requests should not be valid."""
    token = await create_token(company="ExhaustedCorp", max_requests=2)

    await increment_usage(token.token_id)
    await increment_usage(token.token_id)

    fetched = await get_token(token.token_id)
    assert fetched.used_requests == 2
    assert fetched.remaining_requests == 0
    assert fetched.is_valid is False


@pytest.mark.asyncio
async def test_list_tokens_excludes_revoked(setup_test_db):
    """By default, list_tokens should not include revoked tokens."""
    t1 = await create_token(company="ActiveCorp")
    t2 = await create_token(company="RevokedCorp")
    await revoke_token(t2.token_id)

    tokens = await list_tokens(include_revoked=False)
    token_ids = [t.token_id for t in tokens]
    assert t1.token_id in token_ids
    assert t2.token_id not in token_ids


@pytest.mark.asyncio
async def test_list_tokens_includes_revoked(setup_test_db):
    """With include_revoked=True, all tokens should appear."""
    t1 = await create_token(company="ActiveCorp")
    t2 = await create_token(company="RevokedCorp")
    await revoke_token(t2.token_id)

    tokens = await list_tokens(include_revoked=True)
    token_ids = [t.token_id for t in tokens]
    assert t1.token_id in token_ids
    assert t2.token_id in token_ids


@pytest.mark.asyncio
async def test_list_tokens_empty(setup_test_db):
    """list_tokens should return an empty list when no tokens exist."""
    tokens = await list_tokens()
    assert tokens == []