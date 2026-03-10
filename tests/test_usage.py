import pytest

from app.db.tokens import create_token
from app.db.usage import get_usage_by_token, get_usage_summary, log_request


@pytest.mark.asyncio
async def test_log_request(setup_test_db):
    """Logging a request should create a usage_logs entry."""
    token = await create_token(company="LogCorp")

    await log_request(
        token_id=token.token_id,
        ip_address="192.168.1.1",
        method="POST",
        path="/verify",
        input_size=1024,
        status_code=200,
    )

    logs = await get_usage_by_token(token.token_id)
    assert len(logs) == 1
    assert logs[0].token_id == token.token_id
    assert logs[0].ip_address == "192.168.1.1"
    assert logs[0].method == "POST"
    assert logs[0].path == "/verify"
    assert logs[0].input_size == 1024
    assert logs[0].status_code == 200


@pytest.mark.asyncio
async def test_log_multiple_requests(setup_test_db):
    """Multiple logged requests should all be retrievable."""
    token = await create_token(company="MultiLogCorp")

    for i in range(5):
        await log_request(
            token_id=token.token_id,
            ip_address=f"10.0.0.{i}",
            method="POST",
            path="/verify",
            input_size=100 * (i + 1),
        )

    logs = await get_usage_by_token(token.token_id)
    assert len(logs) == 5


@pytest.mark.asyncio
async def test_get_usage_by_token_respects_limit(setup_test_db):
    """get_usage_by_token should respect the limit parameter."""
    token = await create_token(company="LimitCorp")

    for i in range(10):
        await log_request(token_id=token.token_id, ip_address="1.2.3.4")

    logs = await get_usage_by_token(token.token_id, limit=3)
    assert len(logs) == 3


@pytest.mark.asyncio
async def test_get_usage_summary(setup_test_db):
    """Usage summary should aggregate correctly."""
    token = await create_token(company="SummaryCorp")

    await log_request(token_id=token.token_id, ip_address="10.0.0.1")
    await log_request(token_id=token.token_id, ip_address="10.0.0.2")
    await log_request(token_id=token.token_id, ip_address="10.0.0.1")

    summary = await get_usage_summary(token.token_id)
    assert summary.token_id == token.token_id
    assert summary.total_requests == 3
    assert summary.unique_ips == 2
    assert summary.last_used is not None


@pytest.mark.asyncio
async def test_get_usage_summary_no_usage(setup_test_db):
    """Summary for a token with no usage should return zeroes."""
    token = await create_token(company="EmptyCorp")

    summary = await get_usage_summary(token.token_id)
    assert summary.total_requests == 0
    assert summary.unique_ips == 0
    assert summary.last_used is None


@pytest.mark.asyncio
async def test_usage_logs_isolated_between_tokens(setup_test_db):
    """Usage logs for one token should not leak into another."""
    t1 = await create_token(company="Corp1")
    t2 = await create_token(company="Corp2")

    await log_request(token_id=t1.token_id, ip_address="1.1.1.1")
    await log_request(token_id=t1.token_id, ip_address="1.1.1.1")
    await log_request(token_id=t2.token_id, ip_address="2.2.2.2")

    logs_t1 = await get_usage_by_token(t1.token_id)
    logs_t2 = await get_usage_by_token(t2.token_id)

    assert len(logs_t1) == 2
    assert len(logs_t2) == 1
    assert all(log.token_id == t1.token_id for log in logs_t1)
    assert all(log.token_id == t2.token_id for log in logs_t2)