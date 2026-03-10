from datetime import UTC, datetime

from pydantic import BaseModel

from app.db.database import get_connection


class UsageLog(BaseModel):
    id: int | None = None
    token_id: str
    timestamp: datetime
    ip_address: str | None = None
    method: str | None = None
    path: str | None = None
    input_size: int = 0
    status_code: int | None = None


class UsageSummary(BaseModel):
    token_id: str
    total_requests: int
    last_used: datetime | None
    unique_ips: int


async def log_request(
    token_id: str,
    ip_address: str | None = None,
    method: str | None = None,
    path: str | None = None,
    input_size: int = 0,
    status_code: int | None = None,
) -> None:
    """Log a single API request."""
    conn = await get_connection()
    try:
        await conn.execute(
            """
            INSERT INTO usage_logs (token_id, timestamp, ip_address, method, path, input_size, status_code)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                token_id,
                datetime.now(UTC).isoformat(),
                ip_address,
                method,
                path,
                input_size,
                status_code,
            ),
        )
        await conn.commit()
    finally:
        await conn.close()


async def get_usage_by_token(token_id: str, limit: int = 50) -> list[UsageLog]:
    """Get recent usage logs for a specific token."""
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            """
            SELECT * FROM usage_logs
            WHERE token_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (token_id, limit),
        )
        rows = await cursor.fetchall()
        return [
            UsageLog(
                id=row["id"],
                token_id=row["token_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                ip_address=row["ip_address"],
                method=row["method"],
                path=row["path"],
                input_size=row["input_size"],
                status_code=row["status_code"],
            )
            for row in rows
        ]
    finally:
        await conn.close()


async def get_usage_summary(token_id: str) -> UsageSummary:
    """Get aggregated usage stats for a token."""
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            """
            SELECT
                COUNT(*) as total_requests,
                MAX(timestamp) as last_used,
                COUNT(DISTINCT ip_address) as unique_ips
            FROM usage_logs
            WHERE token_id = ?
            """,
            (token_id,),
        )
        row = await cursor.fetchone()
        return UsageSummary(
            token_id=token_id,
            total_requests=row["total_requests"],
            last_used=datetime.fromisoformat(row["last_used"]) if row["last_used"] else None,
            unique_ips=row["unique_ips"],
        )
    finally:
        await conn.close()