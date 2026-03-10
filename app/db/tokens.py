import uuid
from datetime import UTC, datetime, timedelta

from app.db.database import get_connection
from app.models.token import Token, TokenStatus


async def create_token(company: str, max_requests: int = 50, expires_in_days: int = 30) -> Token:
    """Create a new access token for a company."""
    token_id = uuid.uuid4().hex[:12]
    now = datetime.now(UTC)
    expires_at = now + timedelta(days=expires_in_days)

    conn = await get_connection()
    try:
        await conn.execute(
            """
            INSERT INTO tokens (token_id, company, created_at, expires_at, max_requests, used_requests, status)
            VALUES (?, ?, ?, ?, ?, 0, ?)
            """,
            (token_id, company, now.isoformat(), expires_at.isoformat(), max_requests, TokenStatus.ACTIVE.value),
        )
        await conn.commit()
    finally:
        await conn.close()

    return Token(
        token_id=token_id,
        company=company,
        created_at=now,
        expires_at=expires_at,
        max_requests=max_requests,
        used_requests=0,
        status=TokenStatus.ACTIVE,
    )


async def get_token(token_id: str) -> Token | None:
    """Fetch a token by its ID. Returns None if not found."""
    conn = await get_connection()
    try:
        cursor = await conn.execute("SELECT * FROM tokens WHERE token_id = ?", (token_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_token(row)
    finally:
        await conn.close()


async def revoke_token(token_id: str) -> bool:
    """Revoke a token. Returns True if the token existed and was revoked."""
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "UPDATE tokens SET status = ? WHERE token_id = ?",
            (TokenStatus.REVOKED.value, token_id),
        )
        await conn.commit()
        return cursor.rowcount > 0
    finally:
        await conn.close()


async def increment_usage(token_id: str) -> bool:
    """Increment the used_requests counter. Returns True if successful."""
    conn = await get_connection()
    try:
        cursor = await conn.execute(
            "UPDATE tokens SET used_requests = used_requests + 1 WHERE token_id = ?",
            (token_id,),
        )
        await conn.commit()
        return cursor.rowcount > 0
    finally:
        await conn.close()


async def list_tokens(include_revoked: bool = False) -> list[Token]:
    """List all tokens, optionally including revoked ones."""
    conn = await get_connection()
    try:
        if include_revoked:
            cursor = await conn.execute("SELECT * FROM tokens ORDER BY created_at DESC")
        else:
            cursor = await conn.execute(
                "SELECT * FROM tokens WHERE status != ? ORDER BY created_at DESC",
                (TokenStatus.REVOKED.value,),
            )
        rows = await cursor.fetchall()
        return [_row_to_token(row) for row in rows]
    finally:
        await conn.close()


def _row_to_token(row) -> Token:
    """Convert a database row to a Token model."""
    return Token(
        token_id=row["token_id"],
        company=row["company"],
        created_at=datetime.fromisoformat(row["created_at"]),
        expires_at=datetime.fromisoformat(row["expires_at"]),
        max_requests=row["max_requests"],
        used_requests=row["used_requests"],
        status=TokenStatus(row["status"]),
    )