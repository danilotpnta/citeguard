import aiosqlite
from pathlib import Path

_db_path: str = ""


def set_db_path(path: str) -> None:
    global _db_path
    _db_path = path


def get_db_path() -> str:
    return _db_path


async def get_connection() -> aiosqlite.Connection:
    """Get a database connection. Caller is responsible for closing it."""
    db_path = get_db_path()
    if not db_path:
        raise RuntimeError("Database path not configured. Call set_db_path() first.")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(db_path)
    conn.row_factory = aiosqlite.Row
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA foreign_keys=ON")
    return conn


async def init_db() -> None:
    """Create tables if they do not exist."""
    conn = await get_connection()
    try:
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tokens (
                token_id    TEXT PRIMARY KEY,
                company     TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                expires_at  TEXT NOT NULL,
                max_requests INTEGER NOT NULL DEFAULT 50,
                used_requests INTEGER NOT NULL DEFAULT 0,
                status      TEXT NOT NULL DEFAULT 'active'
            )
        """)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS usage_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                token_id    TEXT NOT NULL,
                timestamp   TEXT NOT NULL,
                ip_address  TEXT,
                method      TEXT,
                path        TEXT,
                input_size  INTEGER DEFAULT 0,
                status_code INTEGER,
                FOREIGN KEY (token_id) REFERENCES tokens(token_id)
            )
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_token
            ON usage_logs(token_id)
        """)
        await conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp
            ON usage_logs(timestamp)
        """)
        await conn.commit()
    finally:
        await conn.close()