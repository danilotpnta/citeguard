import os
import tempfile

import pytest
import pytest_asyncio

from app.db.database import init_db, set_db_path


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db(tmp_path):
    """Create a fresh temporary database for each test."""
    db_file = str(tmp_path / "test_citeguard.db")
    set_db_path(db_file)
    await init_db()
    yield db_file