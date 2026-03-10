import os

from app.config import Settings


def test_settings_defaults():
    """Settings should load with sensible defaults even without .env."""
    settings = Settings(
        _env_file=None,  # dont try to read .env during tests
    )
    assert settings.database_path == "data/citeguard.db"
    assert settings.admin_api_key == "change-me-to-a-long-random-string"
    assert settings.langfuse_base_url == "https://cloud.langfuse.com"


def test_settings_from_env(monkeypatch):
    """Settings should pick up values from environment variables."""
    monkeypatch.setenv("ADMIN_API_KEY", "test-admin-key-123")
    monkeypatch.setenv("DATABASE_PATH", "/tmp/test.db")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")

    settings = Settings(_env_file=None)
    assert settings.admin_api_key == "test-admin-key-123"
    assert settings.database_path == "/tmp/test.db"
    assert settings.openai_api_key == "sk-test-key"


def test_database_url_resolves_to_absolute_path():
    """The database_url property should return an absolute path."""
    settings = Settings(_env_file=None, database_path="data/citeguard.db")
    assert os.path.isabs(settings.database_url)
