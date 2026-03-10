from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Langfuse --
    langfuse_secret_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # -- LLM --
    openai_api_key: str = ""

    # -- Admin --
    admin_api_key: str = "change-me-to-a-long-random-string"

    # -- Database --
    database_path: str = "data/citeguard.db"

    # -- Citation APIs (optional) --
    crossref_mailto: str = ""
    semantic_scholar_api_key: str = ""
    ncbi_api_key: str = ""

    @property
    def database_url(self) -> str:
        return str(Path(self.database_path).resolve())


def get_settings() -> Settings:
    return Settings()