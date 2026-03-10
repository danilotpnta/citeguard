from pathlib import Path
from pydantic import Field
from omegaconf import OmegaConf

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # -- Langfuse --
    langfuse_secret_key: str = Field(..., description="Langfuse secret key")
    langfuse_public_key: str = Field(..., description="Langfuse public key")
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # -- LLM --
    openai_api_key: str = Field(..., description="OpenAI API key")
    google_api_key: str = Field(..., description="Google API key for Gemini access")
    groq_api_key: str = Field(..., description="Groq API key for ingredient extraction")

    # -- Admin --
    admin_api_key: str = "change-me-to-a-long-random-string"

    # -- Database --
    database_path: str = "data/citeguard.db"

    # -- Citation APIs (optional) --
    crossref_mailto: str = ""
    semantic_scholar_api_key: str = ""
    ncbi_api_key: str = ""

    # -- Logging --
    log_level: str = Field(..., description="Logging level setting")
    environment: str = Field(..., description="FastAPI setting")

    @property
    def database_url(self) -> str:
        return str(Path(self.database_path).resolve())


def get_settings() -> Settings:
    return Settings()


class Config:
    def __init__(self, path: str = "config/pipeline.yml"):
        self.project_root = Path.cwd()
        self.cfg_path = self.project_root / path
        self.cfg = self._load()

    def _load(self):
        cfg = OmegaConf.load(self.cfg_path)
        cfg = self._resolve_yamls(cfg)
        cfg = self._resolve_pipeline_paths(cfg)
        return cfg

    def _resolve_yamls(self, obj):
        """Recursively load any YAML file references."""
        if OmegaConf.is_dict(obj):
            for key in list(obj.keys()):
                value = obj[key]
                value_str = str(value) if value is not None else None

                if value_str and value_str.endswith((".yml", ".yaml")):
                    yaml_path = self.project_root / value_str
                    if yaml_path.exists():
                        obj[key] = self._resolve_yamls(OmegaConf.load(yaml_path))
                else:
                    obj[key] = self._resolve_yamls(value)
        return obj

    def _resolve_pipeline_paths(self, cfg):
        """Resolve pipeline file paths using base/version/files convention."""
        if not hasattr(cfg, "pipeline"):
            return cfg

        for workflow_name, workflow in cfg.pipeline.items():
            if not hasattr(workflow, "files"):
                continue

            base = Path(workflow.base) / workflow.version

            for key, filename in workflow.files.items():
                full_path = self.project_root / base / filename

                if full_path.exists() and filename.endswith((".yml", ".yaml")):
                    workflow[key] = self._resolve_yamls(OmegaConf.load(full_path))
                else:
                    workflow[key] = str(base / filename)

        return cfg

    def __getattr__(self, name):
        return getattr(self.cfg, name)


# Pipeline Settings
config = Config()

# Global settings instance
settings = Settings()
