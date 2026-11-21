"""Environment-based application settings (prefix ``COGNITRACK_``; see .env.example)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml.config import PROJECT_ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COGNITRACK_", env_file=PROJECT_ROOT / ".env", extra="ignore")

    app_name: str = "CogniTrack AI"
    environment: str = Field("development", pattern="^(development|test|production)$")
    log_level: str = "INFO"
    # Local default works without Docker; docker-compose sets a PostgreSQL URL.
    database_url: str = f"sqlite:///{(PROJECT_ROOT / 'data' / 'cognitrack.db').as_posix()}"
    # Comma-separated API keys per role. Empty in development => auth disabled (logged loudly).
    viewer_api_keys: str = ""
    researcher_api_keys: str = ""
    admin_api_keys: str = ""
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173,http://localhost:8080"
    max_upload_mb: float = 25.0
    allowed_audio_extensions: str = ".wav,.flac,.mp3,.ogg"
    upload_dir: str = "data/uploads"
    # Raw uploaded audio is deleted after feature extraction unless explicitly retained.
    retain_raw_audio: bool = False
    pipeline_config: str | None = None

    @field_validator("database_url")
    @classmethod
    def _normalise_pg(cls, v: str) -> str:
        # Accept plain postgresql:// URLs and use the psycopg3 driver.
        return v.replace("postgresql://", "postgresql+psycopg://", 1) if v.startswith("postgresql://") else v

    def keys(self, role: str) -> set[str]:
        raw = getattr(self, f"{role}_api_keys")
        return {k.strip() for k in raw.split(",") if k.strip()}

    @property
    def auth_enabled(self) -> bool:
        return any(self.keys(r) for r in ("viewer", "researcher", "admin"))

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def allowed_extensions(self) -> set[str]:
        return {e.strip().lower() for e in self.allowed_audio_extensions.split(",") if e.strip()}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
