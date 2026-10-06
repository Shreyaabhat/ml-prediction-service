"""Application settings, read from environment variables (and a local .env file)."""
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # extra="ignore": our .env also holds Postgres/Redis settings that this phase
    # doesn't define yet. Without this, pydantic-settings rejects unknown .env keys.
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"
    model_version: str = "v1"

    # Stored as a comma-separated string. (List-typed env vars must be JSON,
    # which is awkward to write by hand.)
    cors_allowed_origins: str = "http://localhost:3000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Create settings once and reuse them."""
    return Settings()