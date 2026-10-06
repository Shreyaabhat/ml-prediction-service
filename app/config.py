"""Application settings, read from environment variables (and a local .env file)."""
from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_env: str = "development"
    log_level: str = "INFO"
    model_version: str = "v1"
    cors_allowed_origins: str = "http://localhost:3000"

    # Database. User, password, and name have NO defaults on purpose: the app
    # refuses to start without them, so credentials can never be hard-coded.
    postgres_user: str
    postgres_password: SecretStr  # SecretStr hides the value in repr()/logs
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5433

    db_pool_size: int = 5
    db_max_overflow: int = 5
    db_pool_timeout_seconds: float = 5.0
    db_connect_timeout_seconds: int = 3
    db_statement_timeout_ms: int = 5000

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_allowed_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    """Create settings once and reuse them."""
    return Settings()