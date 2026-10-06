"""Database engine (with connection pool) and session factory."""
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL, Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings

logger = logging.getLogger("app.db")


def build_database_url(settings: Settings) -> URL:
    # URL.create escapes special characters in the password for us; building the
    # URL with an f-string breaks as soon as a password contains '@' or '/'.
    return URL.create(
        drivername="postgresql+psycopg",
        username=settings.postgres_user,
        password=settings.postgres_password.get_secret_value(),
        host=settings.postgres_host,
        port=settings.postgres_port,
        database=settings.postgres_db,
    )


def create_db_engine(settings: Settings) -> Engine:
    """Create the engine. This does NOT connect yet; connections open lazily."""
    return create_engine(
        build_database_url(settings),
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_timeout=settings.db_pool_timeout_seconds,  # fail fast if the pool is exhausted
        pool_pre_ping=True,  # detect dead connections before using them
        pool_recycle=1800,  # retire connections after 30 min
        connect_args={
            "connect_timeout": settings.db_connect_timeout_seconds,
            # Server-side cap so one slow query can't hold a connection forever.
            "options": f"-c statement_timeout={settings.db_statement_timeout_ms}",
        },
    )


def create_session_factory(engine: Engine) -> sessionmaker[Session]:
    # expire_on_commit=False: objects stay readable after commit without re-querying.
    return sessionmaker(bind=engine, expire_on_commit=False)


def check_database(engine: Engine) -> bool:
    """Cheap connectivity check used by /ready."""
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except SQLAlchemyError as exc:
        logger.warning("database_check_failed", extra={"fields": {"error": type(exc).__name__}})
        return False