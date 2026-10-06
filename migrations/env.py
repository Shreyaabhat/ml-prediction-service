"""Alembic environment: connects using the app's own settings."""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import create_engine, pool

from app.config import get_settings
from app.db.models import Base
from app.db.session import build_database_url

config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Alembic compares this metadata to the real database to autogenerate migrations.
target_metadata = Base.metadata


def run_migrations_online() -> None:
    # NullPool: a migration is a one-shot process; no need to keep connections around.
    engine = create_engine(build_database_url(get_settings()), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    raise RuntimeError("Offline mode is not supported in this project.")
run_migrations_online()
