"""Alembic environment.

Migrations run synchronously (psycopg3's sync driver, same URL as the app's
async engine), because a migration is a one-shot startup step and an async
runner buys nothing here but complexity.

The database URL is read from the environment rather than alembic.ini so that
a password never lands in a committed file.
"""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Importing every model module is what puts the tables on Base.metadata,
# which is what `alembic check` compares against.
from pii_service.db import custom_entities as _custom_entities  # noqa: F401
from pii_service.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

_DEFAULT_URL = "postgresql+psycopg://pii:pii@pii-db:5432/pii"


def _database_url() -> str:
    return os.environ.get("PII_DATABASE_URL") or _DEFAULT_URL


def run_migrations_offline() -> None:
    """Emit SQL without a connection -- used to review a migration before applying it."""
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    section = config.get_section(config.config_ini_section, {})
    section["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(section, prefix="sqlalchemy.", poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
