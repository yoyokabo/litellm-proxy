"""Alembic environment for pii-api's tables.

``version_table`` is the load-bearing setting. This service shares a database
with pii-service, and each owns a different set of tables. Two chains writing
to the same default ``alembic_version`` row would each see the other's
revision id as unknown and refuse to run -- or worse, one would stamp over the
other. Separate version tables give each service an independent history in one
database.

``include_object`` is the matching half: without it, an autogenerate run here
would see pii_events (not in this metadata) and helpfully emit a DROP TABLE.
"""

from __future__ import annotations

import os
from logging.config import fileConfig
from typing import Any

from alembic import context
from sqlalchemy import engine_from_config, pool

from pii_api.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

VERSION_TABLE = "alembic_version_api"

_DEFAULT_URL = "postgresql+psycopg://pii:pii@pii-db:5432/pii"


def _database_url() -> str:
    return os.environ.get("PII_API_DATABASE_URL") or _DEFAULT_URL


def _include_object(obj: Any, name: str | None, type_: str, *_: Any) -> bool:
    """Ignore every table this service does not own."""
    if type_ == "table":
        return name in target_metadata.tables
    return True


def run_migrations_offline() -> None:
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        version_table=VERSION_TABLE,
        include_object=_include_object,
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
            version_table=VERSION_TABLE,
            include_object=_include_object,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
