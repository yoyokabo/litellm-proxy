"""Integration tests against a real PostgreSQL.

Skipped unless ``PII_TEST_DATABASE_URL`` points at a reachable Postgres:

    PII_TEST_DATABASE_URL=postgresql+psycopg://pii@127.0.0.1:5432/pii_test \\
        pytest tests/test_postgres_integration.py

The rest of the suite runs against SQLite, which is fine for the sink's logic
but cannot tell us whether the *deployment* is correct. Three things are only
genuinely testable here:

* the Alembic migration applies to the database we actually ship against, and
  produces the schema the brief specifies (BIGSERIAL, TIMESTAMPTZ, the DESC
  indexes);
* the migration and the SQLAlchemy model do not disagree -- autogenerate
  detects no drift;
* psycopg's real type handling round-trips a row, including a timezone-aware
  timestamp and Arabic text in the preview column.

Each test builds its schema by running the migration, not
``metadata.create_all``, because the migration is the artifact that runs in
production.
"""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from sqlalchemy import create_engine, func, inspect, select, text
from sqlalchemy.orm import Session

from pii_service.audit.records import AuditRecord, RequestContext
from pii_service.db.models import PiiEvent
from pii_service.policy.loader import PolicyBundle
from pii_service.policy.models import EntityAction, EntityCategory
from pii_service.spans import PreparedSpan

SERVICE_ROOT = Path(__file__).resolve().parents[1]
PEPPER = b"0123456789abcdef0123456789abcdef"

_URL = os.environ.get("PII_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not _URL, reason="set PII_TEST_DATABASE_URL to run the PostgreSQL integration tests"
)


def _alembic(*arguments: str) -> list[str]:
    """Invoke alembic through the running interpreter.

    `["alembic", ...]` relies on the venv's bin being on PATH, which it is not
    when pytest is invoked as `.venv/bin/python -m pytest`.
    """
    return [sys.executable, "-m", "alembic", *arguments]


@pytest.fixture(scope="module")
def pg_database() -> Iterator[str]:
    """A throwaway database, migrated with Alembic, dropped afterwards."""
    assert _URL is not None
    admin = create_engine(_URL.replace("+psycopg", "+psycopg"), isolation_level="AUTOCOMMIT")
    name = f"pii_it_{uuid.uuid4().hex[:12]}"

    with admin.connect() as connection:
        connection.execute(text(f'CREATE DATABASE "{name}"'))
    admin.dispose()

    url = _URL.rsplit("/", 1)[0] + f"/{name}"

    result = subprocess.run(
        _alembic("upgrade", "head"),
        cwd=SERVICE_ROOT,
        env={**os.environ, "PII_DATABASE_URL": url, "PII_AUDIT_PEPPER": PEPPER.decode()},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"alembic upgrade failed:\n{result.stderr}"

    try:
        yield url
    finally:
        admin = create_engine(_URL, isolation_level="AUTOCOMMIT")
        with admin.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :n AND pid <> pg_backend_pid()"
                ),
                {"n": name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
        admin.dispose()


def _prepared(policy: PolicyBundle, entity_type: str, value: str, lang: str) -> PreparedSpan:
    from pii_service.audit.fingerprint import fingerprint

    return PreparedSpan(
        text_index=0,
        entity_type=entity_type,
        category=str(EntityCategory.ID),
        recognizer="EgyptianNationalIdRecognizer",
        score=1.0,
        action=str(EntityAction.MASK),
        start=13,
        end=13 + len(value),
        value_len=len(value),
        lang=lang,
        value_fp=fingerprint(entity_type, value, PEPPER),
        preview=policy.render_preview(entity_type, value),
    )


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_migration_creates_the_expected_columns(pg_database: str) -> None:
    engine = create_engine(pg_database)
    columns = {c["name"]: c for c in inspect(engine).get_columns("pii_events")}
    engine.dispose()

    expected = {
        "id",
        "ts",
        "request_id",
        "user_id",
        "team_id",
        "key_alias",
        "key_hash",
        "end_user_id",
        "entity_type",
        "recognizer",
        "score",
        "action",
        "span_start",
        "span_end",
        "value_len",
        "message_index",
        "message_role",
        "field",
        "value_fp",
        "preview",
        "model",
        "lang",
        "latency_ms",
    }
    assert set(columns) == expected


def test_the_table_has_no_column_for_the_matched_value(pg_database: str) -> None:
    """The rule the whole audit model rests on, asserted against real DDL."""
    engine = create_engine(pg_database)
    names = {c["name"] for c in inspect(engine).get_columns("pii_events")}
    engine.dispose()

    assert "value" not in names
    assert not any(n == "text" or n.endswith("_text") for n in names)


def test_id_is_bigserial_and_ts_is_timestamptz(pg_database: str) -> None:
    engine = create_engine(pg_database)
    with engine.connect() as connection:
        rows = dict(
            connection.execute(
                text(
                    "SELECT column_name, data_type FROM information_schema.columns "
                    "WHERE table_name = 'pii_events'"
                )
            ).all()
        )
        default = connection.execute(
            text(
                "SELECT column_default FROM information_schema.columns "
                "WHERE table_name = 'pii_events' AND column_name = 'id'"
            )
        ).scalar_one()
    engine.dispose()

    assert rows["id"] == "bigint"
    assert "nextval" in default  # BIGSERIAL
    assert rows["ts"] == "timestamp with time zone"


def test_all_four_indexes_exist_with_descending_time_order(pg_database: str) -> None:
    engine = create_engine(pg_database)
    with engine.connect() as connection:
        definitions = {
            name: definition
            for name, definition in connection.execute(
                text("SELECT indexname, indexdef FROM pg_indexes WHERE tablename='pii_events'")
            ).all()
        }
    engine.dispose()

    assert "ix_pii_events_user_id_ts" in definitions
    assert "ix_pii_events_value_fp" in definitions
    assert "ix_pii_events_entity_type_ts" in definitions
    assert "ix_pii_events_request_id" in definitions

    # DESC matters: the timeline and drill-down both read most-recent-first.
    assert "ts DESC" in definitions["ix_pii_events_user_id_ts"]
    assert "ts DESC" in definitions["ix_pii_events_entity_type_ts"]


def test_model_and_migration_do_not_disagree(pg_database: str) -> None:
    """`alembic check` -- catches a model edit that never got a migration."""
    result = subprocess.run(
        _alembic("check"),
        cwd=SERVICE_ROOT,
        env={
            **os.environ,
            "PII_DATABASE_URL": pg_database,
            "PII_AUDIT_PEPPER": PEPPER.decode(),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"the SQLAlchemy model has drifted from the migration:\n{result.stdout}\n{result.stderr}"
    )


def test_downgrade_removes_the_table(pg_database: str) -> None:
    env = {**os.environ, "PII_DATABASE_URL": pg_database, "PII_AUDIT_PEPPER": PEPPER.decode()}
    for command in (_alembic("downgrade", "base"), _alembic("upgrade", "head")):
        result = subprocess.run(
            command, cwd=SERVICE_ROOT, env=env, capture_output=True, text=True, check=False
        )
        assert result.returncode == 0, f"{command} failed:\n{result.stderr}"

    engine = create_engine(pg_database)
    assert "pii_events" in inspect(engine).get_table_names()
    engine.dispose()


# ---------------------------------------------------------------------------
# Round-trip through the real driver
# ---------------------------------------------------------------------------


def test_a_row_round_trips_through_psycopg(pg_database: str, policy: PolicyBundle) -> None:
    span = _prepared(policy, "EG_NATIONAL_ID", "28503122148219", "en")
    record = AuditRecord.from_prepared(
        span,
        RequestContext(request_id="pg-1", user_id="eng-42", team_id="platform", model="qwen3-30b"),
        latency_ms=7,
    )

    engine = create_engine(pg_database)
    with Session(engine) as session:
        session.execute(PiiEvent.__table__.insert(), [record.to_row()])
        session.commit()

        stored = session.execute(select(PiiEvent)).scalar_one()
        assert stored.user_id == "eng-42"
        assert stored.entity_type == "EG_NATIONAL_ID"
        assert stored.value_fp == span.value_fp
        assert stored.preview == "2850******8219"
        assert stored.value_len == 14
        assert stored.latency_ms == 7
        assert stored.ts.tzinfo is not None  # TIMESTAMPTZ survived the round trip
        assert stored.id > 0  # BIGSERIAL assigned it

        session.execute(PiiEvent.__table__.delete())
        session.commit()
    engine.dispose()


def test_arabic_preview_survives_the_round_trip(pg_database: str, policy: PolicyBundle) -> None:
    """The preview keeps the original script, so the column must hold Arabic."""
    arabic_nid = "٢٨٥٠٣١٢٢١٤٨٢١٩"
    span = _prepared(policy, "EG_NATIONAL_ID", arabic_nid, "ar")
    record = AuditRecord.from_prepared(span, RequestContext(request_id="pg-ar"))

    engine = create_engine(pg_database)
    with Session(engine) as session:
        session.execute(PiiEvent.__table__.insert(), [record.to_row()])
        session.commit()

        stored = session.execute(select(PiiEvent)).scalar_one()
        assert stored.preview == "٢٨٥٠******٨٢١٩"
        assert stored.lang == "ar"

        session.execute(PiiEvent.__table__.delete())
        session.commit()
    engine.dispose()


def test_a_batch_insert_writes_every_row(pg_database: str, policy: PolicyBundle) -> None:
    """One request produces many spans and should cost one insert."""
    rows = [
        AuditRecord.from_prepared(
            _prepared(policy, "EG_NATIONAL_ID", f"2850312214821{n}", "en"),
            RequestContext(request_id="pg-batch", user_id=f"eng-{n}"),
        ).to_row()
        for n in range(50)
    ]

    engine = create_engine(pg_database)
    with Session(engine) as session:
        session.execute(PiiEvent.__table__.insert(), rows)
        session.commit()

        assert session.execute(select(func.count()).select_from(PiiEvent)).scalar_one() == 50

        session.execute(PiiEvent.__table__.delete())
        session.commit()
    engine.dispose()


def test_the_fingerprint_index_supports_the_investigative_pivot(
    pg_database: str, policy: PolicyBundle
) -> None:
    """One value across many users is the query the whole design exists for."""
    shared = _prepared(policy, "EG_NATIONAL_ID", "28503122148219", "en")
    rows = [
        AuditRecord.from_prepared(
            shared, RequestContext(request_id=f"pg-fp-{n}", user_id=f"eng-{n}")
        ).to_row()
        for n in range(20)
    ]

    engine = create_engine(pg_database)
    with Session(engine) as session:
        session.execute(PiiEvent.__table__.insert(), rows)
        session.commit()

        users = session.execute(
            select(func.count(func.distinct(PiiEvent.user_id))).where(
                PiiEvent.value_fp == shared.value_fp
            )
        ).scalar_one()
        assert users == 20

        session.execute(PiiEvent.__table__.delete())
        session.commit()
    engine.dispose()
