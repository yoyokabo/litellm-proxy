"""The audit schema.

One table, and the discipline of the whole system is in which columns it does
*not* have (brief §6). There is no ``value`` column, and there will never be
one. Storing matched values here would build a searchable PII database next to
the LLM gateway -- a worse asset to lose than the original leak, and for a
government client an incident rather than a finding.

The partition mirrors LiteLLM's own post-incident fix: identity, entity type,
recognizer, score, action, offsets, timings and counts are kept; anything
carrying the prompt is not. What replaces the value is ``value_fp``, a
pepper-keyed HMAC that supports correlation without disclosure, and
``preview``, which is NULL unless ``preview_policy.yaml`` says otherwise.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    BigInteger,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    String,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["Base", "PiiEvent"]

# Explicit naming so Alembic autogenerate produces stable, reviewable names
# rather than backend defaults that churn between versions.
_NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


class PiiEvent(Base):
    """One detected span. A request with many spans produces many rows."""

    __tablename__ = "pii_events"

    # BIGSERIAL on Postgres. The SQLite variant is not a tuning choice: SQLite
    # only auto-assigns a primary key declared exactly INTEGER (the ROWID
    # alias), so a BIGINT key there fails with a NOT NULL violation on insert
    # and the audit tests could never exercise a real insert path.
    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"),
        primary_key=True,
        autoincrement=True,
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    request_id: Mapped[str] = mapped_column(String, nullable=False)  # litellm_call_id

    # -- identity ---------------------------------------------------------
    user_id: Mapped[str | None] = mapped_column(String)  # user_api_key_user_id
    team_id: Mapped[str | None] = mapped_column(String)  # user_api_key_team_id
    key_alias: Mapped[str | None] = mapped_column(String)
    key_hash: Mapped[str | None] = mapped_column(String)
    end_user_id: Mapped[str | None] = mapped_column(String)  # request body "user"

    # -- what -------------------------------------------------------------
    entity_type: Mapped[str] = mapped_column(String, nullable=False)
    recognizer: Mapped[str] = mapped_column(String, nullable=False)
    score: Mapped[float | None] = mapped_column(Float)
    action: Mapped[str] = mapped_column(String, nullable=False)  # MASK | BLOCK | ALLOW

    # -- where, without the value -----------------------------------------
    span_start: Mapped[int | None] = mapped_column(Integer)
    span_end: Mapped[int | None] = mapped_column(Integer)
    value_len: Mapped[int | None] = mapped_column(Integer)
    message_index: Mapped[int | None] = mapped_column(Integer)
    message_role: Mapped[str | None] = mapped_column(String)
    field: Mapped[str | None] = mapped_column(String)  # content | tool_call.args | system

    # -- correlation, without the value -----------------------------------
    # HMAC-SHA256(canonical(value), pepper)[:16]. The pepper lives in the
    # environment, never in this database -- that separation is the entire
    # security property, so a backup of this table is not a PII store.
    value_fp: Mapped[str | None] = mapped_column(String(32))
    preview: Mapped[str | None] = mapped_column(String)  # '2850******5974' or NULL

    # -- context ----------------------------------------------------------
    model: Mapped[str | None] = mapped_column(String)
    lang: Mapped[str | None] = mapped_column(String(8))
    latency_ms: Mapped[int | None] = mapped_column(Integer)

    __table_args__ = (
        # "what did this user do, most recent first" -- the admin timeline.
        Index("ix_pii_events_user_id_ts", "user_id", ts.desc()),
        # The investigative pivot: every other occurrence of one value.
        Index("ix_pii_events_value_fp", "value_fp"),
        # The stacked-area timeline, per entity category.
        Index("ix_pii_events_entity_type_ts", "entity_type", ts.desc()),
        # Drill-down from a single request.
        Index("ix_pii_events_request_id", "request_id"),
    )

    def __repr__(self) -> str:
        # No preview, no fingerprint: this repr shows up in logs.
        return (
            f"PiiEvent(id={self.id} {self.entity_type} action={self.action} "
            f"user={self.user_id!r} request={self.request_id!r})"
        )
