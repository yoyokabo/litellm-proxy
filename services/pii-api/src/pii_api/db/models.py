"""Tables this service owns.

It shares a database with pii-service but not a schema: ``pii_events`` belongs
to pii-service and is **read-only from here**. The separation matters because
brief §3 makes pii-service the single audit writer, and an audit trail with two
writers is one where nobody can say which component produced a row.

That is also why these tables carry their own Alembic chain, with its own
version table -- see ``db/alembic/env.py``. Two independent migration
histories in one database, each owning its own tables.

The PII rule applies here unchanged: **no table below stores message text or a
matched value.** ``span_flags`` annotates an audit row by id; it does not copy
anything out of it.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

__all__ = ["AppSession", "AppUser", "Base", "SpanFlag", "UserLlmKey"]

_NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_N_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=_NAMING_CONVENTION)


class AppUser(Base):
    """A person who can log into the web app.

    Distinct from LiteLLM's own user table on purpose. This is the identity the
    web app authenticates; ``UserLlmKey`` maps it onto a LiteLLM virtual key,
    and that key is what puts ``user_api_key_user_id`` on the audit rows the
    chat produces.
    """

    __tablename__ = "app_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False, unique=True)
    # scrypt$n$r$p$salt$hash -- see auth/passwords.py. Never a bare digest.
    password_hash: Mapped[str] = mapped_column(String(512), nullable=False)

    # NOTE: there is deliberately no `role` column. This build has exactly one
    # permission level -- every authenticated user has full visibility over
    # the audit trail and the chat. Brief §10 sketches an `auditor` role that
    # is read-only; it is not implemented, and a column that always held
    # 'admin' would look like an access control that is not there.
    #
    # Adding it later is one migration plus one dependency in auth/deps.py.
    # Until then, "who can log in" is the whole of the authorization model,
    # so account creation is the control that matters.

    # The bootstrap account starts true, and admin routes refuse to serve
    # until it is rotated. A deployment whose admin password is the one from
    # the install guide is a deployment with no admin password.
    must_change_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    disabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class AppSession(Base):
    """A logged-in session.

    Opaque random tokens in a table rather than a signed cookie: a signed
    cookie cannot be revoked, and "log this auditor out now" is a request that
    actually gets made about an account with read access to an audit trail.
    Only the token's hash is stored, so a dump of this table does not let
    anyone resume a session.
    """

    __tablename__ = "app_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    token_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    __table_args__ = (Index("ix_app_sessions_user_id", "user_id"),)


class UserLlmKey(Base):
    """One LiteLLM virtual key per application user (brief §3).

    The key is what carries identity to the guardrail: LiteLLM puts the key's
    ``user_id`` into request metadata, the guardrail forwards it, and it lands
    in ``pii_events.user_id``. Without this mapping every chat message would
    audit as the same shared key, and the admin view's central question --
    "who pasted this" -- would have no answer.
    """

    __tablename__ = "user_llm_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(
        ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    # The LiteLLM user id this key is bound to; matches pii_events.user_id.
    llm_user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    virtual_key: Mapped[str] = mapped_column(String(256), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )


class SpanFlag(Base):
    """A reviewer's verdict on one detected span (brief §10, and phase 5's input).

    References a ``pii_events`` row by id and stores a judgement about it. It
    deliberately does **not** copy the entity type, the preview or anything
    else out of that row: this table is an annotation, and duplicating audit
    fields here would create a second, unmanaged copy of the audit trail.

    There is no foreign key to ``pii_events`` because that table belongs to
    pii-service's migration chain, and a cross-chain constraint would make
    either service's migrations depend on the other's ordering.
    """

    __tablename__ = "span_flags"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    event_id: Mapped[int] = mapped_column(Integer, nullable=False)
    # 'false_positive' or 'confirmed'. Confirmed matters as much as the other:
    # phase 5 needs positive examples too, not only corrections.
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    flagged_by: Mapped[int] = mapped_column(
        ForeignKey("app_users.id", ondelete="CASCADE"), nullable=False
    )
    note: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # One verdict per reviewer per span; re-flagging updates in place.
        UniqueConstraint("event_id", "flagged_by", name="uq_span_flags_event_id_flagged_by"),
        Index("ix_span_flags_event_id", "event_id"),
    )
