"""app_users, app_sessions, user_llm_keys, span_flags

Revision ID: 0001_app_tables
Revises:
Create Date: 2026-09-20

The tables the web backend owns. pii_events is not here -- it belongs to
pii-service's chain, which runs against the same database with its own
version table.
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "0001_app_tables"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "app_users",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("email", sa.String(length=320), nullable=False),
        sa.Column("password_hash", sa.String(length=512), nullable=False),
        sa.Column("must_change_password", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("disabled", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("last_login_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_app_users")),
        sa.UniqueConstraint("email", name=op.f("uq_app_users_email")),
    )

    op.create_table(
        "app_sessions",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["app_users.id"],
            name=op.f("fk_app_sessions_user_id"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_app_sessions")),
        sa.UniqueConstraint("token_hash", name=op.f("uq_app_sessions_token_hash")),
    )
    op.create_index("ix_app_sessions_user_id", "app_sessions", ["user_id"])

    op.create_table(
        "user_llm_keys",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("llm_user_id", sa.String(length=128), nullable=False),
        sa.Column("virtual_key", sa.String(length=256), nullable=False),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["app_users.id"],
            name=op.f("fk_user_llm_keys_user_id"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_user_llm_keys")),
        sa.UniqueConstraint("user_id", name=op.f("uq_user_llm_keys_user_id")),
    )

    op.create_table(
        "span_flags",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.Integer(), nullable=False),
        sa.Column("verdict", sa.String(length=32), nullable=False),
        sa.Column("flagged_by", sa.Integer(), nullable=False),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ["flagged_by"],
            ["app_users.id"],
            name=op.f("fk_span_flags_flagged_by"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_span_flags")),
        sa.UniqueConstraint("event_id", "flagged_by", name="uq_span_flags_event_id_flagged_by"),
    )
    op.create_index("ix_span_flags_event_id", "span_flags", ["event_id"])


def downgrade() -> None:
    op.drop_index("ix_span_flags_event_id", table_name="span_flags")
    op.drop_table("span_flags")
    op.drop_table("user_llm_keys")
    op.drop_index("ix_app_sessions_user_id", table_name="app_sessions")
    op.drop_table("app_sessions")
    op.drop_table("app_users")
