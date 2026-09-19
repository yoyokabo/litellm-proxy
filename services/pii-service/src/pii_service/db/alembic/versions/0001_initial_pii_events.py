"""Initial audit schema: pii_events.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-19

Mirrors brief §6. Note what is absent: there is no column for the matched
value, and adding one would defeat the purpose of the table.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001_initial"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "pii_events",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column(
            "ts",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("request_id", sa.String(), nullable=False),
        # identity
        sa.Column("user_id", sa.String(), nullable=True),
        sa.Column("team_id", sa.String(), nullable=True),
        sa.Column("key_alias", sa.String(), nullable=True),
        sa.Column("key_hash", sa.String(), nullable=True),
        sa.Column("end_user_id", sa.String(), nullable=True),
        # what
        sa.Column("entity_type", sa.String(), nullable=False),
        sa.Column("recognizer", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("action", sa.String(), nullable=False),
        # where, without the value
        sa.Column("span_start", sa.Integer(), nullable=True),
        sa.Column("span_end", sa.Integer(), nullable=True),
        sa.Column("value_len", sa.Integer(), nullable=True),
        sa.Column("message_index", sa.Integer(), nullable=True),
        sa.Column("message_role", sa.String(), nullable=True),
        sa.Column("field", sa.String(), nullable=True),
        # correlation, without the value
        sa.Column("value_fp", sa.String(length=32), nullable=True),
        sa.Column("preview", sa.String(), nullable=True),
        # context
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("lang", sa.String(length=8), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_pii_events"),
    )

    # "What did this user do, most recent first" -- the admin timeline.
    op.create_index("ix_pii_events_user_id_ts", "pii_events", ["user_id", sa.text("ts DESC")])
    # The investigative pivot: every other occurrence of one value.
    op.create_index("ix_pii_events_value_fp", "pii_events", ["value_fp"])
    # The stacked-area timeline, per entity type.
    op.create_index(
        "ix_pii_events_entity_type_ts", "pii_events", ["entity_type", sa.text("ts DESC")]
    )
    # Drill-down from a single request.
    op.create_index("ix_pii_events_request_id", "pii_events", ["request_id"])


def downgrade() -> None:
    op.drop_index("ix_pii_events_request_id", table_name="pii_events")
    op.drop_index("ix_pii_events_entity_type_ts", table_name="pii_events")
    op.drop_index("ix_pii_events_value_fp", table_name="pii_events")
    op.drop_index("ix_pii_events_user_id_ts", table_name="pii_events")
    op.drop_table("pii_events")
