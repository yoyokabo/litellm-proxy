"""Runtime entity overlay: custom tier-3 labels and replacement policy.

Revision ID: 0002_custom_entities
Revises: 0001_initial
Create Date: 2026-09-21

The table an administrator edits at runtime. The YAML files remain the
baseline; this layers on top. Note there is still no column anywhere that holds
a matched value -- an overlay row holds *policy*, including the surrogate names
an entity is replaced with, which are fixed strings chosen by an operator and
never anything a user typed.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002_custom_entities"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "custom_entities",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("entity_type", sa.String(length=64), nullable=False),
        # The natural-language label tier 3 (GLiNER2) is conditioned on. NULL
        # means this row only overrides policy for an entity another tier finds.
        sa.Column("gliner_prompt", sa.String(length=200), nullable=True),
        sa.Column("category", sa.String(length=32), nullable=True),
        sa.Column("action", sa.String(length=16), nullable=True),
        sa.Column("score_threshold", sa.Float(), nullable=True),
        sa.Column("placeholder", sa.String(length=64), nullable=True),
        sa.Column("replacement_strategy", sa.String(length=32), nullable=True),
        sa.Column("replacement_value", sa.String(length=256), nullable=True),
        sa.Column("replacement_pool", sa.JSON(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("note", sa.String(length=500), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("updated_by", sa.String(length=200), nullable=True),
        sa.PrimaryKeyConstraint("id", name="pk_custom_entities"),
        sa.UniqueConstraint("entity_type", name="uq_custom_entities_entity_type"),
    )
    op.create_index("ix_custom_entities_enabled", "custom_entities", ["enabled"])


def downgrade() -> None:
    op.drop_index("ix_custom_entities_enabled", table_name="custom_entities")
    op.drop_table("custom_entities")
