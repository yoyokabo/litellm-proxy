"""Runtime entity configuration, editable from the admin API.

Two layers, deliberately:

``config/*.yaml``   the baseline. Lives in git, goes through review, ships with
                    the image, and is mounted read-only. This is what the
                    deployment agreed to.

``custom_entities`` the overlay. Rows an administrator adds or changes at
                    runtime, for the things that cannot wait for a release: a
                    new project codename to redact, a customer identifier
                    format, a replacement switched from ``<PERSON>`` to a
                    surrogate.

The overlay can add entity types and override fields of existing ones. It
cannot delete a baseline entity -- disabling one is ``enabled = false``, which
leaves a row saying who turned it off and when. A detector that silently has no
record of being switched off is how coverage disappears.

Tier 3 (GLiNER2) is what makes adding a *detector* at runtime possible at all:
it is schema-conditioned, so a new label is a prompt passed at inference, not a
retrain. ``gliner_prompt`` is that prompt. An overlay row without one is a
policy override for an entity some other tier already finds.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from pii_service.db.models import Base

__all__ = ["CustomEntity"]


class CustomEntity(Base):
    """One administrator-managed entity type or policy override."""

    __tablename__ = "custom_entities"

    id: Mapped[int] = mapped_column(
        BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True
    )
    entity_type: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)

    # --- detection --------------------------------------------------------
    # The natural-language label GLiNER2 is conditioned on. None means this row
    # only overrides policy for an entity another tier already detects.
    gliner_prompt: Mapped[str | None] = mapped_column(String(200))

    # --- policy -----------------------------------------------------------
    category: Mapped[str | None] = mapped_column(String(32))
    action: Mapped[str | None] = mapped_column(String(16))
    score_threshold: Mapped[float | None] = mapped_column(Float)
    placeholder: Mapped[str | None] = mapped_column(String(64))

    # --- replacement ------------------------------------------------------
    # Mirrors ReplacementRule. Stored as columns rather than one JSON blob for
    # `strategy`, because it is the field an operator filters and audits on.
    replacement_strategy: Mapped[str | None] = mapped_column(String(32))
    replacement_value: Mapped[str | None] = mapped_column(String(256))
    replacement_pool: Mapped[list[str] | None] = mapped_column(JSON)

    # --- provenance -------------------------------------------------------
    # Who changed detection policy, and when. An audit table that cannot answer
    # "who turned this off" is not an audit table.
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    note: Mapped[str | None] = mapped_column(String(500))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_by: Mapped[str | None] = mapped_column(String(200))

    __table_args__ = (Index("ix_custom_entities_enabled", "enabled"),)

    def __repr__(self) -> str:
        return (
            f"CustomEntity({self.entity_type} enabled={self.enabled} "
            f"action={self.action} strategy={self.replacement_strategy})"
        )
