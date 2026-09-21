"""The runtime overlay: administrator changes layered onto the YAML baseline.

``EntityOverlay`` is deliberately independent of SQLAlchemy. The merge rules
are the part that decides what gets masked, so they are testable without a
database, and the DB row is just one way to produce one.

Merge rules, and the reasoning for each:

* An overlay for an entity the baseline does not define **adds** it. That is
  how a new tier-3 label reaches the detector without a release.
* An overlay for an entity the baseline does define **overrides only the
  fields it sets**. A row that changes a replacement must not silently reset
  the threshold somebody tuned.
* ``enabled: false`` **disables** the entity -- it stops being detected. It
  does not delete the baseline entry, so the row remains as the record of who
  turned it off.
* An overlay adding a *new* entity must carry a ``gliner_prompt``. Without one
  nothing would ever detect it, and an entity that can never fire is a policy
  entry that quietly does nothing.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pii_service.policy.models import EntityAction, EntityCategory, EntityPolicy
from pii_service.policy.replacement import ReplacementRule

__all__ = ["EntityOverlay", "apply_overlay"]

_ENTITY_TYPE_PATTERN = r"^[A-Z][A-Z0-9_]{1,63}$"


class EntityOverlay(BaseModel):
    """One administrator-managed entity type or policy override."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    entity_type: str = Field(pattern=_ENTITY_TYPE_PATTERN)

    # The natural-language prompt GLiNER2 is conditioned on. Required when this
    # overlay introduces an entity the baseline does not already detect.
    gliner_prompt: str | None = Field(default=None, min_length=2, max_length=200)

    category: EntityCategory | None = None
    action: EntityAction | None = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    placeholder: str | None = Field(default=None, min_length=1, max_length=64)
    replacement: ReplacementRule | None = None

    enabled: bool = True
    note: str | None = Field(default=None, max_length=500)
    updated_by: str | None = None
    updated_at: datetime | None = None

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.placeholder is not None and not self.placeholder.strip():
            raise ValueError("placeholder must not be blank")
        return self

    def merged_onto(self, base: EntityPolicy | None) -> EntityPolicy:
        """Apply this overlay to a baseline entry, or synthesise a new one."""
        if base is None:
            return EntityPolicy(
                entity_type=self.entity_type,
                category=self.category or EntityCategory.OTHER,
                action=self.action or EntityAction.MASK,
                score_threshold=0.5 if self.score_threshold is None else self.score_threshold,
                placeholder=self.placeholder or f"<{self.entity_type}>",
                tier=3 if self.gliner_prompt else None,
                gliner_prompt=self.gliner_prompt,
            )
        return EntityPolicy(
            entity_type=base.entity_type,
            category=self.category or base.category,
            action=self.action or base.action,
            score_threshold=(
                base.score_threshold if self.score_threshold is None else self.score_threshold
            ),
            placeholder=self.placeholder or base.placeholder,
            tier=base.tier,
            gliner_prompt=self.gliner_prompt or base.gliner_prompt,
        )


def apply_overlay(
    baseline: dict[str, EntityPolicy],
    overlays: tuple[EntityOverlay, ...],
) -> dict[str, EntityPolicy]:
    """Merge overlays into the baseline entity policy.

    Disabled overlays remove the entity from the effective policy: the router
    drops findings for types it has no policy for, so a disabled entity stops
    being reported without any other code needing to know about overlays.
    """
    effective = dict(baseline)

    for overlay in overlays:
        if not overlay.enabled:
            effective.pop(overlay.entity_type, None)
            continue

        base = effective.get(overlay.entity_type)
        if base is None and not overlay.gliner_prompt:
            raise ValueError(
                f"{overlay.entity_type} is not in the baseline policy and has no "
                "gliner_prompt, so nothing would ever detect it. Give it a prompt, "
                "or add it to entities.yaml."
            )
        effective[overlay.entity_type] = overlay.merged_onto(base)

    return effective
