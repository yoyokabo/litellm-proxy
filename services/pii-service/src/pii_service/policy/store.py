"""Persistence and hot-reload for the runtime policy overlay.

Holds the effective ``PolicyBundle`` -- YAML baseline plus administrator
overlay -- and swaps it atomically when an administrator changes something.

Why a swap rather than mutation: a request that has already read the policy
must keep seeing a consistent one for its whole analysis. Rebinding a single
attribute to a fully-built new bundle means a request either sees the old
policy or the new one, never half of each.

The detection cache is cleared on every change, because it stores the
*resolved* replacement text. Its key already includes the policy fingerprint,
so stale entries would be unreachable rather than wrong -- clearing is belt and
braces, and it keeps the hit-rate statistic honest after an edit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker

from pii_service.db.custom_entities import CustomEntity
from pii_service.policy.loader import PolicyBundle
from pii_service.policy.overlay import EntityOverlay
from pii_service.policy.replacement import ReplacementRule

__all__ = ["PolicyStore"]

logger: Final = structlog.get_logger(__name__)


def _to_overlay(row: CustomEntity) -> EntityOverlay:
    replacement = None
    if row.replacement_strategy:
        replacement = ReplacementRule(
            strategy=row.replacement_strategy,  # type: ignore[arg-type]
            value=row.replacement_value,
            pool=tuple(row.replacement_pool or ()),
        )
    return EntityOverlay(
        entity_type=row.entity_type,
        gliner_prompt=row.gliner_prompt,
        category=row.category,  # type: ignore[arg-type]
        action=row.action,  # type: ignore[arg-type]
        score_threshold=row.score_threshold,
        placeholder=row.placeholder,
        replacement=replacement,
        enabled=row.enabled,
        note=row.note,
        updated_by=row.updated_by,
        updated_at=row.updated_at,
    )


class PolicyStore:
    """The effective policy, and the means to reload it from the database."""

    def __init__(self, baseline: PolicyBundle, engine: AsyncEngine) -> None:
        self._baseline: Final = baseline
        self._sessions: Final = async_sessionmaker(engine, expire_on_commit=False)
        self._current: PolicyBundle = baseline
        self._listeners: list[Callable[[PolicyBundle], None]] = []

    def add_listener(self, listener: Callable[[PolicyBundle], None]) -> None:
        """Call this whenever the effective policy changes.

        Tier 3 uses it to pick up a new label without reloading the model. A
        listener that raises is logged and skipped: a broken listener must not
        leave the store holding a policy nobody installed.
        """
        self._listeners.append(listener)

    @property
    def current(self) -> PolicyBundle:
        """The effective policy. Read once per request, never mid-analysis."""
        return self._current

    @property
    def baseline(self) -> PolicyBundle:
        """The YAML policy with no overlay -- what the repository says."""
        return self._baseline

    async def refresh(self) -> PolicyBundle:
        """Rebuild the effective policy from the database.

        A malformed overlay row does not take the service down: it is logged
        and skipped, and the rest of the policy still applies. The alternative
        is that one bad admin edit stops the gateway masking anything at all,
        which is a far worse failure than one entity reverting to its baseline.
        """
        async with self._sessions() as session:
            rows = (
                (await session.execute(select(CustomEntity).order_by(CustomEntity.entity_type)))
                .scalars()
                .all()
            )

        overlays: list[EntityOverlay] = []
        for row in rows:
            try:
                overlays.append(_to_overlay(row))
            except Exception as exc:
                logger.error(
                    "policy.overlay.invalid_row",
                    entity_type=row.entity_type,
                    error=type(exc).__name__,
                )

        try:
            candidate = self._baseline.with_overlay(overlays)
            # Touch the derived views so an inconsistent overlay fails here,
            # while the old policy is still installed, rather than on the next
            # request.
            _ = candidate.entities, candidate.gliner_prompts
        except Exception as exc:
            logger.error(
                "policy.overlay.rejected",
                error=type(exc).__name__,
                message=str(exc),
                kept="previous policy",
            )
            return self._current

        self._current = candidate
        for listener in self._listeners:
            try:
                listener(candidate)
            except Exception as exc:
                logger.error(
                    "policy.listener_failed",
                    listener=getattr(listener, "__qualname__", repr(listener)),
                    error=type(exc).__name__,
                )
        logger.info(
            "policy.reloaded",
            overlays=len(overlays),
            entities=len(candidate.entities),
            tier3_labels=len(candidate.gliner_prompts),
            realistic_replacements=list(candidate.realistic_replacement_entities),
        )
        return candidate
