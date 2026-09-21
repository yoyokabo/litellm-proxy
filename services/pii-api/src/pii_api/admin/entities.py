"""Entity-policy administration: the screen behind "Entities" in the admin menu.

Two things an operator does here, both of which change what the gateway masks
without a deployment:

* **Add a custom label.** A natural-language English prompt -- "internal
  project codename", "employee badge number" -- that tier 3 is conditioned on
  at inference. No retraining, no image rebuild.
* **Choose what a label is replaced with.** ``<PERSON>`` by default; "John Doe"
  or a pool of surrogates when a downstream model reads better with realistic
  text. The trade-offs live in pii-service's replacement module and are
  surfaced to the UI through /replacement-strategies.

**This module deliberately validates almost nothing.** pii-service owns the
policy and every rule about what a valid overlay is -- which fields may be
omitted, that a new label needs a prompt, that a surrogate pool may not be
empty -- and a second copy of those rules here would be a copy that drifts.
Bodies are forwarded as received and upstream refusals are passed back with
their status and their message, because "needs a gliner_prompt, otherwise
nothing would ever detect it" is exactly what the operator needs to read.

What this layer *does* add is identity. pii-service's admin API takes one
bearer token and has no notion of users; this backend has users, so it holds
that token as a service-to-service credential, authorises the caller itself,
and forwards their email as ``X-Admin-User`` so the change is attributed to a
person in ``custom_entities.updated_by``.
"""

from __future__ import annotations

import re
from typing import Any, Final

import structlog
from fastapi import APIRouter, HTTPException, Request, status

from pii_api.auth.deps import RotatedUser
from pii_api.upstream import AdminDisabled, UpstreamError

__all__ = ["router"]

logger: Final = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

# The same shape pii-service enforces. Checked here too, and only here, because
# this value is interpolated into an upstream URL: without it a caller could
# reach a different admin route by naming an entity "../reload".
ENTITY_TYPE = re.compile(r"^[A-Z][A-Z0-9_]{1,63}$")


def _check_entity_type(entity_type: str) -> None:
    if not ENTITY_TYPE.fullmatch(entity_type):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "entity type must be uppercase letters, digits and underscores, "
            "starting with a letter -- for example EMPLOYEE_BADGE",
        )


async def _forward(request: Request, call: Any) -> dict[str, Any]:
    """Run one upstream admin call, translating its failures."""
    try:
        return await call  # type: ignore[no-any-return]
    except AdminDisabled:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "entity administration is disabled: PII_API_PII_ADMIN_TOKEN is unset, so "
            "this backend holds no credential for pii-service's admin API.",
        ) from None
    except UpstreamError as exc:
        raise HTTPException(exc.status, exc.body) from exc


def _invalidate_categories(request: Request) -> None:
    """Drop the cached entity_type -> category map.

    The timeline and the event log colour rows from it and cache it for the
    process lifetime, since it normally changes only when someone edits
    entities.yaml. Adding a label through this screen is exactly the case that
    assumption misses: without this, a custom entity would render in the
    "other" colour until the next restart.
    """
    request.app.state.category_map = None


@router.get("/entities")
async def list_entities(request: Request, _user: RotatedUser) -> dict[str, Any]:
    """The effective policy: baseline plus overlay, with worked examples.

    Returned as pii-service sends it. The response shape is that service's
    ``PolicyView`` and the UI's ``PolicyView`` in types.ts; re-declaring it
    here would add a third copy for no gain.
    """
    return await _forward(request, request.app.state.pii.admin_policy())


@router.get("/replacement-strategies")
async def replacement_strategies(request: Request, _user: RotatedUser) -> dict[str, Any]:
    """What the strategy dropdown is populated from, including its warnings."""
    return await _forward(request, request.app.state.pii.replacement_strategies())


@router.put("/entities/{entity_type}")
async def upsert_entity(
    entity_type: str, payload: dict[str, Any], request: Request, user: RotatedUser
) -> dict[str, Any]:
    """Add a custom label, or change an existing entity's policy."""
    _check_entity_type(entity_type)
    view = await _forward(
        request,
        request.app.state.pii.upsert_entity(entity_type, payload, acting_user=user.email),
    )
    _invalidate_categories(request)
    # The values are policy, not PII: an entity name, a label prompt and a
    # replacement the operator typed. `by` is the point of the line.
    logger.info("admin.entity.upsert", entity_type=entity_type, by=user.email)
    return view


@router.delete("/entities/{entity_type}")
async def delete_entity(entity_type: str, request: Request, user: RotatedUser) -> dict[str, Any]:
    """Revert an entity to its baseline policy, or remove a custom label."""
    _check_entity_type(entity_type)
    view = await _forward(
        request, request.app.state.pii.delete_entity(entity_type, acting_user=user.email)
    )
    _invalidate_categories(request)
    logger.info("admin.entity.delete", entity_type=entity_type, by=user.email)
    return view
