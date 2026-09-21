"""Admin API: custom tier-3 labels and replacement policy.

The endpoints an admin menu drives. The menu itself is phase 2 (brief §10) and
is not built here; this is the surface it will call, and it is usable directly
with curl in the meantime.

AUTHENTICATION IS A PLACEHOLDER, DELIBERATELY. The brief specifies phase-2 auth
in detail -- a default admin bootstrapped from ADMIN_EMAIL /
ADMIN_INITIAL_PASSWORD, flagged must_change_password, with roles ``admin`` and
``auditor``. Building that now would mean guessing at a design the brief has
already committed to, so these routes take a single bearer token from
``PII_ADMIN_TOKEN`` instead, compared in constant time. When phase 2 lands,
replace ``require_admin`` and nothing else changes.

Consequences of that: the token is all-or-nothing (there is no read-only
auditor yet), and every change is attributed to whoever holds it. The
``updated_by`` column exists so that attribution becomes real the moment
per-user auth does -- callers may set it now via ``X-Admin-User``.
"""

from __future__ import annotations

import hmac
from typing import Annotated, Final

import structlog
from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from pii_service.db.custom_entities import CustomEntity
from pii_service.policy.overlay import EntityOverlay
from pii_service.policy.replacement import ReplacementRule, ReplacementStrategy

__all__ = ["router"]

logger: Final = structlog.get_logger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------


async def require_admin(request: Request, authorization: str = Header(default="")) -> str:
    """Bearer-token gate. Replaced wholesale by phase-2 role auth."""
    expected: str = getattr(request.app.state, "admin_token", "") or ""
    if not expected:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "the admin API is disabled because PII_ADMIN_TOKEN is unset",
        )

    scheme, _, presented = authorization.partition(" ")
    if scheme.lower() != "bearer" or not hmac.compare_digest(presented, expected):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid admin credentials")
    return presented


AdminAuth = Annotated[str, Depends(require_admin)]


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ReplacementSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    strategy: ReplacementStrategy = ReplacementStrategy.PLACEHOLDER
    value: str | None = None
    pool: list[str] = Field(default_factory=list)

    def to_rule(self) -> ReplacementRule:
        return ReplacementRule(strategy=self.strategy, value=self.value, pool=tuple(self.pool))


class EntityUpsert(BaseModel):
    """Add a label, or override policy for an existing entity."""

    model_config = ConfigDict(extra="forbid")

    entity_type: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,63}$")

    # Required when introducing a type the baseline does not detect. This is
    # the label GLiNER2 is conditioned on -- a natural-language phrase, not an
    # identifier: "employee badge number", not "EMPLOYEE_BADGE".
    gliner_prompt: str | None = Field(default=None, min_length=2, max_length=200)

    category: str | None = None
    action: str | None = None
    score_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    placeholder: str | None = Field(default=None, min_length=1, max_length=64)
    replacement: ReplacementSpec | None = None

    enabled: bool = True
    note: str | None = Field(default=None, max_length=500)

    # The two policy enums disagree on case -- EntityAction is MASK/BLOCK/ALLOW
    # and EntityCategory is id/person/contact/... -- because each matches how
    # its values already appear in entities.yaml and in the audit rows. That is
    # fine inside the service and a trap at an HTTP boundary, where a caller
    # round-tripping a value out of GET /policy would send "MASK" and one typing
    # it by hand would send "mask". Normalise here rather than making every
    # caller remember which is which.
    @field_validator("action", mode="before")
    @classmethod
    def _upper_action(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("category", mode="before")
    @classmethod
    def _lower_category(cls, value: object) -> object:
        return value.lower() if isinstance(value, str) else value


class EntityView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    source: str  # "baseline" | "overlay"
    category: str
    action: str
    score_threshold: float
    placeholder: str
    tier: int | None = None
    gliner_prompt: str | None = None
    replacement_strategy: str
    replacement_example: str
    replacement_is_realistic: bool
    enabled: bool = True
    note: str | None = None
    updated_by: str | None = None


class PolicyView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entities: list[EntityView]
    tier3_labels: dict[str, str]
    tier3_enabled: bool
    realistic_replacement_entities: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


def _sessions(request: Request) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(request.app.state.sink._engine, expire_on_commit=False)


def _view(request: Request) -> PolicyView:
    store = request.app.state.policy_store
    policy = store.current
    overlay = {o.entity_type: o for o in policy.overlay}
    baseline_names = set(store.baseline.entities)

    entities: list[EntityView] = []
    for name, entry in sorted(policy.entities.items()):
        rule = policy.replacement_rule_for(name)
        row = overlay.get(name)
        entities.append(
            EntityView(
                entity_type=name,
                source="baseline" if name in baseline_names and row is None else "overlay",
                category=str(entry.category),
                action=str(entry.action),
                score_threshold=entry.score_threshold,
                placeholder=entry.placeholder,
                tier=entry.tier,
                gliner_prompt=entry.gliner_prompt,
                replacement_strategy=str(rule.strategy),
                # A worked example, because "surrogate" tells an operator
                # nothing about what their prompts will actually look like.
                replacement_example=rule.render(
                    placeholder=entry.placeholder, value_fp="9f75871d3d222bcb"
                ),
                replacement_is_realistic=rule.is_realistic,
                enabled=True,
                note=row.note if row else None,
                updated_by=row.updated_by if row else None,
            )
        )

    warnings: list[str] = []
    realistic = list(policy.realistic_replacement_entities)
    if realistic:
        warnings.append(
            f"{', '.join(realistic)} are replaced with realistic-looking text, so a "
            "masked prompt no longer looks masked to a human reader. A value matching "
            "one of those replacements is also left alone, to keep a second masking "
            "pass a no-op."
        )
    if policy.gliner_prompts and not request.app.state.tiers.get("tier3_gliner"):
        warnings.append(
            f"{len(policy.gliner_prompts)} tier-3 labels are configured but tier 3 is "
            "disabled (PII_ENABLE_TIER3_GLINER=false), so none of them detect anything."
        )

    return PolicyView(
        entities=entities,
        tier3_labels=dict(policy.gliner_prompts),
        tier3_enabled=bool(request.app.state.tiers.get("tier3_gliner")),
        realistic_replacement_entities=realistic,
        warnings=warnings,
    )


@router.get("/policy", response_model=PolicyView)
async def get_policy(request: Request, _: AdminAuth) -> PolicyView:
    """The effective policy: baseline plus overlay, with worked examples."""
    return _view(request)


@router.get("/replacement-strategies")
async def list_strategies(_: AdminAuth) -> dict[str, object]:
    """What an admin menu populates its strategy dropdown from."""
    return {
        "strategies": [
            {
                "name": str(ReplacementStrategy.PLACEHOLDER),
                "example": "<PERSON>",
                "realistic": False,
                "description": "Unambiguous. Nobody mistakes it for data.",
            },
            {
                "name": str(ReplacementStrategy.CONSTANT),
                "example": "John Doe",
                "realistic": True,
                "description": (
                    "One fixed string for every match. Collapses distinct values: "
                    "'Ahmed emailed Sara' becomes 'John Doe emailed John Doe'. Use "
                    "for entities that appear at most once per message."
                ),
            },
            {
                "name": str(ReplacementStrategy.SURROGATE),
                "example": "John Doe / Jane Roe / Sam Poe",
                "realistic": True,
                "description": (
                    "A stable pick from a pool, keyed by the value's fingerprint. "
                    "The same person is the same fake name in every turn, and two "
                    "people stay two people. Prefer this over 'constant'."
                ),
            },
            {
                "name": str(ReplacementStrategy.REDACT),
                "example": "",
                "realistic": False,
                "description": "Remove the span entirely.",
            },
            {
                "name": str(ReplacementStrategy.LABELLED_FINGERPRINT),
                "example": "<PERSON:9f75871d>",
                "realistic": False,
                "description": (
                    "Distinct values stay distinct and an auditor can correlate "
                    "against pii_events, while the text stays obviously masked."
                ),
            },
        ]
    }


@router.put("/entities/{entity_type}", response_model=PolicyView, status_code=status.HTTP_200_OK)
async def upsert_entity(
    entity_type: str,
    payload: EntityUpsert,
    request: Request,
    _: AdminAuth,
    x_admin_user: str | None = Header(default=None),
) -> PolicyView:
    """Add a custom label, or override an existing entity's policy."""
    if payload.entity_type != entity_type:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"path says {entity_type} but body says {payload.entity_type}",
        )

    store = request.app.state.policy_store
    is_new = entity_type not in store.baseline.entities
    if is_new and not payload.gliner_prompt:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{entity_type} is not in the baseline policy, so it needs a gliner_prompt "
            "-- otherwise nothing would ever detect it.",
        )

    # Validate before writing: a row that cannot become a policy is a row that
    # breaks the next reload for everyone.
    try:
        EntityOverlay(
            entity_type=entity_type,
            gliner_prompt=payload.gliner_prompt,
            category=payload.category,  # type: ignore[arg-type]
            action=payload.action,  # type: ignore[arg-type]
            score_threshold=payload.score_threshold,
            placeholder=payload.placeholder,
            replacement=payload.replacement.to_rule() if payload.replacement else None,
            enabled=payload.enabled,
            note=payload.note,
        )
    except Exception as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    replacement = payload.replacement
    async with _sessions(request)() as session, session.begin():
        existing = (
            await session.execute(
                select(CustomEntity).where(CustomEntity.entity_type == entity_type)
            )
        ).scalar_one_or_none()

        if existing is None:
            existing = CustomEntity(entity_type=entity_type)
            session.add(existing)

        existing.gliner_prompt = payload.gliner_prompt
        existing.category = payload.category
        existing.action = payload.action
        existing.score_threshold = payload.score_threshold
        existing.placeholder = payload.placeholder
        existing.replacement_strategy = str(replacement.strategy) if replacement else None
        existing.replacement_value = replacement.value if replacement else None
        existing.replacement_pool = list(replacement.pool) if replacement else None
        existing.enabled = payload.enabled
        existing.note = payload.note
        existing.updated_by = x_admin_user

    await store.refresh()
    request.app.state.service.invalidate_cache()

    logger.info(
        "admin.entity.upsert",
        entity_type=entity_type,
        created=is_new,
        enabled=payload.enabled,
        strategy=str(replacement.strategy) if replacement else None,
        updated_by=x_admin_user,
    )
    return _view(request)


@router.delete("/entities/{entity_type}", response_model=PolicyView)
async def delete_entity(entity_type: str, request: Request, _: AdminAuth) -> PolicyView:
    """Remove an overlay row, reverting the entity to its baseline policy.

    A baseline entity reverts to entities.yaml; a purely custom one disappears.
    To stop detecting a baseline entity without deleting its history, PUT it
    with ``enabled: false`` instead.
    """
    async with _sessions(request)() as session, session.begin():
        row = (
            await session.execute(
                select(CustomEntity).where(CustomEntity.entity_type == entity_type)
            )
        ).scalar_one_or_none()
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"no overlay row for {entity_type}")
        await session.delete(row)

    await request.app.state.policy_store.refresh()
    request.app.state.service.invalidate_cache()
    logger.info("admin.entity.delete", entity_type=entity_type)
    return _view(request)


@router.post("/reload", response_model=PolicyView)
async def reload_policy(request: Request, _: AdminAuth) -> PolicyView:
    """Re-read the overlay from the database.

    For the case where another replica made the change: each process holds its
    own effective policy, so a change made on one is picked up by the others at
    their next restart or reload.
    """
    await request.app.state.policy_store.refresh()
    request.app.state.service.invalidate_cache()
    return _view(request)
