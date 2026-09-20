"""HTTP routes.

Only three endpoints in phase 1. The break-glass reveal path is defined in the
data model and deliberately absent here (brief §6).
"""

from __future__ import annotations

from typing import Annotated, Final

import structlog
from fastapi import APIRouter, Depends, Request

from pii_service.api.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    EntityPolicyView,
    HealthResponse,
    PolicyResponse,
)
from pii_service.service import PiiService

__all__ = ["router"]

logger: Final = structlog.get_logger(__name__)

router = APIRouter()


def get_service(request: Request) -> PiiService:
    return request.app.state.service  # type: ignore[no-any-return]


ServiceDep = Annotated[PiiService, Depends(get_service)]


@router.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: AnalyzeRequest, service: ServiceDep) -> AnalyzeResponse:
    """Detect, mask and record.

    Runs synchronously: the caller needs the masked text back before it can
    forward the request, so there is nothing to defer. The *audit write* is
    what gets deferred, inside the service.
    """
    response = service.analyze(payload)

    # Counts and status only. Never the texts, never the findings' offsets --
    # this line goes to stdout and from there to whatever log shipper the
    # deployment runs.
    logger.info(
        "analyze",
        request_id=payload.request_id,
        user_id=payload.identity.user_id,
        entity_counts=response.entity_counts,
        blocked=response.blocked,
        latency_ms=response.latency_ms,
        cached=response.cached,
        text_count=len(payload.texts),
    )
    return response


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, service: ServiceDep) -> HealthResponse:
    """Liveness and audit-pipeline state.

    Reports ``degraded`` -- not unhealthy -- when the database is unreachable.
    The service is still masking correctly and spilling to the WAL, and an
    orchestrator that restarts it for that would turn a logging outage into a
    gateway outage, which is exactly the failure mode brief §8 forbids.
    """
    sink = request.app.state.sink
    reachable = await sink.database_reachable()

    return HealthResponse(
        status="ok" if reachable else "degraded",
        version=request.app.version,
        database_reachable=reachable,
        audit={**sink.health(), "cache": service.cache_stats()},
        tiers=request.app.state.tiers,
    )


@router.get("/policy", response_model=PolicyResponse)
async def policy(request: Request) -> PolicyResponse:
    """The entity policy, for callers that must render or group by it.

    Read-only and derived entirely from the YAML this service already
    validated at startup. It exists so the admin UI can map entity types to
    categories without a second copy of entities.yaml drifting out of sync
    with this one.
    """
    bundle = request.app.state.policy
    return PolicyResponse(
        version=bundle.entities_file.version,
        entities=[
            EntityPolicyView(
                entity_type=name,
                category=str(entity.category),
                action=str(entity.action),
                tier=entity.tier,
                score_threshold=entity.score_threshold,
                placeholder=entity.placeholder,
            )
            for name, entity in sorted(bundle.entities.items())
        ],
    )


@router.get("/livez")
async def livez() -> dict[str, str]:
    """Bare liveness: is the process up. Never touches the database."""
    return {"status": "ok"}
