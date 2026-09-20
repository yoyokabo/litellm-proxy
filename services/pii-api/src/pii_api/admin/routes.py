"""Admin routes: timeline, events, drill-down, the fingerprint pivot, flags.

Every route here requires an authenticated user who has rotated the bootstrap
password. There is one permission level in this build -- see db/models.py.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Final, Literal

import structlog
from fastapi import APIRouter, HTTPException, Query, Request, status

from pii_api.admin import queries
from pii_api.api.schemas import (
    EventDetail,
    EventPage,
    FingerprintSummary,
    FlagRequest,
    SummaryStats,
    TimelineBucket,
    TimelineResponse,
)
from pii_api.auth.deps import DbSession, RotatedUser

__all__ = ["router"]

logger: Final = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/admin", tags=["admin"])

MAX_PAGE = 500


async def _categories(request: Request) -> dict[str, str]:
    """entity_type -> category, from pii-service's policy, cached per process.

    Cached because it changes only when someone edits entities.yaml and
    restarts pii-service, and the timeline would otherwise fetch it on every
    brush drag.
    """
    cached = getattr(request.app.state, "category_map", None)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    try:
        policy = await request.app.state.pii.policy()
        mapping = {e["entity_type"]: e["category"] for e in policy["entities"]}
    except Exception:
        # A policy fetch failure must not take the admin view down; every
        # entity simply renders in the "other" colour until pii-service is
        # back. Losing the colour key is not losing the audit trail.
        logger.warning("admin.policy_unavailable")
        return {}

    request.app.state.category_map = mapping
    return mapping


def _filters(
    since: datetime | None,
    until: datetime | None,
    user_id: str | None,
    entity_type: list[str] | None,
    action: list[str] | None,
    value_fp: str | None,
    search: str | None,
) -> queries.EventFilters:
    return queries.EventFilters(
        since=since,
        until=until,
        user_id=user_id,
        entity_types=tuple(entity_type or ()),
        actions=tuple(action or ()),
        value_fp=value_fp,
        search=search,
    )


@router.get("/timeline", response_model=TimelineResponse)
async def timeline(
    request: Request,
    db: DbSession,
    _user: RotatedUser,
    since: datetime | None = None,
    until: datetime | None = None,
    bucket: Literal["minute", "hour", "day"] = "hour",
    user_id: str | None = None,
    entity_type: Annotated[list[str] | None, Query()] = None,
    action: Annotated[list[str] | None, Query()] = None,
    value_fp: str | None = None,
    search: str | None = None,
) -> TimelineResponse:
    """The stacked area, its blocked lane, and the header tiles.

    Blocked counts ride along per bucket rather than as a separate query: the
    UI draws them in their own lane (brief §10) so a handful of blocks beside
    thousands of masks stays visible instead of becoming a one-pixel band.
    """
    if since is None or until is None:
        default_since, default_until = await queries.recent_window(db, hours=24)
        since = since or default_since
        until = until or default_until

    filters = _filters(since, until, user_id, entity_type, action, value_fp, search)
    categories = await _categories(request)

    rows = await queries.timeline(db, filters, bucket)
    stats = await queries.summary_stats(db, filters)

    return TimelineResponse(
        since=since,
        until=until,
        bucket=bucket,
        points=[
            TimelineBucket(
                bucket=row["bucket"],
                entity_type=row["entity_type"],
                category=categories.get(row["entity_type"], "other"),
                total=row["total"],
                blocked=row["blocked"],
            )
            for row in rows
        ],
        summary=SummaryStats(**{key: int(value or 0) for key, value in stats.items()}),
        categories=categories,
    )


@router.get("/summary", response_model=SummaryStats)
async def summary(
    db: DbSession,
    _user: RotatedUser,
    since: datetime | None = None,
    until: datetime | None = None,
    user_id: str | None = None,
    entity_type: Annotated[list[str] | None, Query()] = None,
    action: Annotated[list[str] | None, Query()] = None,
    value_fp: str | None = None,
    search: str | None = None,
) -> SummaryStats:
    """Header tiles for the event log, without the bucketed aggregation.

    /timeline returns these too, but only alongside a GROUP BY over time
    buckets. The log view has no chart, so paying for that grouping on every
    filter change would be work done to be discarded.
    """
    filters = _filters(since, until, user_id, entity_type, action, value_fp, search)
    stats = await queries.summary_stats(db, filters)
    return SummaryStats(**{key: int(value or 0) for key, value in stats.items()})


@router.get("/events", response_model=EventPage)
async def events(
    request: Request,
    db: DbSession,
    _user: RotatedUser,
    since: datetime | None = None,
    until: datetime | None = None,
    user_id: str | None = None,
    entity_type: Annotated[list[str] | None, Query()] = None,
    action: Annotated[list[str] | None, Query()] = None,
    value_fp: str | None = None,
    search: str | None = None,
    limit: Annotated[int, Query(ge=1, le=MAX_PAGE)] = 200,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> EventPage:
    filters = _filters(since, until, user_id, entity_type, action, value_fp, search)
    categories = await _categories(request)

    rows = await queries.list_events(db, filters, limit=limit, offset=offset)
    flags = await queries.flags_for(db, [row["id"] for row in rows])

    return EventPage(
        total=await queries.count_events(db, filters),
        offset=offset,
        limit=limit,
        events=[
            EventDetail(
                **row,
                category=categories.get(row["entity_type"], "other"),
                flag=flags.get(row["id"]),
            )
            for row in rows
        ],
    )


@router.get("/events/{event_id}", response_model=EventDetail)
async def event(request: Request, db: DbSession, _user: RotatedUser, event_id: int) -> EventDetail:
    row = await queries.load_event(db, event_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such event")
    categories = await _categories(request)
    flags = await queries.flags_for(db, [event_id])
    return EventDetail(
        **row,
        category=categories.get(row["entity_type"], "other"),
        flag=flags.get(event_id),
    )


@router.get("/fingerprints/{value_fp}", response_model=FingerprintSummary)
async def fingerprint(db: DbSession, _user: RotatedUser, value_fp: str) -> FingerprintSummary:
    """Every other occurrence of one value. The core investigative move.

    Note what the caller gets and does not get: counts, users, timestamps --
    and no way to learn the value. The fingerprint is an HMAC under a pepper
    held outside this database, so this endpoint cannot disclose what it is
    correlating, only that occurrences are the same.
    """
    summary: dict[str, Any] = await queries.fingerprint_summary(db, value_fp)
    return FingerprintSummary(
        value_fp=value_fp,
        occurrences=int(summary.get("occurrences") or 0),
        distinct_users=int(summary.get("distinct_users") or 0),
        distinct_requests=int(summary.get("distinct_requests") or 0),
        first_seen=summary.get("first_seen"),
        last_seen=summary.get("last_seen"),
        per_user=summary.get("per_user") or [],
    )


@router.post("/events/{event_id}/flag", status_code=status.HTTP_204_NO_CONTENT)
async def flag(db: DbSession, user: RotatedUser, event_id: int, payload: FlagRequest) -> None:
    """Record a verdict on one span. Phase 5's eval loop reads these."""
    if await queries.load_event(db, event_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such event")
    await queries.upsert_flag(
        db, event_id=event_id, verdict=payload.verdict, user_id=user.id, note=payload.note
    )
    logger.info("admin.span_flagged", event_id=event_id, verdict=payload.verdict, by=user.id)
