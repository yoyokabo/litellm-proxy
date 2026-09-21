"""Read-only queries over the audit trail.

Every function here reads ``pii_events``, which belongs to pii-service. None of
them writes it. The only table this module writes is ``span_flags``, which
annotates an audit row without copying anything out of it.

Two shapes the admin view needs that the schema does not hand over directly:

* **The timeline** stacks by entity *category*, and the table stores
  ``entity_type``. The mapping comes from pii-service's ``/policy`` endpoint,
  so there is exactly one copy of entities.yaml in the system.
* **The pivot** -- every other occurrence of one ``value_fp`` -- is the core
  investigative move (brief §10). It is a plain indexed lookup, which is the
  whole argument for storing a keyed fingerprint rather than a counter.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final, Literal

from sqlalchemy import Row, and_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

__all__ = [
    "BUCKETS",
    "EventFilters",
    "count_events",
    "fingerprint_summary",
    "list_events",
    "load_event",
    "timeline",
    "upsert_flag",
]

# Bucket width -> the SQL interval used to truncate the timestamp.
BUCKETS: Final[dict[str, str]] = {
    "minute": "minute",
    "hour": "hour",
    "day": "day",
}

# Reading pii_events through raw column names rather than an ORM model.
# Importing pii-service's model would make this service's image depend on
# that package, and the point of the split is that it does not.
_EVENT_COLUMNS: Final = (
    "id, ts, request_id, user_id, team_id, key_alias, end_user_id, entity_type, "
    "recognizer, score, action, span_start, span_end, value_len, message_index, "
    "message_role, field, value_fp, preview, model, lang, latency_ms"
)


@dataclass(frozen=True, slots=True)
class EventFilters:
    """Everything the timeline brush and the table's filter row can set."""

    since: datetime | None = None
    until: datetime | None = None
    user_id: str | None = None
    entity_types: tuple[str, ...] = ()
    actions: tuple[str, ...] = ()
    value_fp: str | None = None
    # Free-text search over identity columns only. There is no text column
    # holding a prompt to search, which is the design working as intended.
    search: str | None = None

    def where(self) -> tuple[str, dict[str, Any]]:
        clauses: list[str] = []
        params: dict[str, Any] = {}

        if self.since is not None:
            clauses.append("ts >= :since")
            params["since"] = self.since
        if self.until is not None:
            clauses.append("ts <= :until")
            params["until"] = self.until
        if self.user_id:
            clauses.append("user_id = :user_id")
            params["user_id"] = self.user_id
        if self.value_fp:
            clauses.append("value_fp = :value_fp")
            params["value_fp"] = self.value_fp
        # IN lists are expanded to one named parameter per value rather than
        # interpolated, so an entity type arriving from a query string is a
        # bound value and can never become SQL.
        for prefix, values in (("entity", self.entity_types), ("action", self.actions)):
            if not values:
                continue
            keys = [f"{prefix}_{index}" for index in range(len(values))]
            column = "entity_type" if prefix == "entity" else "action"
            placeholders = ", ".join(f":{key}" for key in keys)
            clauses.append(f"{column} IN ({placeholders})")
            params.update(dict(zip(keys, values, strict=True)))
        if self.search:
            clauses.append(
                "(user_id ILIKE :search OR key_alias ILIKE :search "
                "OR end_user_id ILIKE :search OR request_id ILIKE :search)"
            )
            params["search"] = f"%{self.search}%"

        return (" AND ".join(clauses) if clauses else "TRUE"), params


def _row_to_dict(row: Row[Any]) -> dict[str, Any]:
    return dict(row._mapping)


async def timeline(
    session: AsyncSession,
    filters: EventFilters,
    bucket: Literal["minute", "hour", "day"] = "hour",
) -> list[dict[str, Any]]:
    """Events per time bucket per entity type, plus a blocked count.

    Returns entity *types*; the caller folds them into categories using the
    policy. Blocked events are counted separately rather than being folded
    into the stack, because brief §10 puts them in their own thin lane -- in
    a stacked area a handful of blocks next to thousands of masks is a line
    one pixel high, which is to say invisible exactly when it matters.
    """
    where, params = filters.where()
    unit = BUCKETS[bucket]

    statement = text(
        f"""
        SELECT date_trunc(:unit, ts) AS bucket,
               entity_type,
               COUNT(*) AS total,
               COUNT(*) FILTER (WHERE action = 'BLOCK') AS blocked
        FROM pii_events
        WHERE {where}
        GROUP BY 1, 2
        ORDER BY 1
        """  # noqa: S608 -- `where` is built from named params only, never input
    )
    result = await session.execute(statement, {**params, "unit": unit})
    return [_row_to_dict(row) for row in result]


async def list_events(
    session: AsyncSession, filters: EventFilters, *, limit: int = 200, offset: int = 0
) -> list[dict[str, Any]]:
    """One page of the event table, newest first."""
    where, params = filters.where()
    statement = text(
        f"SELECT {_EVENT_COLUMNS} FROM pii_events WHERE {where} "  # noqa: S608
        "ORDER BY ts DESC, id DESC LIMIT :limit OFFSET :offset"
    )
    result = await session.execute(statement, {**params, "limit": limit, "offset": offset})
    return [_row_to_dict(row) for row in result]


async def count_events(session: AsyncSession, filters: EventFilters) -> int:
    where, params = filters.where()
    statement = text(f"SELECT COUNT(*) FROM pii_events WHERE {where}")  # noqa: S608
    return int(await session.scalar(statement, params) or 0)


async def load_event(session: AsyncSession, event_id: int) -> dict[str, Any] | None:
    statement = text(f"SELECT {_EVENT_COLUMNS} FROM pii_events WHERE id = :id")  # noqa: S608
    row = (await session.execute(statement, {"id": event_id})).first()
    return _row_to_dict(row) if row is not None else None


async def fingerprint_summary(session: AsyncSession, value_fp: str) -> dict[str, Any]:
    """The pivot's headline: how one value is distributed across people.

    This is the question the pepper exists to answer (brief §6). "One user
    pasted a customer list" and "forty users each sent one record" produce the
    same event count and completely different answers here -- and neither
    requires storing a digit.
    """
    statement = text(
        """
        SELECT COUNT(*) AS occurrences,
               COUNT(DISTINCT user_id) AS distinct_users,
               COUNT(DISTINCT request_id) AS distinct_requests,
               MIN(ts) AS first_seen,
               MAX(ts) AS last_seen
        FROM pii_events
        WHERE value_fp = :fp
        """
    )
    row = (await session.execute(statement, {"fp": value_fp})).first()
    summary = _row_to_dict(row) if row is not None else {}

    per_user = await session.execute(
        text(
            """
            SELECT user_id, COUNT(*) AS occurrences, MAX(ts) AS last_seen
            FROM pii_events WHERE value_fp = :fp
            GROUP BY user_id ORDER BY occurrences DESC LIMIT 50
            """
        ),
        {"fp": value_fp},
    )
    summary["per_user"] = [_row_to_dict(row) for row in per_user]
    return summary


async def flags_for(session: AsyncSession, event_ids: list[int]) -> dict[int, str]:
    """Existing verdicts, so the table can show what has already been reviewed."""
    if not event_ids:
        return {}
    from pii_api.db.models import SpanFlag

    rows = await session.execute(
        select(SpanFlag.event_id, SpanFlag.verdict).where(SpanFlag.event_id.in_(event_ids))
    )
    return {event_id: verdict for event_id, verdict in rows}


async def upsert_flag(
    session: AsyncSession, *, event_id: int, verdict: str, user_id: int, note: str | None
) -> None:
    """Record or replace this reviewer's verdict on one span.

    Feeds phase 5's eval loop. ``confirmed`` is stored as well as
    ``false_positive`` on purpose -- an eval set built only from corrections
    learns what the detector gets wrong and nothing about what it gets right.
    """
    from pii_api.db.models import SpanFlag

    existing = await session.scalar(
        select(SpanFlag).where(and_(SpanFlag.event_id == event_id, SpanFlag.flagged_by == user_id))
    )
    if existing is None:
        session.add(SpanFlag(event_id=event_id, verdict=verdict, flagged_by=user_id, note=note))
    else:
        existing.verdict = verdict
        existing.note = note
    await session.commit()


async def summary_stats(session: AsyncSession, filters: EventFilters) -> dict[str, Any]:
    """The header tiles: totals, people, distinct values, block count."""
    where, params = filters.where()
    statement = text(
        f"""
        SELECT COUNT(*) AS events,
               COUNT(DISTINCT user_id) AS users,
               COUNT(DISTINCT value_fp) AS distinct_values,
               COUNT(*) FILTER (WHERE action = 'BLOCK') AS blocked,
               COUNT(*) FILTER (WHERE action = 'MASK') AS masked
        FROM pii_events WHERE {where}
        """  # noqa: S608
    )
    row = (await session.execute(statement, params)).first()
    return _row_to_dict(row) if row is not None else {}


async def recent_window(session: AsyncSession, hours: int = 24) -> tuple[datetime, datetime]:
    """Default time range: the last N hours, or the last N hours of data.

    An empty-looking dashboard on a system with a week-old demo seed is a
    support ticket, so if nothing landed recently the window falls back to
    the most recent event rather than to now.
    """
    latest = await session.scalar(text("SELECT MAX(ts) FROM pii_events"))
    end = latest if isinstance(latest, datetime) else datetime.now(UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return end - timedelta(hours=hours), end
