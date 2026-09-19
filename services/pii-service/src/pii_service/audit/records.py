"""The audit record and its construction.

A record is built from a ``PreparedSpan``, which by construction has no matched
value -- the fingerprint and preview were computed in
``PreparedSpan.from_detected`` and the value discarded there. So this module
never sees a value and cannot write one out, whatever a future serializer,
logger or retry handler does.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Self

from pii_service.spans import PreparedSpan

__all__ = ["AuditRecord", "RequestContext"]


@dataclass(frozen=True, slots=True)
class RequestContext:
    """Identity and provenance for one request, as LiteLLM reports it."""

    request_id: str
    user_id: str | None = None
    team_id: str | None = None
    key_alias: str | None = None
    key_hash: str | None = None
    end_user_id: str | None = None
    model: str | None = None
    message_index: int | None = None
    message_role: str | None = None
    field_name: str | None = None


@dataclass(frozen=True, slots=True)
class AuditRecord:
    """One row of ``pii_events``. Deliberately has no ``value`` field."""

    request_id: str
    entity_type: str
    recognizer: str
    action: str
    ts: datetime = field(default_factory=lambda: datetime.now(UTC))

    user_id: str | None = None
    team_id: str | None = None
    key_alias: str | None = None
    key_hash: str | None = None
    end_user_id: str | None = None

    score: float | None = None
    span_start: int | None = None
    span_end: int | None = None
    value_len: int | None = None
    message_index: int | None = None
    message_role: str | None = None
    field: str | None = None

    value_fp: str | None = None
    preview: str | None = None

    model: str | None = None
    lang: str | None = None
    latency_ms: int | None = None

    @classmethod
    def from_prepared(
        cls,
        span: PreparedSpan,
        context: RequestContext,
        *,
        latency_ms: int | None = None,
        ts: datetime | None = None,
    ) -> Self:
        """Build a row from a value-free span plus this request's identity."""
        return cls(
            request_id=context.request_id,
            entity_type=span.entity_type,
            recognizer=span.recognizer,
            action=span.action,
            ts=ts or datetime.now(UTC),
            user_id=context.user_id,
            team_id=context.team_id,
            key_alias=context.key_alias,
            key_hash=context.key_hash,
            end_user_id=context.end_user_id,
            score=span.score,
            span_start=span.start,
            span_end=span.end,
            value_len=span.value_len,
            message_index=context.message_index,
            message_role=context.message_role,
            field=context.field_name,
            value_fp=span.value_fp,
            preview=span.preview,
            model=context.model,
            lang=span.lang,
            latency_ms=latency_ms,
        )

    def to_row(self) -> dict[str, Any]:
        """Column mapping for a bulk insert."""
        return asdict(self)

    def to_json(self) -> str:
        """Serialize for the write-ahead log."""
        row = self.to_row()
        row["ts"] = self.ts.isoformat()
        return json.dumps(row, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str) -> Self:
        payload = json.loads(line)
        payload["ts"] = datetime.fromisoformat(payload["ts"])
        return cls(**payload)
