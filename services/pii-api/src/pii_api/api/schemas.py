"""Request and response models for the web API.

Same rule as everywhere else in this system: nothing here that reaches the
audit trail or a log carries a matched value. The one deliberate exception is
``ChatSpan.text``, and it is worth being explicit about why it is allowed.

Brief §3: showing people the PII *they themselves just typed* is not a
disclosure. The chat needs it to underline the span in their own message. It
travels from pii-service to the browser that sent it and is never written to
a database, never logged, and never shown to anyone else -- the admin view,
which is about other people's messages, has no equivalent field.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "ChatRequest",
    "ChatSpan",
    "EventDetail",
    "EventPage",
    "FingerprintSummary",
    "FlagRequest",
    "LoginRequest",
    "PasswordChangeRequest",
    "TimelineResponse",
    "UserView",
]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------------------------------------------------------------------------
# auth
# ---------------------------------------------------------------------------


class LoginRequest(_Model):
    email: Annotated[str, Field(min_length=3, max_length=320)]
    password: Annotated[str, Field(min_length=1, max_length=1024)]


class PasswordChangeRequest(_Model):
    current_password: Annotated[str, Field(min_length=1, max_length=1024)]
    # Matches MIN_ADMIN_PASSWORD_LENGTH in settings.py: the rotated password
    # must not be weaker than the bootstrap one it replaces.
    new_password: Annotated[str, Field(min_length=12, max_length=1024)]


class UserView(_Model):
    id: int
    email: str
    must_change_password: bool


# ---------------------------------------------------------------------------
# admin
# ---------------------------------------------------------------------------


class TimelineBucket(_Model):
    bucket: datetime
    entity_type: str
    category: str
    total: int
    blocked: int


class SummaryStats(_Model):
    events: int
    users: int
    distinct_values: int
    blocked: int
    masked: int


class TimelineResponse(_Model):
    since: datetime
    until: datetime
    bucket: Literal["minute", "hour", "day"]
    points: list[TimelineBucket]
    summary: SummaryStats
    # entity_type -> category, so the client colours the stack without
    # holding its own copy of entities.yaml.
    categories: dict[str, str]


class EventDetail(_Model):
    """One audit row, exactly as stored. No value, by construction."""

    id: int
    ts: datetime
    request_id: str
    user_id: str | None = None
    team_id: str | None = None
    key_alias: str | None = None
    end_user_id: str | None = None
    entity_type: str
    category: str
    recognizer: str
    score: float | None = None
    action: str
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
    flag: str | None = None


class EventPage(_Model):
    total: int
    offset: int
    limit: int
    events: list[EventDetail]


class PerUserOccurrence(_Model):
    user_id: str | None = None
    occurrences: int
    last_seen: datetime | None = None


class FingerprintSummary(_Model):
    """The investigative pivot (brief §10).

    ``distinct_users`` against ``occurrences`` is the whole question: one
    person pasting a customer list and forty people each sending one record
    produce identical event counts and completely different rows here.
    """

    value_fp: str
    occurrences: int
    distinct_users: int
    distinct_requests: int
    first_seen: datetime | None = None
    last_seen: datetime | None = None
    per_user: list[PerUserOccurrence]


class FlagRequest(_Model):
    verdict: Literal["false_positive", "confirmed"]
    note: Annotated[str | None, Field(max_length=500)] = None


# ---------------------------------------------------------------------------
# chat
# ---------------------------------------------------------------------------


class ChatMessage(_Model):
    role: Literal["user", "assistant"]
    content: Annotated[str, Field(max_length=100_000)]


class ChatRequest(_Model):
    messages: Annotated[list[ChatMessage], Field(min_length=1, max_length=100)]
    model: str | None = None


class ChatSpan(_Model):
    """One detected span in the user's own message, for rendering.

    ``text`` is the matched value. See this module's docstring: it is the
    user's own input going back to the user's own browser, it is never
    persisted, and it is what lets the composer underline the right
    characters. ``start``/``end`` are offsets into the original string.
    """

    entity_type: str
    category: str
    action: str
    start: int
    end: int
    text: str
    placeholder: str
    score: float


class ChatAnalysis(_Model):
    """What was filtered, for the feedback chip above the message."""

    spans: list[ChatSpan]
    masked_text: str
    blocked: bool
    block_reason: dict[str, int] | None = None
    counts: dict[str, int]
    latency_ms: int


class AnalyzeOnlyResponse(_Model):
    """Pre-send detection. Debounced by the client and off by default."""

    analysis: ChatAnalysis


def event_to_detail(row: dict[str, Any], category: str, flag: str | None) -> EventDetail:
    return EventDetail(**{**row, "category": category, "flag": flag})
