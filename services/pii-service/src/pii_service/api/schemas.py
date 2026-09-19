"""Request and response schemas for the detection API.

The endpoint is deliberately generic -- it takes a list of strings and returns a
list of strings. Both callers use it: the LiteLLM guardrail flattens message
content and tool-call arguments into that list, and the chat backend sends
message text. Wire-format knowledge (OpenAI vs Anthropic tool-call shapes)
belongs in the guardrail adapter, not here.

The one rule that shapes every model below: **a Finding never carries the
matched value.** ``preview`` is governed by ``preview_policy.yaml`` and is
usually NULL. ``context`` is the single exception, is opt-in per request, and
is never persisted -- see ``AnalyzeRequest.include_context``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "AnalyzeRequest",
    "AnalyzeResponse",
    "Finding",
    "HealthResponse",
    "RequestIdentity",
]


class RequestIdentity(BaseModel):
    """Who is making the request, as LiteLLM reports it.

    Every field is optional: a call from a script with a raw key has no team,
    and a call from the chat backend has an end user but no key alias. Missing
    identity must degrade the audit row, never reject the request.
    """

    model_config = ConfigDict(extra="ignore")

    user_id: str | None = None  # user_api_key_user_id
    team_id: str | None = None  # user_api_key_team_id
    key_alias: str | None = None
    key_hash: str | None = None
    end_user_id: str | None = None  # request body "user"


class AnalyzeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    texts: Annotated[list[str], Field(max_length=512)]
    request_id: str = Field(min_length=1, max_length=200)
    identity: RequestIdentity = RequestIdentity()
    model: str | None = None

    # Where each text came from, parallel to `texts`. Used only to fill the
    # audit row's message_index / message_role / field columns, so a drill-down
    # can say "message 3, tool call arguments" without storing the text.
    message_indices: list[int] | None = None
    message_roles: list[str] | None = None
    fields: list[str] | None = None

    # Override script detection. Omit to let the router decide.
    language: Literal["en", "ar"] | None = None

    # Return a short snippet of surrounding text with each finding.
    #
    # Off by default and never written to the database. Showing someone the
    # text they just typed is not a disclosure (brief §3), so the chat backend
    # turns this on to render underlined spans; the proxy guardrail, whose
    # output reaches logs, leaves it off.
    include_context: bool = False
    context_chars: Annotated[int, Field(ge=0, le=200)] = 40

    # Skip the detection cache. Used by the benchmark, which would otherwise
    # measure dictionary lookups.
    bypass_cache: bool = False


class Finding(BaseModel):
    """One detected span. Carries no matched value."""

    model_config = ConfigDict(extra="forbid")

    text_index: int
    entity_type: str
    category: str
    recognizer: str
    score: float
    action: str

    # Offsets into the *original* text at `text_index`, never the normalized
    # form. Character offsets over Arabic render in visually confusing order
    # (brief §10), so a UI must show these numerically and isolate the context
    # with dir="auto" / unicode-bidi: isolate rather than concatenating them.
    start: int
    end: int
    value_len: int

    value_fp: str | None = None
    preview: str | None = None
    context_term: str | None = None
    lang: str | None = None

    # Present only when include_context was set. Never persisted.
    context: str | None = None
    context_offset: int | None = None


class AnalyzeResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str
    # Same length and order as the request's `texts`, with MASK spans replaced.
    texts: list[str]
    findings: list[Finding]
    entity_counts: dict[str, int]
    blocked: bool = False
    # Machine-readable and value-free: entity types and counts only.
    block_reason: dict[str, int] | None = None
    lang: str | None = None
    latency_ms: int = 0
    cached: bool = False


class HealthResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: Literal["ok", "degraded"]
    version: str
    database_reachable: bool
    audit: dict[str, object]
    tiers: dict[str, bool]
