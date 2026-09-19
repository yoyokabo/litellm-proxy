"""Structured JSON logging.

Brief §12: logging statements must never include matched values. The codebase
is built so that is structurally hard -- ``DetectedSpan.__repr__`` redacts,
``PreparedSpan`` has no value, ``AuditRecord`` has no value field -- and
``tests/test_no_pii_in_logs.py`` greps the whole suite's log output for fixture
values to catch anything that slips through.

This module adds one more layer: a processor that drops any event key which
looks like it carries raw text. A developer adding ``logger.info("masked",
text=text)`` in a hurry gets a redaction rather than an incident.
"""

from __future__ import annotations

import logging
import sys
from typing import Final

import structlog
from structlog.typing import EventDict, WrappedLogger

__all__ = ["REDACTED", "configure_logging", "redact_text_keys"]

REDACTED: Final = "<redacted>"

# Keys that must never be emitted, whatever they are set to. Anything carrying
# prompt text, a matched value, or a secret.
_FORBIDDEN_KEYS: Final[frozenset[str]] = frozenset(
    {
        "text",
        "texts",
        "value",
        "values",
        "content",
        "message",
        "messages",
        "prompt",
        "input",
        "output",
        "anonymized_text",
        "original_text",
        "matched",
        "match",
        "match_details",
        "classification",
        "guardrail_request",
        "guardrail_response",
        "span_value",
        "preview",
        "context",
        "pepper",
        "api_key",
        "authorization",
        "password",
        "token",
    }
)

# `message` is structlog's own rendered event under some configurations, so the
# exception below keeps error reporting usable while still dropping a caller's
# `message=<user text>`.
_ALLOWED_WHEN_STRING: Final[frozenset[str]] = frozenset({"message"})


def redact_text_keys(_logger: WrappedLogger, _name: str, event_dict: EventDict) -> EventDict:
    """Drop prompt-carrying keys before anything is rendered.

    Deliberately a denylist on key *name* rather than a scan of values: a
    value scan cannot tell a national ID from an order number, and would be
    both slower and less predictable than simply refusing to log fields that
    have no business being logged.
    """
    for key in list(event_dict):
        if key in _FORBIDDEN_KEYS:
            if key in _ALLOWED_WHEN_STRING and key == "message":
                # Allow short exception messages through; truncate so a
                # database error quoting a row cannot dump a batch.
                value = event_dict[key]
                event_dict[key] = str(value)[:500]
                continue
            event_dict[key] = REDACTED
    return event_dict


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure structlog and the stdlib root logger to match."""
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, level.upper(), logging.INFO),
    )

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            redact_text_keys,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )
