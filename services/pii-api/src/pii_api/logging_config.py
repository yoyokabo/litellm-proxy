"""Structured JSON logging.

Same rule as pii-service, for the same reason: **no logging statement in this
service may include message text or a matched value.** This process handles
prompts in the clear -- it is the one that masks them -- so it is the easiest
place in the system to accidentally build a plaintext prompt log.
"""

from __future__ import annotations

import logging
from typing import Final

import structlog

__all__ = ["configure_logging"]


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(format="%(message)s", level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


logger: Final = structlog.get_logger("pii_api")
