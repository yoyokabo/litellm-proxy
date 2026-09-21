"""The value-free span.

``DetectedSpan`` (in ``detect.router``) carries the matched text, because
fingerprinting and preview rendering need it. ``PreparedSpan`` is what exists
after those two computations have happened: everything downstream -- the audit
record, the API response, the detection cache -- consumes this instead.

The point of the split is that it makes the safe thing structural. The value
exists inside one function call and is gone by the time anything is stored,
serialized or cached, so no later code path can leak it by forgetting to.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

from pii_service.audit.fingerprint import fingerprint
from pii_service.detect.router import DetectedSpan
from pii_service.policy.loader import PolicyBundle
from pii_service.policy.models import EntityAction

__all__ = ["PreparedSpan"]


@dataclass(frozen=True, slots=True)
class PreparedSpan:
    """A finding with its fingerprint and preview resolved, and no value."""

    text_index: int
    entity_type: str
    category: str
    recognizer: str
    score: float
    action: str
    start: int
    end: int
    value_len: int
    lang: str
    value_fp: str | None = None
    preview: str | None = None
    context_term: str | None = None
    replacement: str = ""
    """The exact text this span becomes when masked.

    Resolved here, in the one function that still holds the value, because a
    ``surrogate`` rule keys on the fingerprint computed two lines above. Every
    later stage -- cache, masking, API -- carries the answer rather than the
    inputs, so none of them needs the value or the pepper.
    """

    @classmethod
    def from_detected(
        cls,
        span: DetectedSpan,
        *,
        text_index: int,
        policy: PolicyBundle,
        pepper: bytes,
    ) -> Self:
        """Consume a detected span's value, returning a span without one."""
        return cls(
            text_index=text_index,
            entity_type=span.entity_type,
            category=str(span.category),
            recognizer=span.recognizer,
            score=span.score,
            action=str(span.action),
            start=span.start,
            end=span.end,
            value_len=span.value_len,
            lang=span.lang,
            value_fp=(value_fp := fingerprint(span.entity_type, span.value, pepper)),
            preview=policy.render_preview(span.entity_type, span.value),
            context_term=span.context_term,
            replacement=policy.render_replacement(span.entity_type, value_fp),
        )

    @property
    def is_masked(self) -> bool:
        return self.action == EntityAction.MASK

    @property
    def is_blocking(self) -> bool:
        return self.action == EntityAction.BLOCK
