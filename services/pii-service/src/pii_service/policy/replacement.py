"""What a masked span is replaced *with*.

Until now every masked span became its entity's placeholder, ``<PERSON>``.
That is unambiguous -- nobody reading the prompt mistakes it for data -- but it
reads poorly to a model, and some deployments would rather the prompt stay
natural. So replacement is now a per-entity policy with several strategies.

Two consequences of realistic replacement that the code has to handle, not
just document:

**It breaks masking idempotency.** The architecture (brief §2) masks twice: the
chat backend analyses and masks, then the proxy guardrail masks the result
again, relying on "re-running on masked text finds nothing". ``<PERSON>`` is
not a name, so the second pass finds nothing. "John Doe" *is* a name, so the
second pass detects it and replaces it again -- with a different surrogate,
corrupting the text a little more on every hop.

The fix is to make each rule's output vocabulary a fixed point: everything a
rule can emit is registered, and a detected value already in that vocabulary is
suppressed rather than re-replaced. So ``mask(mask(x)) == mask(x)``.

The cost is real and worth stating plainly: **a person genuinely named "John
Doe" is never masked for that entity type.** Choose a vocabulary unlikely to
collide with the people in your data -- which for an Egyptian deployment is an
argument for obviously-Western placeholder names.

**A constant collapses distinct people.** With ``PERSON -> "John Doe"``,
"Ahmed emailed Sara about Omar" becomes "John Doe emailed John Doe about John
Doe". The prompt is now false, and a model asked to summarise it will answer
about one person. ``surrogate`` exists for this: it picks deterministically
from a pool keyed by the value's fingerprint, so distinct people stay distinct
and the same person is the same name across every turn of a conversation --
without storing a mapping anywhere.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "ReplacementRule",
    "ReplacementStrategy",
]


class ReplacementStrategy(StrEnum):
    PLACEHOLDER = "placeholder"
    """``<PERSON>``. Unambiguous, and the safe default."""

    CONSTANT = "constant"
    """One fixed string for every match. Collapses distinct values -- see above."""

    SURROGATE = "surrogate"
    """A stable pick from a pool, keyed by the value's fingerprint."""

    REDACT = "redact"
    """Remove the span entirely, leaving nothing."""

    LABELLED_FINGERPRINT = "labelled_fingerprint"
    """``<PERSON:9f75871d>`` -- distinct values stay distinct and an auditor can
    correlate against pii_events without the text carrying anything readable."""


class ReplacementRule(BaseModel):
    """How one entity type's spans are rewritten."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strategy: ReplacementStrategy = ReplacementStrategy.PLACEHOLDER
    value: str | None = None
    pool: tuple[str, ...] = ()
    placeholder: str | None = Field(
        default=None, description="Overrides the entity's own placeholder"
    )
    fingerprint_chars: int = Field(default=8, ge=4, le=16)

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.strategy is ReplacementStrategy.CONSTANT and not (self.value or "").strip():
            raise ValueError("strategy 'constant' requires a non-empty value")
        if self.strategy is ReplacementStrategy.SURROGATE:
            if not self.pool:
                raise ValueError("strategy 'surrogate' requires a non-empty pool")
            if any(not entry.strip() for entry in self.pool):
                raise ValueError("surrogate pool entries must be non-empty")
            if len(set(self.pool)) != len(self.pool):
                raise ValueError("surrogate pool contains duplicates")
        if self.strategy is not ReplacementStrategy.SURROGATE and self.pool:
            raise ValueError(f"strategy '{self.strategy}' must not set a pool")
        if self.strategy is not ReplacementStrategy.CONSTANT and self.value is not None:
            raise ValueError(f"strategy '{self.strategy}' must not set a value")
        return self

    # -- rendering ---------------------------------------------------------

    def render(self, *, placeholder: str, value_fp: str | None) -> str:
        """The text this span becomes.

        ``value_fp`` is the audit fingerprint. It is the only input that varies
        per value, which is deliberate: the replacement is derived from the
        keyed HMAC rather than from the value itself, so nothing here can
        reconstruct what was matched.
        """
        effective = self.placeholder or placeholder

        match self.strategy:
            case ReplacementStrategy.PLACEHOLDER:
                return effective
            case ReplacementStrategy.REDACT:
                return ""
            case ReplacementStrategy.CONSTANT:
                return self.value or effective
            case ReplacementStrategy.LABELLED_FINGERPRINT:
                if not value_fp:
                    return effective
                return f"{effective.rstrip('>')}:{value_fp[: self.fingerprint_chars]}>"
            case ReplacementStrategy.SURROGATE:
                if not value_fp or not self.pool:
                    # No fingerprint means the value canonicalised away; there
                    # is nothing stable to key on, so fall back to the
                    # unambiguous form rather than pick arbitrarily.
                    return effective
                return self.pool[int(value_fp, 16) % len(self.pool)]

        raise AssertionError(f"unhandled strategy {self.strategy}")  # pragma: no cover

    # -- idempotency -------------------------------------------------------

    def vocabulary(self, placeholder: str) -> frozenset[str]:
        """Every string this rule can emit.

        A detected value that is already in here has been masked once already,
        so masking it again would corrupt the text. The router suppresses those
        spans, which is what makes a second masking pass a no-op.

        ``labelled_fingerprint`` is excluded on purpose: its output varies with
        the value, so it cannot be enumerated -- but it also cannot be mistaken
        for a name by a detector, so it needs no suppression.
        """
        effective = self.placeholder or placeholder

        match self.strategy:
            case ReplacementStrategy.CONSTANT:
                return frozenset({self.value} if self.value else set()) | {effective}
            case ReplacementStrategy.SURROGATE:
                return frozenset(self.pool) | {effective}
            case _:
                return frozenset({effective})

    @property
    def is_realistic(self) -> bool:
        """True when the output could be mistaken for real data by a reader.

        Drives the warning in ``/health`` and the admin API: a deployment using
        these should know that its masked prompts no longer look masked.
        """
        return self.strategy in (
            ReplacementStrategy.CONSTANT,
            ReplacementStrategy.SURROGATE,
        )
