"""Tier 1: deterministic recognizers.

Highest priority, near-zero false positives, and the tier the Egyptian national
ID belongs to -- never a model (brief §4).

Every recognizer here expects **DIGITS-normalized** text, so Arabic-Indic,
Persian and full-width digits have already been folded to ASCII and zero-width
and bidi characters are gone. Offsets are in that normalized coordinate space;
``detect/router.py`` maps them back to the original before anything is reported.

How the three-way score works, because it is load-bearing and non-obvious:
Presidio's ``PatternRecognizer`` turns ``validate_result`` into a score
directly -- ``True`` becomes 1.0, ``False`` drops the match, ``None`` keeps the
pattern's own score. So a national ID pattern scored 0.85 that returns ``None``
for a structurally valid ID with an unverified check digit, and ``True`` when
the check digit verifies, lands exactly on the brief's 1.0 / 0.85 split without
any score arithmetic of our own.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from datetime import date
from typing import Final

from presidio_analyzer import Pattern, PatternRecognizer, RecognizerResult

from pii_service.detect.context import (
    DEFAULT_CONTEXT_BOOST,
    DEFAULT_CONTEXT_WINDOW,
    find_context_term,
    normalize_context_terms,
)
from pii_service.detect.eg_validators import (
    iban_is_valid,
    normalize_eg_mobile,
    parse_national_id,
    tax_id_is_plausible,
)

__all__ = [
    "CONTEXT_TERM_KEY",
    "ContextBoostedPatternRecognizer",
    "EgyptianIbanRecognizer",
    "EgyptianMobileRecognizer",
    "EgyptianNationalIdRecognizer",
    "EgyptianPassportRecognizer",
    "EgyptianTaxIdRecognizer",
    "build_tier1_recognizers",
]

# Key under which a fired context term is recorded on the result, so the audit
# row can say *why* a low-base-score finding cleared its threshold.
CONTEXT_TERM_KEY: Final = "pii_context_term"

# No IGNORECASE: passport and IBAN letters are meaningfully uppercase, and
# case-folding them turns every `a12345678` variable name into a passport.
_CASE_SENSITIVE_FLAGS: Final = re.DOTALL | re.MULTILINE
_CASE_INSENSITIVE_FLAGS: Final = re.DOTALL | re.MULTILINE | re.IGNORECASE


class ContextBoostedPatternRecognizer(PatternRecognizer):
    """A ``PatternRecognizer`` that raises its score near a configured term.

    Note that the terms are *not* handed to ``PatternRecognizer``'s own
    ``context`` argument. That feeds Presidio's lemma-based enhancer, which
    cannot work here and would double-boost if a lemmatizer were ever added to
    the pipeline. See ``detect/context.py``.
    """

    def __init__(
        self,
        *,
        supported_entity: str,
        patterns: list[Pattern],
        supported_language: str,
        name: str,
        context_terms: Sequence[str] = (),
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        context_boost: float = DEFAULT_CONTEXT_BOOST,
        max_score: float = 1.0,
        global_regex_flags: int = _CASE_INSENSITIVE_FLAGS,
    ) -> None:
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            supported_language=supported_language,
            name=name,
            global_regex_flags=global_regex_flags,
        )
        self._context_terms: Final = normalize_context_terms(context_terms)
        self._context_window: Final = context_window
        self._context_boost: Final = context_boost
        self._max_score: Final = max_score

    def analyze(  # noqa: D102 -- inherited
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: object | None = None,
        regex_flags: int | None = None,
    ) -> list[RecognizerResult]:
        results = super().analyze(text, entities, nlp_artifacts, regex_flags)  # type: ignore[arg-type]
        if not self._context_terms:
            return results

        for result in results:
            term = find_context_term(
                text, result.start, result.end, self._context_terms, self._context_window
            )
            if term is None:
                continue
            result.score = min(self._max_score, result.score + self._context_boost)
            if result.recognition_metadata is None:  # pragma: no cover - defensive
                result.recognition_metadata = {}
            result.recognition_metadata[CONTEXT_TERM_KEY] = term

        return results


# ---------------------------------------------------------------------------
# National ID
# ---------------------------------------------------------------------------


class EgyptianNationalIdRecognizer(ContextBoostedPatternRecognizer):
    """14-digit Egyptian national ID, structurally validated.

    ``today`` is injectable so the "birth date is not in the future" rule is
    testable without the tests going stale.
    """

    ENTITY: Final = "EG_NATIONAL_ID"

    def __init__(
        self,
        *,
        supported_language: str = "en",
        context_terms: Sequence[str] = (),
        today: date | None = None,
    ) -> None:
        self._today: Final = today
        super().__init__(
            supported_entity=self.ENTITY,
            name="EgyptianNationalIdRecognizer",
            supported_language=supported_language,
            patterns=[
                Pattern(
                    name="eg_nid_14_digits",
                    # Anchored so a 14-digit window inside a longer digit run is
                    # not reported -- that run is an order number, not an ID.
                    regex=r"(?<!\d)\d{14}(?!\d)",
                    # The unverified-checksum score. validate_result promotes a
                    # verified one to 1.0.
                    score=0.85,
                )
            ],
            context_terms=context_terms,
        )

    def validate_result(self, pattern_text: str) -> bool | None:
        parse = parse_national_id(pattern_text, today=self._today)
        if parse is None:
            return False  # structurally impossible -- drop it
        if parse.checksum_valid:
            return True  # -> 1.0
        return None  # -> keep 0.85; the checksum is a booster, not a gate


# ---------------------------------------------------------------------------
# Mobile
# ---------------------------------------------------------------------------


class EgyptianMobileRecognizer(ContextBoostedPatternRecognizer):
    """Egyptian mobile numbers: ``01[0125]`` + 8 digits, local and ``+20`` forms."""

    ENTITY: Final = "EG_MOBILE"

    def __init__(
        self,
        *,
        supported_language: str = "en",
        context_terms: Sequence[str] = (),
    ) -> None:
        super().__init__(
            supported_entity=self.ENTITY,
            name="EgyptianMobileRecognizer",
            supported_language=supported_language,
            patterns=[
                Pattern(
                    name="eg_mobile_international",
                    regex=r"(?<!\d)(?:\+|00)?20[ \-.]?1[0125][ \-.]?\d{4}[ \-.]?\d{4}(?!\d)",
                    score=0.7,
                ),
                Pattern(
                    name="eg_mobile_local",
                    # The lookbehind also excludes '+', so the local pattern
                    # cannot match the tail of an international number.
                    regex=r"(?<![\d+])01[0125][ \-.]?\d{4}[ \-.]?\d{4}(?!\d)",
                    score=0.6,
                ),
            ],
            context_terms=context_terms,
        )

    def invalidate_result(self, pattern_text: str) -> bool:
        """Drop anything the E.164 normalizer will not accept."""
        return normalize_eg_mobile(pattern_text) is None


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------


class EgyptianIbanRecognizer(ContextBoostedPatternRecognizer):
    """Egyptian IBAN: ``EG`` + 2 check digits + 25, validated by mod-97.

    Here the checksum *is* a gate. Unlike the national ID algorithm, IBAN mod-97
    is unambiguous and universally implemented the same way, so a failure means
    the string is not an IBAN rather than that we disagree about the arithmetic.
    """

    ENTITY: Final = "EG_IBAN"

    def __init__(
        self,
        *,
        supported_language: str = "en",
        context_terms: Sequence[str] = (),
    ) -> None:
        super().__init__(
            supported_entity=self.ENTITY,
            name="EgyptianIbanRecognizer",
            supported_language=supported_language,
            patterns=[
                Pattern(
                    name="eg_iban",
                    regex=r"(?<![A-Za-z0-9])EG\d{2}(?:[ \-]?[A-Za-z0-9]){25}(?![A-Za-z0-9])",
                    score=0.8,
                )
            ],
            context_terms=context_terms,
        )

    def invalidate_result(self, pattern_text: str) -> bool:
        return not iban_is_valid(pattern_text)


# ---------------------------------------------------------------------------
# Tax ID and passport -- shape-only, so context carries them
# ---------------------------------------------------------------------------


class EgyptianTaxIdRecognizer(ContextBoostedPatternRecognizer):
    """9-digit Egyptian tax registration number, usually written 123-456-789.

    There is no public check digit, and nine digits is an extremely common
    shape, so the base score is deliberately below every sane threshold: this
    entity only ever reports when a context term is present.
    """

    ENTITY: Final = "EG_TAX_ID"

    def __init__(
        self,
        *,
        supported_language: str = "en",
        context_terms: Sequence[str] = (),
    ) -> None:
        super().__init__(
            supported_entity=self.ENTITY,
            name="EgyptianTaxIdRecognizer",
            supported_language=supported_language,
            patterns=[
                Pattern(
                    name="eg_tax_id",
                    regex=r"(?<![\d/-])\d{3}[ \-/]?\d{3}[ \-/]?\d{3}(?![\d/-])",
                    score=0.3,
                )
            ],
            context_terms=context_terms,
        )

    def invalidate_result(self, pattern_text: str) -> bool:
        return not tax_id_is_plausible(pattern_text)


class EgyptianPassportRecognizer(ContextBoostedPatternRecognizer):
    """Egyptian passport number: one uppercase letter followed by 8 digits.

    Shape-only and therefore context-gated, exactly like the tax ID.
    """

    ENTITY: Final = "EG_PASSPORT"

    def __init__(
        self,
        *,
        supported_language: str = "en",
        context_terms: Sequence[str] = (),
    ) -> None:
        super().__init__(
            supported_entity=self.ENTITY,
            name="EgyptianPassportRecognizer",
            supported_language=supported_language,
            patterns=[
                Pattern(
                    name="eg_passport",
                    regex=r"(?<![A-Za-z0-9])[A-Z]\d{8}(?![A-Za-z0-9])",
                    score=0.3,
                )
            ],
            context_terms=context_terms,
            global_regex_flags=_CASE_SENSITIVE_FLAGS,
        )


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_tier1_recognizers(
    *,
    supported_language: str,
    context_terms: dict[str, Sequence[str]],
    today: date | None = None,
) -> list[PatternRecognizer]:
    """Instantiate every tier-1 recognizer for one language.

    Tier 1 runs on everything regardless of detected script (brief §4), so the
    registry gets one copy of this list per supported language.

    ``context_terms`` is the ``context_terms`` block of ``gazetteer_eg.yaml``,
    keyed by family (``national_id``, ``mobile``, ...).
    """
    return [
        EgyptianNationalIdRecognizer(
            supported_language=supported_language,
            context_terms=context_terms.get("national_id", ()),
            today=today,
        ),
        EgyptianMobileRecognizer(
            supported_language=supported_language,
            context_terms=context_terms.get("mobile", ()),
        ),
        EgyptianIbanRecognizer(
            supported_language=supported_language,
            context_terms=context_terms.get("bank", ()),
        ),
        EgyptianTaxIdRecognizer(
            supported_language=supported_language,
            context_terms=context_terms.get("tax", ()),
        ),
        EgyptianPassportRecognizer(
            supported_language=supported_language,
            context_terms=context_terms.get("passport", ()),
        ),
    ]
