"""Detection orchestration: normalize, analyze, map back, deconflict, apply policy.

The invariant this module exists to hold: **everything it returns is expressed
in original-string coordinates**. Presidio runs against DIGITS-normalized text,
because a ``\\d{14}`` pattern cannot see ٢٨٥٠; the offsets that come back are in
that normalized space and are wrong for the caller. Mapping them back through
the offset map is not a nicety, it is the difference between masking a national
ID and masking six characters next to one.

Script routing (brief §4) is a Unicode block ratio -- cheap, no model, no
dependency. Tier 1 runs on everything; ``script_segments`` carves the text into
Arabic and Latin runs so tiers 2 and 3 can each be handed only the runs they
can actually read.
"""

from __future__ import annotations

import time
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from presidio_analyzer import AnalyzerEngine, RecognizerResult

from pii_service.detect.mask import splice
from pii_service.detect.normalize import DIGITS, map_span_to_original, normalize
from pii_service.detect.tier1_patterns import CONTEXT_TERM_KEY
from pii_service.policy.loader import PolicyBundle
from pii_service.policy.models import EntityAction, EntityCategory

__all__ = [
    "AnalysisOutcome",
    "DetectedSpan",
    "PiiRouter",
    "Script",
    "ScriptSegment",
    "dominant_script",
    "script_segments",
]

# Arabic script blocks: Arabic, Arabic Supplement, Arabic Extended-A/B,
# and the presentation-form blocks that survive a copy-paste out of a PDF.
_ARABIC_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x0600, 0x06FF),
    (0x0750, 0x077F),
    (0x0870, 0x089F),
    (0x08A0, 0x08FF),
    (0xFB50, 0xFDFF),
    (0xFE70, 0xFEFF),
)

# Below this share of script-bearing characters, a run is not worth sending to
# the Arabic tier.
_ARABIC_DOMINANCE_THRESHOLD: Final = 0.2


class Script(str):
    """Marker strings for the two script families we route on."""

    ARABIC: Final = "arabic"
    LATIN: Final = "latin"


def _is_arabic(char: str) -> bool:
    codepoint = ord(char)
    return any(low <= codepoint <= high for low, high in _ARABIC_RANGES)


def _is_script_bearing(char: str) -> bool:
    """Letters only. Digits, spaces and punctuation are script-neutral here."""
    return unicodedata.category(char).startswith("L")


def dominant_script(text: str) -> str:
    """Return ``Script.ARABIC`` or ``Script.LATIN`` for the text as a whole.

    Neutral characters do not vote. A prompt that is mostly code with one
    Arabic name in it still routes the Arabic run to the Arabic tier -- that is
    what ``script_segments`` is for -- but its *language* for Presidio's
    purposes is decided here.
    """
    arabic = sum(1 for ch in text if _is_script_bearing(ch) and _is_arabic(ch))
    bearing = sum(1 for ch in text if _is_script_bearing(ch))
    if not bearing:
        return Script.LATIN
    return Script.ARABIC if arabic / bearing >= _ARABIC_DOMINANCE_THRESHOLD else Script.LATIN


@dataclass(frozen=True, slots=True)
class ScriptSegment:
    """A maximal run of one script, in original coordinates."""

    script: str
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def script_segments(text: str, *, min_length: int = 1) -> list[ScriptSegment]:
    """Split ``text`` into maximal Arabic and Latin runs.

    Script-neutral characters (digits, spaces, punctuation) extend whichever run
    they follow, so "محمد، 25 سنة" stays one Arabic segment rather than three.

    Tier 2 gets the Arabic segments, tier 3 the Latin ones. Neither ever sees
    text in a script it was not trained on, which is the whole point: GLiNER2
    does not support Arabic, and feeding it Arabic anyway produces confident
    nonsense rather than an error.
    """
    if not text:
        return []

    segments: list[ScriptSegment] = []
    current_script: str | None = None
    current_start = 0

    for index, char in enumerate(text):
        if not _is_script_bearing(char):
            continue
        script = Script.ARABIC if _is_arabic(char) else Script.LATIN
        if current_script is None:
            current_script = script
            current_start = 0
        elif script != current_script:
            segments.append(ScriptSegment(current_script, current_start, index))
            current_script = script
            current_start = index

    if current_script is not None:
        segments.append(ScriptSegment(current_script, current_start, len(text)))

    return [s for s in segments if s.length >= min_length]


@dataclass(slots=True)
class DetectedSpan:
    """One finding, in original coordinates.

    ``value`` is the matched text. It is needed to compute the fingerprint and
    the preview, and it must never leave this process: not in the API response,
    not in a log line, not in an exception message.

    Hence the custom ``__repr__``. A dataclass-generated one would put the
    matched national ID into every traceback, every ``logger.debug("%s", span)``
    and every pytest assertion diff -- which is exactly the accident the brief's
    logging test is written to catch, and this makes that accident impossible
    rather than merely detectable.
    """

    entity_type: str
    recognizer: str
    score: float
    start: int
    end: int
    value: str = field(repr=False)
    action: EntityAction
    category: EntityCategory
    lang: str
    context_term: str | None = None

    def __repr__(self) -> str:
        return (
            f"DetectedSpan({self.entity_type} {self.start}:{self.end} "
            f"score={self.score:.2f} len={len(self.value)} value=<redacted>)"
        )

    @property
    def value_len(self) -> int:
        return len(self.value)


@dataclass(frozen=True, slots=True)
class AnalysisOutcome:
    """Everything one analysis produced."""

    original_text: str
    anonymized_text: str
    spans: list[DetectedSpan]
    lang: str
    latency_ms: int

    @property
    def blocked(self) -> bool:
        return any(span.action is EntityAction.BLOCK for span in self.spans)

    @property
    def masked_entity_counts(self) -> dict[str, int]:
        """Counts by entity type -- the only detection detail safe to hand LiteLLM."""
        counts: dict[str, int] = {}
        for span in self.spans:
            counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
        return counts


class PiiRouter:
    """Runs the tiers and turns their output into policy decisions."""

    def __init__(self, analyzer: AnalyzerEngine, policy: PolicyBundle) -> None:
        self._analyzer: Final = analyzer
        self._policy: Final = policy

    def analyze(self, text: str, *, language: str | None = None) -> AnalysisOutcome:
        """Detect, apply policy and produce the masked text."""
        started = time.perf_counter()

        if not text:
            return AnalysisOutcome(
                original_text=text,
                anonymized_text=text,
                spans=[],
                lang=language or Script.LATIN,
                latency_ms=0,
            )

        lang = language or self._presidio_language(text)

        normalized, offset_map = normalize(text, DIGITS)
        raw_results = (
            self._analyzer.analyze(text=normalized, language=lang) if normalized else []
        )

        spans = self._to_original_spans(raw_results, normalized, offset_map, text, lang)
        spans = self._apply_thresholds(spans)
        spans = _deconflict(spans)
        spans.sort(key=lambda s: (s.start, s.end))

        anonymized = self._anonymize(text, spans)
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        return AnalysisOutcome(
            original_text=text,
            anonymized_text=anonymized,
            spans=spans,
            lang=lang,
            latency_ms=elapsed_ms,
        )

    # -- internals ---------------------------------------------------------

    def _presidio_language(self, text: str) -> str:
        """Map the detected script onto a language the registry was built for."""
        supported = set(self._analyzer.supported_languages)
        if dominant_script(text) == Script.ARABIC and "ar" in supported:
            return "ar"
        return "en" if "en" in supported else next(iter(sorted(supported)))

    def _to_original_spans(
        self,
        results: Iterable[RecognizerResult],
        normalized: str,
        offset_map: list[int],
        text: str,
        lang: str,
    ) -> list[DetectedSpan]:
        spans: list[DetectedSpan] = []
        for result in results:
            if not 0 <= result.start < result.end <= len(normalized):
                # A recognizer returned a span outside the text it was given.
                # Dropping it is right: a bad offset would splice the wrong
                # characters and could expose the PII it was meant to hide.
                continue

            start, end = map_span_to_original(offset_map, result.start, result.end, len(text))
            policy = self._policy.policy_for(result.entity_type)
            if policy is None:
                continue

            metadata = result.recognition_metadata or {}
            spans.append(
                DetectedSpan(
                    entity_type=result.entity_type,
                    recognizer=str(metadata.get("recognizer_name") or "unknown"),
                    score=float(result.score),
                    start=start,
                    end=end,
                    value=text[start:end],
                    action=policy.action,
                    category=policy.category,
                    lang=lang,
                    context_term=metadata.get(CONTEXT_TERM_KEY),
                )
            )
        return spans

    def _apply_thresholds(self, spans: Sequence[DetectedSpan]) -> list[DetectedSpan]:
        kept: list[DetectedSpan] = []
        for span in spans:
            policy = self._policy.policy_for(span.entity_type)
            if policy is not None and span.score >= policy.score_threshold:
                kept.append(span)
        return kept

    def _anonymize(self, text: str, spans: Sequence[DetectedSpan]) -> str:
        """Splice placeholders over MASK spans, leaving ALLOW spans intact."""
        return splice(
            text,
            [
                (span.start, span.end, self._placeholder(span.entity_type))
                for span in spans
                if span.action is EntityAction.MASK
            ],
        )

    def _placeholder(self, entity_type: str) -> str:
        policy = self._policy.policy_for(entity_type)
        return policy.placeholder if policy else f"<{entity_type}>"


# ---------------------------------------------------------------------------
# Conflict resolution
# ---------------------------------------------------------------------------


def _specificity(entity_type: str) -> int:
    """Country-specific entities outrank generic ones on an identical span.

    This matters concretely: a 14-digit national ID sits inside the credit
    card recognizer's 13-19 digit window, and roughly one in ten of them will
    satisfy Luhn by chance. Both findings mask the same characters, so nothing
    leaks either way -- but the audit row would say CREDIT_CARD, and an
    investigator pivoting on card fraud would be reading national IDs.
    """
    return 1 if entity_type.startswith("EG_") or entity_type.startswith("AR_") else 0


def _deconflict(spans: list[DetectedSpan]) -> list[DetectedSpan]:
    """Drop spans that are contained in, or duplicate, a better span.

    Ordering: longest first, then highest score, then most specific entity,
    then entity name for determinism. The first span to claim a region wins it.
    """
    ordered = sorted(
        spans,
        key=lambda s: (
            -(s.end - s.start),
            -s.score,
            -_specificity(s.entity_type),
            s.entity_type,
            s.start,
        ),
    )

    kept: list[DetectedSpan] = []
    for span in ordered:
        if any(keeper.start <= span.start and span.end <= keeper.end for keeper in kept):
            continue
        kept.append(span)
    return kept
