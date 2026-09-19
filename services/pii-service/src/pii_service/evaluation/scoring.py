"""Scoring for the Arabic NER comparison.

The brief (§5) is explicit about the primary metric: **coverage**, the share of
gold spans overlapped by any prediction, *not* exact-match F1. A sloppy
boundary that still covers the name is a pass; a missed span is a leak. Exact
match punishes the first as hard as the second, which is the wrong ranking for
a masking system.

Two corrections to that, both of which a coverage-only score would hide:

**Type confusion does not cause a leak.** If a PER is predicted as LOC the span
is still masked, so ``coverage_any`` ignores the label. ``coverage_typed`` is
reported alongside it as a quality signal, but it does not decide the ranking.

**A model that masks everything scores 100% coverage.** It is also useless: it
would redact اليوم and الشركة out of every prompt on the gateway. So the
negative cases carry a false-positive rate, and over-masking is measured in
characters. The verdict ranks on coverage first and refuses a variant whose
false-positive rate exceeds a stated ceiling.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Final

from pii_service.evaluation.gold import GoldCase, GoldSet, GoldSpan, Register

__all__ = [
    "MAX_ACCEPTABLE_FP_PER_SENTENCE",
    "Prediction",
    "RegisterScore",
    "VariantScore",
    "compare",
    "score_variant",
]

# A model that fires more than this on entity-free text makes the gateway
# unusable, whatever its coverage. Chosen so that roughly one spurious span in
# ten clean prompts is tolerable and one in three is not.
MAX_ACCEPTABLE_FP_PER_SENTENCE: Final = 0.10

# Coverage below this is disqualifying regardless of precision: the point of the
# tier is to find Arabic names, and one in five missed is not a working system.
MIN_ACCEPTABLE_COVERAGE: Final = 0.80


@dataclass(frozen=True, slots=True)
class Prediction:
    start: int
    end: int
    label: str
    score: float = 1.0

    def overlaps(self, span: GoldSpan) -> bool:
        return self.start < span.end and span.start < self.end


def _covered_chars(spans: Iterable[tuple[int, int]]) -> set[int]:
    covered: set[int] = set()
    for start, end in spans:
        covered.update(range(start, end))
    return covered


@dataclass
class RegisterScore:
    """Scores for one register, so a dialect regression cannot hide in an average."""

    register: str
    gold_spans: int = 0
    covered_any: int = 0
    covered_typed: int = 0
    gold_chars: int = 0
    covered_gold_chars: int = 0

    @property
    def coverage_any(self) -> float:
        return self.covered_any / self.gold_spans if self.gold_spans else 0.0

    @property
    def coverage_typed(self) -> float:
        return self.covered_typed / self.gold_spans if self.gold_spans else 0.0

    @property
    def char_recall(self) -> float:
        return self.covered_gold_chars / self.gold_chars if self.gold_chars else 0.0


@dataclass
class VariantScore:
    """Everything measured for one model variant."""

    name: str
    gold_spans: int = 0
    covered_any: int = 0
    covered_typed: int = 0
    gold_chars: int = 0
    covered_gold_chars: int = 0
    predicted_spans: int = 0
    predicted_hitting_gold: int = 0
    negative_sentences: int = 0
    false_positive_spans: int = 0
    over_masked_chars: int = 0
    total_chars: int = 0
    latency_ms: list[float] = field(default_factory=list)
    per_register: dict[str, RegisterScore] = field(default_factory=dict)
    per_label: dict[str, RegisterScore] = field(default_factory=dict)

    # -- primary -----------------------------------------------------------

    @property
    def coverage_any(self) -> float:
        """THE metric. Share of gold spans overlapped by any prediction."""
        return self.covered_any / self.gold_spans if self.gold_spans else 0.0

    @property
    def coverage_typed(self) -> float:
        return self.covered_typed / self.gold_spans if self.gold_spans else 0.0

    @property
    def char_recall(self) -> float:
        return self.covered_gold_chars / self.gold_chars if self.gold_chars else 0.0

    # -- the cost side -----------------------------------------------------

    @property
    def span_precision(self) -> float:
        """Share of predictions that touch a gold span."""
        return self.predicted_hitting_gold / self.predicted_spans if self.predicted_spans else 0.0

    @property
    def false_positives_per_clean_sentence(self) -> float:
        return (
            self.false_positive_spans / self.negative_sentences if self.negative_sentences else 0.0
        )

    @property
    def over_mask_ratio(self) -> float:
        """Share of all characters masked that did not need to be."""
        return self.over_masked_chars / self.total_chars if self.total_chars else 0.0

    @property
    def p50_ms(self) -> float:
        return _quantile(sorted(self.latency_ms), 0.50)

    @property
    def p95_ms(self) -> float:
        return _quantile(sorted(self.latency_ms), 0.95)

    # -- verdict inputs ----------------------------------------------------

    @property
    def is_acceptable(self) -> bool:
        return (
            self.coverage_any >= MIN_ACCEPTABLE_COVERAGE
            and self.false_positives_per_clean_sentence <= MAX_ACCEPTABLE_FP_PER_SENTENCE
        )

    def rejection_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.coverage_any < MIN_ACCEPTABLE_COVERAGE:
            reasons.append(
                f"coverage {self.coverage_any:.1%} is below the "
                f"{MIN_ACCEPTABLE_COVERAGE:.0%} floor — it misses too much PII"
            )
        if self.false_positives_per_clean_sentence > MAX_ACCEPTABLE_FP_PER_SENTENCE:
            reasons.append(
                f"{self.false_positives_per_clean_sentence:.2f} false positives per clean "
                f"sentence exceeds the {MAX_ACCEPTABLE_FP_PER_SENTENCE:.2f} ceiling — "
                "it would mask ordinary words out of normal prompts"
            )
        return reasons

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "coverage_any": round(self.coverage_any, 4),
            "coverage_typed": round(self.coverage_typed, 4),
            "char_recall": round(self.char_recall, 4),
            "span_precision": round(self.span_precision, 4),
            "false_positives_per_clean_sentence": round(self.false_positives_per_clean_sentence, 4),
            "over_mask_ratio": round(self.over_mask_ratio, 4),
            "gold_spans": self.gold_spans,
            "predicted_spans": self.predicted_spans,
            "p50_ms": round(self.p50_ms, 2),
            "p95_ms": round(self.p95_ms, 2),
            "acceptable": self.is_acceptable,
            "rejection_reasons": self.rejection_reasons(),
            "per_register": {
                name: {
                    "coverage_any": round(score.coverage_any, 4),
                    "coverage_typed": round(score.coverage_typed, 4),
                    "gold_spans": score.gold_spans,
                }
                for name, score in sorted(self.per_register.items())
            },
            "per_label": {
                name: {
                    "coverage_any": round(score.coverage_any, 4),
                    "gold_spans": score.gold_spans,
                }
                for name, score in sorted(self.per_label.items())
            },
        }


def _quantile(ordered: Sequence[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
    return ordered[index]


def score_variant(
    name: str,
    gold: GoldSet,
    predictions: dict[str, Sequence[Prediction]],
    latencies: dict[str, float] | None = None,
) -> VariantScore:
    """Score one variant's predictions, keyed by gold case id."""
    result = VariantScore(name=name)
    latencies = latencies or {}

    for case in gold.cases:
        predicted = list(predictions.get(case.case_id, ()))
        result.total_chars += len(case.text)
        result.predicted_spans += len(predicted)

        if case.case_id in latencies:
            result.latency_ms.append(latencies[case.case_id])

        register = result.per_register.setdefault(
            str(case.register), RegisterScore(register=str(case.register))
        )

        if case.is_negative:
            result.negative_sentences += 1
            result.false_positive_spans += len(predicted)
        else:
            _score_positive(result, register, case, predicted)

        gold_chars = _covered_chars((s.start, s.end) for s in case.spans)
        predicted_chars = _covered_chars((p.start, p.end) for p in predicted)
        result.over_masked_chars += len(predicted_chars - gold_chars)

    return result


def _score_positive(
    result: VariantScore,
    register: RegisterScore,
    case: GoldCase,
    predicted: Sequence[Prediction],
) -> None:
    for span in case.spans:
        result.gold_spans += 1
        result.gold_chars += span.length
        register.gold_spans += 1
        register.gold_chars += span.length

        label_score = result.per_label.setdefault(span.label, RegisterScore(register=span.label))
        label_score.gold_spans += 1
        label_score.gold_chars += span.length

        touching = [p for p in predicted if p.overlaps(span)]
        if touching:
            result.covered_any += 1
            register.covered_any += 1
            label_score.covered_any += 1

            if any(p.label == span.label for p in touching):
                result.covered_typed += 1
                register.covered_typed += 1
                label_score.covered_typed += 1

            overlap = len(
                _covered_chars((p.start, p.end) for p in touching)
                & set(range(span.start, span.end))
            )
            result.covered_gold_chars += overlap
            register.covered_gold_chars += overlap
            label_score.covered_gold_chars += overlap

    for prediction in predicted:
        if any(prediction.overlaps(span) for span in case.spans):
            result.predicted_hitting_gold += 1


@dataclass(frozen=True, slots=True)
class Comparison:
    winner: str | None
    reason: str
    scores: tuple[VariantScore, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "winner": self.winner,
            "reason": self.reason,
            "variants": [score.as_dict() for score in self.scores],
        }


def compare(scores: Sequence[VariantScore]) -> Comparison:
    """Rank variants: coverage first, then false positives, then latency.

    Coverage first because a missed span is a leak and a leak is the failure
    this system exists to prevent. False positives break the tie, because past
    the acceptability gate the remaining difference is usability rather than
    safety. Latency breaks that tie in turn -- it is real, this runs pre_call
    on every request, but it is the cheapest of the three to fix later by
    turning the tier off.
    """
    if not scores:
        return Comparison(winner=None, reason="no variants were scored", scores=())

    acceptable = [s for s in scores if s.is_acceptable]
    if not acceptable:
        detail = "; ".join(f"{s.name}: {', '.join(s.rejection_reasons())}" for s in scores)
        return Comparison(
            winner=None,
            reason=f"no variant met the acceptability gate ({detail})",
            scores=tuple(scores),
        )

    ranked = sorted(
        acceptable,
        key=lambda s: (
            -round(s.coverage_any, 3),
            s.false_positives_per_clean_sentence,
            s.p95_ms,
            s.name,
        ),
    )
    best = ranked[0]

    if len(ranked) == 1:
        reason = (
            f"{best.name} is the only variant meeting the gate "
            f"(coverage {best.coverage_any:.1%}, "
            f"{best.false_positives_per_clean_sentence:.2f} FP/clean sentence)"
        )
    else:
        runner_up = ranked[1]
        delta = best.coverage_any - runner_up.coverage_any
        if round(delta, 3) == 0:
            reason = (
                f"{best.name} and {runner_up.name} tie on coverage "
                f"({best.coverage_any:.1%}); {best.name} wins on false positives "
                f"({best.false_positives_per_clean_sentence:.2f} vs "
                f"{runner_up.false_positives_per_clean_sentence:.2f} per clean sentence)"
            )
        else:
            reason = (
                f"{best.name} covers {delta:.1%} more gold spans than {runner_up.name} "
                f"({best.coverage_any:.1%} vs {runner_up.coverage_any:.1%})"
            )

    return Comparison(winner=best.name, reason=reason, scores=tuple(ranked))


def register_breakdown(score: VariantScore) -> list[tuple[str, float, int]]:
    """(register, coverage_any, gold_spans), Egyptian first -- it decides this."""
    order = {
        str(Register.EGYPTIAN): 0,
        str(Register.MIXED_SCRIPT): 1,
        str(Register.MSA): 2,
    }
    return sorted(
        (
            (name, entry.coverage_any, entry.gold_spans)
            for name, entry in score.per_register.items()
        ),
        key=lambda row: order.get(row[0], 99),
    )
