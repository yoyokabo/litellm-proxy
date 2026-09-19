"""The gold set and the scoring that picks a CAMeLBERT variant.

If this scoring is wrong, the model decision is wrong, and nobody finds out
until Arabic names are going through the gateway unmasked. So the ranking rules
are tested as carefully as the detector itself.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_service.evaluation.gold import GoldSet, Register, parse_markup
from pii_service.evaluation.scoring import (
    MAX_ACCEPTABLE_FP_PER_SENTENCE,
    MIN_ACCEPTABLE_COVERAGE,
    Prediction,
    compare,
    register_breakdown,
    score_variant,
)

GOLD_PATH = Path(__file__).resolve().parents[1] / "eval" / "arabic_ner_gold.yaml"


# ---------------------------------------------------------------------------
# Markup parsing -- offsets correct by construction
# ---------------------------------------------------------------------------


def test_markup_produces_offsets_that_slice_back_to_the_surface() -> None:
    text, spans = parse_markup("اجتمع [محمد علي](PER) في [القاهرة](LOC).")

    assert text == "اجتمع محمد علي في القاهرة."
    assert [text[s.start : s.end] for s in spans] == ["محمد علي", "القاهرة"]
    assert [s.label for s in spans] == ["PER", "LOC"]


def test_markup_with_no_entities_is_a_negative_case() -> None:
    text, spans = parse_markup("لا يوجد شيء هنا.")
    assert text == "لا يوجد شيء هنا."
    assert spans == ()


def test_adjacent_entities_keep_distinct_offsets() -> None:
    text, spans = parse_markup("[هدى](PER) و[نورا](PER)")
    assert [text[s.start : s.end] for s in spans] == ["هدى", "نورا"]
    assert spans[0].end < spans[1].start


def test_entity_at_the_very_start_and_end() -> None:
    text, spans = parse_markup("[محمد](PER) قال [القاهرة](LOC)")
    assert spans[0].start == 0
    assert spans[-1].end == len(text)


def test_markup_inside_latin_text_keeps_character_offsets() -> None:
    text, spans = parse_markup('{"name": "[أحمد](PER)"}')
    assert text[spans[0].start : spans[0].end] == "أحمد"


def test_unknown_entity_type_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown entity type"):
        parse_markup("[محمد](PERSON)")


def test_diacritics_and_tatweel_are_inside_the_span() -> None:
    text, spans = parse_markup("الاسم [مُحَمَّد](PER) هنا")
    assert text[spans[0].start : spans[0].end] == "مُحَمَّد"


# ---------------------------------------------------------------------------
# The shipped gold set
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def gold() -> GoldSet:
    return GoldSet.load(GOLD_PATH)


def test_the_gold_set_loads(gold: GoldSet) -> None:
    assert len(gold.cases) >= 50
    assert gold.span_count >= 80


def test_every_gold_span_slices_to_non_empty_text(gold: GoldSet) -> None:
    for case in gold.cases:
        for span in case.spans:
            surface = case.text[span.start : span.end]
            assert surface.strip(), f"{case.case_id} has an empty span"
            assert "[" not in surface and "](" not in surface, f"{case.case_id} kept markup"


def test_no_markup_leaked_into_any_plain_text(gold: GoldSet) -> None:
    for case in gold.cases:
        assert "](" not in case.text, f"{case.case_id} still contains markup"


def test_all_three_labels_are_represented(gold: GoldSet) -> None:
    counts = gold.label_counts()
    assert set(counts) == {"PER", "LOC", "ORG"}
    assert all(count >= 10 for count in counts.values())


def test_the_egyptian_register_is_the_largest(gold: GoldSet) -> None:
    """Dialect is what separates mix-ner from msa-ner, and what we actually see."""
    by_register = {r: len(gold.by_register(r)) for r in Register}
    assert by_register[Register.EGYPTIAN] >= by_register[Register.MSA]


def test_there_are_enough_negatives_to_measure_false_positives(gold: GoldSet) -> None:
    assert len(gold.negatives) >= 10


def test_case_ids_are_unique(gold: GoldSet) -> None:
    ids = [case.case_id for case in gold.cases]
    assert len(ids) == len(set(ids))


def test_duplicate_ids_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text(
        "version: 1\ncases:\n  - {id: a, text: 'x'}\n  - {id: a, text: 'y'}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate gold case id"):
        GoldSet.load(path)


def test_an_empty_gold_set_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "empty.yaml"
    path.write_text("version: 1\ncases: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        GoldSet.load(path)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def _tiny_gold(tmp_path: Path) -> GoldSet:
    path = tmp_path / "tiny.yaml"
    path.write_text(
        "version: 1\n"
        "cases:\n"
        "  - {id: p1, register: egyptian, text: '[محمد](PER) في [القاهرة](LOC)'}\n"
        "  - {id: p2, register: msa, text: 'زار [أحمد](PER) المكان'}\n"
        "  - {id: n1, register: egyptian, text: 'لا يوجد شيء هنا'}\n",
        encoding="utf-8",
    )
    return GoldSet.load(path)


def test_perfect_predictions_score_full_coverage(tmp_path: Path) -> None:
    gold = _tiny_gold(tmp_path)
    predictions = {
        case.case_id: [Prediction(s.start, s.end, s.label) for s in case.spans]
        for case in gold.cases
    }
    score = score_variant("perfect", gold, predictions)

    assert score.coverage_any == 1.0
    assert score.coverage_typed == 1.0
    assert score.false_positives_per_clean_sentence == 0.0
    assert score.is_acceptable


def test_a_sloppy_boundary_still_counts_as_covered(tmp_path: Path) -> None:
    """The brief's rule: a span that overlaps the gold name is a pass."""
    gold = _tiny_gold(tmp_path)
    case = gold.cases[0]
    span = case.spans[0]
    predictions = {case.case_id: [Prediction(span.start, span.end + 3, span.label)]}

    score = score_variant("sloppy", gold, predictions)
    assert score.covered_any == 1
    # Every character of *that* span was inside the prediction. char_recall is
    # over the whole set, so it is not 1.0 here -- only one span was predicted.
    assert score.covered_gold_chars == span.length


def test_a_one_character_overlap_counts(tmp_path: Path) -> None:
    """Kept narrow on purpose: a wider span would also reach the next gold span."""
    gold = _tiny_gold(tmp_path)
    case = gold.cases[0]
    span = case.spans[0]
    predictions = {case.case_id: [Prediction(span.end - 1, span.end + 1, span.label)]}

    assert score_variant("edge", gold, predictions).covered_any == 1


def test_an_adjacent_but_non_overlapping_span_does_not_count(tmp_path: Path) -> None:
    gold = _tiny_gold(tmp_path)
    case = gold.cases[0]
    span = case.spans[0]
    predictions = {case.case_id: [Prediction(span.end, span.end + 4, span.label)]}

    assert score_variant("adjacent", gold, predictions).covered_any == 0


def test_the_wrong_type_still_counts_as_covered(tmp_path: Path) -> None:
    """A PER predicted as LOC is still masked, so it is not a leak."""
    gold = _tiny_gold(tmp_path)
    case = gold.cases[0]
    span = case.spans[0]
    predictions = {case.case_id: [Prediction(span.start, span.end, "LOC")]}

    score = score_variant("confused", gold, predictions)
    assert score.covered_any == 1
    assert score.covered_typed == 0


def test_predictions_on_a_clean_sentence_are_false_positives(tmp_path: Path) -> None:
    gold = _tiny_gold(tmp_path)
    predictions = {"n1": [Prediction(0, 2, "PER"), Prediction(3, 7, "LOC")]}

    score = score_variant("noisy", gold, predictions)
    assert score.false_positive_spans == 2
    assert score.false_positives_per_clean_sentence == 2.0
    assert not score.is_acceptable


def test_a_model_that_masks_everything_is_rejected(tmp_path: Path) -> None:
    """Full coverage, zero usefulness -- the gate exists for exactly this."""
    gold = _tiny_gold(tmp_path)
    predictions = {case.case_id: [Prediction(0, len(case.text), "PER")] for case in gold.cases}

    score = score_variant("mask-all", gold, predictions)
    assert score.coverage_any == 1.0
    assert not score.is_acceptable
    assert any("false positives" in reason for reason in score.rejection_reasons())


def test_a_model_that_finds_nothing_is_rejected(tmp_path: Path) -> None:
    gold = _tiny_gold(tmp_path)
    score = score_variant("silent", gold, {})

    assert score.coverage_any == 0.0
    assert not score.is_acceptable
    assert any("below the" in reason for reason in score.rejection_reasons())


def test_over_masking_is_measured_in_characters(tmp_path: Path) -> None:
    gold = _tiny_gold(tmp_path)
    case = gold.cases[0]
    predictions = {case.case_id: [Prediction(0, len(case.text), "PER")]}

    score = score_variant("greedy", gold, predictions)
    assert score.over_masked_chars > 0
    assert 0 < score.over_mask_ratio <= 1


def test_per_register_breakdown_separates_dialect_from_msa(tmp_path: Path) -> None:
    """An Egyptian regression must not average away behind good MSA numbers."""
    gold = _tiny_gold(tmp_path)
    msa_case = next(c for c in gold.cases if c.register is Register.MSA)
    predictions = {msa_case.case_id: [Prediction(s.start, s.end, s.label) for s in msa_case.spans]}

    score = score_variant("msa-only", gold, predictions)
    registers = dict((name, cov) for name, cov, _ in register_breakdown(score))

    assert registers["msa"] == 1.0
    assert registers["egyptian"] == 0.0


# ---------------------------------------------------------------------------
# The verdict
# ---------------------------------------------------------------------------


def _score(name: str, gold: GoldSet, fraction: float, fps: int = 0) -> object:
    """A variant covering the first `fraction` of gold spans, plus `fps` noise."""
    predictions: dict[str, list[Prediction]] = {}
    positives = [(c, s) for c in gold.positives for s in c.spans]
    take = round(fraction * len(positives))

    for case, span in positives[:take]:
        predictions.setdefault(case.case_id, []).append(
            Prediction(span.start, span.end, span.label)
        )
    for case in gold.negatives[:fps]:
        predictions.setdefault(case.case_id, []).append(Prediction(0, 3, "PER"))

    return score_variant(name, gold, predictions)


def test_the_higher_coverage_variant_wins(gold: GoldSet) -> None:
    verdict = compare([_score("low", gold, 0.85), _score("high", gold, 0.95)])

    assert verdict.winner == "high"
    assert "more gold spans" in verdict.reason


def test_coverage_beats_precision(gold: GoldSet) -> None:
    """A missed span is a leak; a false positive is an annoyance. Coverage first."""
    clean_but_blind = _score("clean-but-blind", gold, 0.85, fps=0)
    noisier_but_thorough = _score("thorough", gold, 0.97, fps=1)

    verdict = compare([clean_but_blind, noisier_but_thorough])
    assert verdict.winner == "thorough"


def test_false_positives_break_a_coverage_tie(gold: GoldSet) -> None:
    verdict = compare([_score("noisy", gold, 0.95, fps=1), _score("clean", gold, 0.95, fps=0)])

    assert verdict.winner == "clean"
    assert "tie on coverage" in verdict.reason


def test_a_variant_failing_the_gate_cannot_win(gold: GoldSet) -> None:
    verdict = compare([_score("blind", gold, 0.40), _score("ok", gold, 0.90)])
    assert verdict.winner == "ok"


def test_no_winner_when_every_variant_fails_the_gate(gold: GoldSet) -> None:
    verdict = compare([_score("a", gold, 0.30), _score("b", gold, 0.20)])

    assert verdict.winner is None
    assert "no variant met the acceptability gate" in verdict.reason


def test_comparing_nothing_is_not_a_crash() -> None:
    verdict = compare([])
    assert verdict.winner is None


def test_the_gate_thresholds_are_the_documented_ones() -> None:
    assert MIN_ACCEPTABLE_COVERAGE == 0.80
    assert MAX_ACCEPTABLE_FP_PER_SENTENCE == 0.10


def test_the_verdict_serializes_for_the_record(gold: GoldSet) -> None:
    payload = compare([_score("a", gold, 0.95), _score("b", gold, 0.85)]).as_dict()

    assert payload["winner"] == "a"
    assert len(payload["variants"]) == 2
    assert "per_register" in payload["variants"][0]
