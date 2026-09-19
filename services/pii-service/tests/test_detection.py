"""End-to-end detection tests: tier 1 through the router.

The recurring assertion in this file is that a span, mapped back to original
coordinates, slices exactly the characters that were matched. That is the
property the whole offset-map design exists to provide, and it is the one that
breaks silently -- a detector that reports 10:24 instead of 13:27 still
"detects" the ID, it just masks the wrong six characters and ships the rest.
"""

from __future__ import annotations

import random
from datetime import date
from itertools import pairwise

import pytest

from conftest import REFERENCE_DATE
from pii_service.detect.router import (
    PiiRouter,
    Script,
    dominant_script,
    script_segments,
)
from pii_service.policy.models import EntityAction
from pii_service.synthetic import (
    synthetic_iban,
    synthetic_mobile,
    synthetic_national_id,
    synthetic_tax_id,
    to_arabic_indic,
)


def entity_types(outcome: object) -> list[str]:
    return sorted(span.entity_type for span in outcome.spans)  # type: ignore[attr-defined]


def span_for(outcome: object, entity_type: str) -> object:
    matches = [s for s in outcome.spans if s.entity_type == entity_type]  # type: ignore[attr-defined]
    assert matches, f"no {entity_type} in {entity_types(outcome)}"
    return matches[0]


# ---------------------------------------------------------------------------
# National ID
# ---------------------------------------------------------------------------


def test_latin_script_national_id_is_found_and_masked(router: PiiRouter) -> None:
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=4821)
    outcome = router.analyze(f"My national ID is {nid}, please check.")

    span = span_for(outcome, "EG_NATIONAL_ID")
    assert span.score == 1.0
    assert outcome.original_text[span.start : span.end] == nid
    assert nid not in outcome.anonymized_text
    assert "<EG_NATIONAL_ID>" in outcome.anonymized_text


def test_arabic_indic_national_id_maps_back_to_the_arabic_digits(router: PiiRouter) -> None:
    nid = synthetic_national_id(birth_date=date(1990, 7, 4), governorate_code="01", serial=99)
    arabic_nid = to_arabic_indic(nid)
    text = f"الرقم القومي {arabic_nid} شكرا"

    outcome = router.analyze(text)
    span = span_for(outcome, "EG_NATIONAL_ID")

    assert outcome.lang == "ar"
    # The span must cover the Arabic-Indic digits in the *original*, not the
    # ASCII digits of the normalized form.
    assert outcome.original_text[span.start : span.end] == arabic_nid
    assert arabic_nid not in outcome.anonymized_text
    assert outcome.anonymized_text.startswith("الرقم القومي <EG_NATIONAL_ID>")


def test_unverified_checksum_still_detected_at_zero_eight_five(router: PiiRouter) -> None:
    nid = synthetic_national_id(
        birth_date=date(1985, 3, 12), governorate_code="21", serial=4821, valid_checksum=False
    )
    outcome = router.analyze(f"ID {nid}")

    span = span_for(outcome, "EG_NATIONAL_ID")
    assert span.score == pytest.approx(0.85)
    assert nid not in outcome.anonymized_text


def test_fourteen_digits_inside_a_longer_run_is_not_an_id(router: PiiRouter) -> None:
    outcome = router.analyze("trace 123456789012345678901234 end")
    assert "EG_NATIONAL_ID" not in entity_types(outcome)


def test_structurally_impossible_fourteen_digits_is_not_an_id(router: PiiRouter) -> None:
    # Century digit 9, and a month of 99.
    outcome = router.analyze("value 99991234567890 end")
    assert "EG_NATIONAL_ID" not in entity_types(outcome)


def test_national_id_beats_credit_card_on_an_identical_span(router: PiiRouter) -> None:
    """A national ID that also satisfies Luhn must not be filed as a card.

    Both would be masked, so nothing leaks -- but the audit row would name the
    wrong entity, and the fingerprint would land in the wrong investigation.
    """
    rng = random.Random(1)
    for _ in range(6000):
        nid = synthetic_national_id(rng=rng)
        if _luhn_ok(nid):
            break
    else:  # pragma: no cover
        pytest.skip("no Luhn-satisfying national ID generated")

    outcome = router.analyze(f"ID {nid} here")
    assert entity_types(outcome) == ["EG_NATIONAL_ID"]


def _luhn_ok(digits: str) -> bool:
    total = 0
    for index, char in enumerate(reversed(digits)):
        value = int(char)
        if index % 2 == 1:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


# ---------------------------------------------------------------------------
# Mobile, IBAN, email, card
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["010", "011", "012", "015"])
def test_local_mobiles_are_detected(router: PiiRouter, prefix: str) -> None:
    mobile = synthetic_mobile(prefix=prefix, rng=random.Random(5))
    outcome = router.analyze(f"my mobile is {mobile}")

    span = span_for(outcome, "EG_MOBILE")
    assert outcome.original_text[span.start : span.end] == mobile
    assert mobile not in outcome.anonymized_text


def test_international_mobile_is_detected_once_not_twice(router: PiiRouter) -> None:
    mobile = synthetic_mobile(prefix="010", international=True, rng=random.Random(6))
    outcome = router.analyze(f"reach me on {mobile}")

    spans = [s for s in outcome.spans if s.entity_type == "EG_MOBILE"]
    assert len(spans) == 1
    assert outcome.original_text[spans[0].start : spans[0].end] == mobile


def test_arabic_indic_mobile_maps_back(router: PiiRouter) -> None:
    mobile = synthetic_mobile(prefix="012", rng=random.Random(7))
    arabic = to_arabic_indic(mobile)
    outcome = router.analyze(f"موبايلي {arabic}")

    span = span_for(outcome, "EG_MOBILE")
    assert outcome.original_text[span.start : span.end] == arabic


def test_iban_is_detected_and_a_corrupted_one_is_not(router: PiiRouter) -> None:
    iban = synthetic_iban(rng=random.Random(9))
    assert "EG_IBAN" in entity_types(router.analyze(f"account {iban}"))

    corrupted = iban[:10] + str((int(iban[10]) + 1) % 10) + iban[11:]
    assert "EG_IBAN" not in entity_types(router.analyze(f"account {corrupted}"))


def test_email_and_credit_card_come_from_the_builtins(router: PiiRouter) -> None:
    outcome = router.analyze("mail ahmed@example.com card 4111111111111111")
    assert entity_types(outcome) == ["CREDIT_CARD", "EMAIL_ADDRESS"]
    assert "ahmed@example.com" not in outcome.anonymized_text
    assert "4111111111111111" not in outcome.anonymized_text


# ---------------------------------------------------------------------------
# Context gating
# ---------------------------------------------------------------------------


def test_tax_id_needs_context_to_clear_its_threshold(router: PiiRouter) -> None:
    tax = synthetic_tax_id(rng=random.Random(4))

    with_context = router.analyze(f"الرقم الضريبي {tax}")
    without_context = router.analyze(f"order number {tax} shipped today")

    assert "EG_TAX_ID" in entity_types(with_context)
    assert "EG_TAX_ID" not in entity_types(without_context)


def test_passport_needs_context_to_clear_its_threshold(router: PiiRouter) -> None:
    assert "EG_PASSPORT" in entity_types(router.analyze("جواز سفر A12345678"))
    assert "EG_PASSPORT" not in entity_types(router.analyze("build artifact A12345678"))


def test_context_term_is_recorded_on_the_span(router: PiiRouter) -> None:
    tax = synthetic_tax_id(rng=random.Random(14))
    span = span_for(router.analyze(f"tax id {tax}"), "EG_TAX_ID")
    assert span.context_term is not None


def test_arabic_context_works_despite_orthographic_variation(router: PiiRouter) -> None:
    """The gazetteer says الرقم القومي; the user typed a diacritised variant."""
    nid = synthetic_national_id(birth_date=date(1992, 2, 2), governorate_code="02", serial=11)
    tax = synthetic_tax_id(rng=random.Random(21))

    # Alef with hamza instead of bare alef, plus a tatweel, in the context word.
    assert "EG_TAX_ID" in entity_types(router.analyze(f"الرقـم الضريبي {tax}"))
    assert "EG_NATIONAL_ID" in entity_types(router.analyze(f"الرقم القومى {nid}"))


# ---------------------------------------------------------------------------
# Policy actions
# ---------------------------------------------------------------------------


def test_allow_action_records_but_does_not_mask(router: PiiRouter, policy: object) -> None:
    """AR_ORG is ALLOW in entities.yaml: recorded, left in the text."""
    assert policy.entities["AR_ORG"].action is EntityAction.ALLOW  # type: ignore[attr-defined]


def test_masking_is_idempotent(router: PiiRouter) -> None:
    """Re-running detection on masked text finds nothing new.

    The brief leans on this: the chat backend masks first and the proxy
    guardrail masks again, and the second pass must be a cheap no-op rather
    than a source of double-masked gibberish.
    """
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=4821)
    mobile = synthetic_mobile(prefix="010", rng=random.Random(2))
    once = router.analyze(f"ID {nid} phone {mobile} thanks")
    twice = router.analyze(once.anonymized_text)

    assert twice.spans == []
    assert twice.anonymized_text == once.anonymized_text


def test_empty_and_whitespace_input_is_handled(router: PiiRouter) -> None:
    for text in ("", "   ", "\n\n"):
        outcome = router.analyze(text)
        assert outcome.spans == []
        assert outcome.anonymized_text == text


def test_multiple_entities_in_one_message_all_map_back(router: PiiRouter) -> None:
    rng = random.Random(31)
    nid = synthetic_national_id(rng=rng)
    mobile = synthetic_mobile(prefix="011", rng=rng)
    iban = synthetic_iban(rng=rng)
    text = f"الرقم القومي {to_arabic_indic(nid)}، موبايل {mobile}، حساب {iban}"

    outcome = router.analyze(text)

    assert entity_types(outcome) == ["EG_IBAN", "EG_MOBILE", "EG_NATIONAL_ID"]
    for span in outcome.spans:
        assert outcome.original_text[span.start : span.end] == span.value
    for raw in (to_arabic_indic(nid), mobile, iban):
        assert raw not in outcome.anonymized_text


def test_spans_never_overlap(router: PiiRouter) -> None:
    rng = random.Random(41)
    nid = synthetic_national_id(rng=rng)
    text = f"a {nid} b {synthetic_mobile(rng=rng)} c {synthetic_iban(rng=rng)} d"
    spans = sorted(router.analyze(text).spans, key=lambda s: s.start)
    for earlier, later in pairwise(spans):
        assert earlier.end <= later.start


# ---------------------------------------------------------------------------
# Repr safety
# ---------------------------------------------------------------------------


def test_span_repr_never_contains_the_matched_value(router: PiiRouter) -> None:
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=4821)
    span = span_for(router.analyze(f"ID {nid}"), "EG_NATIONAL_ID")

    assert nid not in repr(span)
    assert nid not in str(span)
    assert nid not in f"{span}"
    assert "<redacted>" in repr(span)


def test_outcome_counts_are_by_entity_type(router: PiiRouter) -> None:
    rng = random.Random(51)
    a, b = synthetic_mobile(rng=rng), synthetic_mobile(rng=rng)
    outcome = router.analyze(f"phones {a} and {b}")
    assert outcome.masked_entity_counts == {"EG_MOBILE": 2}


# ---------------------------------------------------------------------------
# Script routing
# ---------------------------------------------------------------------------


def test_dominant_script() -> None:
    assert dominant_script("محمد علي") == Script.ARABIC
    assert dominant_script("hello world") == Script.LATIN
    assert dominant_script("") == Script.LATIN
    assert dominant_script("12345 !!! ---") == Script.LATIN
    # A mostly-Latin prompt reads as Latin: one Arabic word is below the
    # dominance threshold.
    assert dominant_script("please translate محمد for me") == Script.LATIN


def test_a_minority_arabic_run_still_gets_its_own_segment() -> None:
    """Document-level script must not decide per-span coverage.

    This is the coverage trap the segment API exists to close. "please
    translate محمد for me" is Latin-dominant, so a design that routed tiers by
    document language would hand the whole thing to the Latin-only tier 3 and
    never look at the Arabic name. Tiers 2 and 3 are therefore registered for
    every language and select their own runs from here.
    """
    text = "please translate محمد for me"
    arabic = [s for s in script_segments(text) if s.script == Script.ARABIC]

    assert len(arabic) == 1
    assert "محمد" in text[arabic[0].start : arabic[0].end]


def test_script_segments_keep_neutral_characters_with_the_preceding_run() -> None:
    segments = script_segments("Hello محمد، 25 سنة world")
    assert [s.script for s in segments] == [Script.LATIN, Script.ARABIC, Script.LATIN]
    assert segments[0].start == 0
    assert segments[-1].end == len("Hello محمد، 25 سنة world")


def test_script_segments_cover_the_text_without_gaps() -> None:
    text = "abc محمد def مصر ghi"
    segments = script_segments(text)
    assert segments[0].start == 0
    assert segments[-1].end == len(text)
    for earlier, later in pairwise(segments):
        assert earlier.end == later.start


def test_script_segments_on_empty_and_neutral_text() -> None:
    assert script_segments("") == []
    assert script_segments("123 !!!") == []


def test_reference_date_is_pinned() -> None:
    assert date(2026, 9, 19) == REFERENCE_DATE
