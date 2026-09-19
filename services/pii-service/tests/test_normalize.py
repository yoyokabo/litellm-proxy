"""Offset-map tests for the Arabic normalizer.

These are written before the normalizer itself, per the brief (§5). The
contract they pin down:

* ``normalize(text, profile)`` returns ``(normalized, offset_map)``.
* ``len(offset_map) == len(normalized)``.
* ``offset_map[i]`` is the index, in the *original* string, of the character
  that produced ``normalized[i]``.
* The map is non-decreasing. It is strictly increasing when every
  transformation is 1:1-or-delete; it repeats an index where a single original
  character expanded into several normalized ones (ligatures).
* A normalized span ``[s, e)`` maps back to
  ``(offset_map[s], offset_map[e - 1] + 1)`` -- so a run of stripped characters
  *before* the span is excluded, and so is a run *after* it.

Every fixture here is synthetic.
"""

from __future__ import annotations

import random

import pytest

from pii_service.detect.normalize import (
    DIGITS,
    GAZETTEER,
    NAMES,
    map_span_to_original,
    normalize,
)

# --------------------------------------------------------------------------
# Character constants, spelled as escapes where the literal is ambiguous.
# --------------------------------------------------------------------------

TATWEEL = "ـ"
FATHA = "َ"
DAMMA = "ُ"
SHADDA = "ّ"
SUKUN = "ْ"
ZWNJ = "‌"
ZWJ = "‍"
LRM = "‎"
RLM = "‏"
RLE = "‫"
PDF = "‬"
ALM = "؜"

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
EXT_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"


def _assert_map_invariants(original: str, normalized: str, offset_map: list[int]) -> None:
    """Invariants that must hold for every (text, profile) pair."""
    assert len(offset_map) == len(normalized), "one map entry per normalized char"
    assert all(0 <= i < len(original) for i in offset_map), "indices address the original"
    assert offset_map == sorted(offset_map), "map is non-decreasing"


# --------------------------------------------------------------------------
# Case 1 -- Arabic-Indic digits mixed with ASCII
# --------------------------------------------------------------------------


def test_arabic_indic_digits_fold_to_ascii() -> None:
    normalized, offset_map = normalize(ARABIC_INDIC, DIGITS)
    assert normalized == "0123456789"
    assert offset_map == list(range(10))


def test_extended_arabic_indic_digits_fold_to_ascii() -> None:
    normalized, offset_map = normalize(EXT_ARABIC_INDIC, DIGITS)
    assert normalized == "0123456789"
    assert offset_map == list(range(10))


def test_arabic_indic_digits_mixed_with_ascii_keeps_offsets_identity() -> None:
    # Digit folding is 1:1, so with nothing stripped the map is the identity.
    original = "ID ٢٨٥٠ and 1234 done"
    normalized, offset_map = normalize(original, DIGITS)

    assert normalized == "ID 2850 and 1234 done"
    assert offset_map == list(range(len(original)))
    _assert_map_invariants(original, normalized, offset_map)


def test_mixed_digit_span_maps_back_to_the_arabic_original() -> None:
    original = "الرقم ٢٨٥٠ fin"
    normalized, offset_map = normalize(original, DIGITS)

    start = normalized.index("2850")
    span = map_span_to_original(offset_map, start, start + 4, len(original))

    assert original[span[0] : span[1]] == "٢٨٥٠"


def test_digits_separated_by_zero_width_joiner_become_contiguous() -> None:
    # A zero-width char wedged into a number must not break a numeric regex,
    # and the recovered span must still cover the whole original run.
    original = f"٢٨{ZWNJ}٥٠"
    normalized, offset_map = normalize(original, DIGITS)

    assert normalized == "2850"
    span = map_span_to_original(offset_map, 0, 4, len(original))
    assert original[span[0] : span[1]] == original  # includes the ZWNJ
    _assert_map_invariants(original, normalized, offset_map)


def test_fullwidth_digits_fold_to_ascii() -> None:
    original = "２８５０"
    normalized, offset_map = normalize(original, DIGITS)
    assert normalized == "2850"
    _assert_map_invariants(original, normalized, offset_map)


# --------------------------------------------------------------------------
# Case 2 -- diacritics inside a matched name
# --------------------------------------------------------------------------


def test_diacritics_are_stripped_for_name_matching() -> None:
    original = f"م{DAMMA}ح{FATHA}م{FATHA}{SHADDA}د"
    normalized, offset_map = normalize(original, NAMES)

    assert normalized == "محمد"
    _assert_map_invariants(original, normalized, offset_map)


def test_span_over_a_diacritised_name_recovers_the_full_original_run() -> None:
    name_with_marks = f"م{DAMMA}ح{FATHA}م{FATHA}{SHADDA}د"
    original = f"{name_with_marks} علي"
    normalized, offset_map = normalize(original, NAMES)

    assert normalized == "محمد علي"

    span = map_span_to_original(offset_map, 0, 4, len(original))
    assert original[span[0] : span[1]] == name_with_marks


def test_digits_profile_preserves_diacritics() -> None:
    # Stripping marks is a name-matching concern; the numeric pass must not
    # silently change what a downstream name recognizer sees.
    original = f"م{DAMMA}حمد"
    normalized, _ = normalize(original, DIGITS)
    assert normalized == original


# --------------------------------------------------------------------------
# Case 3 -- RTL/LTR mixed sentences
# --------------------------------------------------------------------------


def test_bidi_controls_are_stripped_in_every_profile() -> None:
    original = f"Call {RLE}محمد{PDF} on {ALM}٠١٠"
    for profile in (DIGITS, NAMES, GAZETTEER):
        normalized, offset_map = normalize(original, profile)
        assert RLE not in normalized
        assert PDF not in normalized
        assert ALM not in normalized
        _assert_map_invariants(original, normalized, offset_map)


def test_rtl_ltr_mixed_sentence_maps_number_span_back_correctly() -> None:
    arabic_number = "٠١٠١٢٣٤٥٦٧٨"
    original = f"{LRM}Reach محمد{RLM} at {arabic_number} today"
    normalized, offset_map = normalize(original, DIGITS)

    assert "01012345678" in normalized
    start = normalized.index("01012345678")
    span = map_span_to_original(offset_map, start, start + 11, len(original))

    assert original[span[0] : span[1]] == arabic_number


def test_latin_text_in_digits_profile_is_untouched() -> None:
    original = "Contact me at +20 100 123 4567"
    normalized, offset_map = normalize(original, DIGITS)
    assert normalized == original
    assert offset_map == list(range(len(original)))


# --------------------------------------------------------------------------
# Case 4 -- a span that starts inside / abuts a stripped character run
# --------------------------------------------------------------------------


def test_span_preceded_by_a_stripped_tatweel_run() -> None:
    original = f"{TATWEEL * 3}محمد"
    normalized, offset_map = normalize(original, NAMES)

    assert normalized == "محمد"
    assert offset_map == [3, 4, 5, 6]

    span = map_span_to_original(offset_map, 0, 4, len(original))
    # The span starts *after* the stripped run -- the tatweels are not swallowed.
    assert span == (3, 7)
    assert original[span[0] : span[1]] == "محمد"


def test_trailing_stripped_run_is_not_swallowed_by_the_span() -> None:
    original = f"محمد{TATWEEL * 3} علي"
    normalized, offset_map = normalize(original, NAMES)

    span = map_span_to_original(offset_map, 0, 4, len(original))
    assert span == (0, 4)
    assert original[span[0] : span[1]] == "محمد"


def test_stripped_run_in_the_middle_of_a_span_is_included() -> None:
    original = f"مح{TATWEEL * 4}مد"
    normalized, offset_map = normalize(original, NAMES)

    assert normalized == "محمد"
    span = map_span_to_original(offset_map, 0, 4, len(original))
    assert original[span[0] : span[1]] == original


def test_span_ending_at_end_of_string_with_trailing_stripped_chars() -> None:
    original = f"محمد{TATWEEL}{ZWJ}"
    normalized, offset_map = normalize(original, NAMES)

    span = map_span_to_original(offset_map, 0, len(normalized), len(original))
    assert span == (0, 4)


def test_text_that_normalizes_to_empty() -> None:
    original = TATWEEL * 5
    normalized, offset_map = normalize(original, NAMES)
    assert normalized == ""
    assert offset_map == []


def test_map_span_on_empty_normalization_is_rejected() -> None:
    with pytest.raises(ValueError):
        map_span_to_original([], 0, 0, 5)


# --------------------------------------------------------------------------
# Gazetteer folding
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("original", "expected"),
    [
        ("أحمد", "احمد"),  # alef hamza above
        ("إسلام", "اسلام"),  # hamza below
        ("آمال", "امال"),  # alef maddah
        ("قاهرة", "قاهره"),  # ta marbuta
        ("مصطفى", "مصطفي"),  # alef maqsura
    ],
)
def test_gazetteer_letter_folding(original: str, expected: str) -> None:
    normalized, offset_map = normalize(original, GAZETTEER)
    assert normalized == expected
    assert offset_map == list(range(len(original)))


def test_names_profile_does_not_fold_letters() -> None:
    # Letter folding is lossy for display; it belongs to gazetteer lookup only.
    original = "أحمد"
    normalized, _ = normalize(original, NAMES)
    assert normalized == original


# --------------------------------------------------------------------------
# Invariants under randomized input
# --------------------------------------------------------------------------


def test_invariants_hold_over_randomized_mixed_script_input() -> None:
    alphabet = (
        list(ARABIC_INDIC)
        + list(EXT_ARABIC_INDIC)
        + list("0123456789 abcXYZ+-")
        + ["م", "ح", "د", "أ", "ة", "ى"]
        + [TATWEEL, FATHA, DAMMA, SHADDA, SUKUN, ZWNJ, ZWJ, LRM, RLM, RLE, PDF, ALM]
        + ["２", "ﻻ"]  # fullwidth digit, lam-alef ligature (1 -> 2 chars)
    )
    rng = random.Random(20240917)

    for _ in range(600):
        original = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 40)))
        for profile in (DIGITS, NAMES, GAZETTEER):
            normalized, offset_map = normalize(original, profile)
            _assert_map_invariants(original, normalized, offset_map)

            if normalized:
                # Every single-character span must map back inside the original.
                for i in range(len(normalized)):
                    start, end = map_span_to_original(offset_map, i, i + 1, len(original))
                    assert 0 <= start < end <= len(original)


def test_ligature_expansion_repeats_the_source_index() -> None:
    # U+FEFB LAM WITH ALEF expands to two characters under NFKC; both must point
    # at the single original index, keeping the map non-decreasing.
    original = "ﻻ"
    normalized, offset_map = normalize(original, NAMES)

    assert len(normalized) == 2
    assert offset_map == [0, 0]

    span = map_span_to_original(offset_map, 0, 2, len(original))
    assert span == (0, 1)
