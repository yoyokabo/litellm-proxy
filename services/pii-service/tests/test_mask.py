"""Tests for the masking splice.

Small module, but it is the last thing that touches the text before it goes to
the model, so every failure mode here is a leaked character.
"""

from __future__ import annotations

import random

import pytest

from pii_service.detect.mask import splice


def test_no_replacements_returns_the_text_unchanged() -> None:
    assert splice("hello world", []) == "hello world"


def test_single_replacement() -> None:
    assert splice("id 28503122148219 end", [(3, 17, "<ID>")]) == "id <ID> end"


def test_multiple_replacements_are_applied_right_to_left() -> None:
    """Left-to-right would invalidate later offsets as earlier ones resize."""
    text = "a 1111 b 2222 c"
    assert splice(text, [(2, 6, "<X>"), (9, 13, "<YYYYYYY>")]) == "a <X> b <YYYYYYY> c"


def test_replacements_are_sorted_before_application() -> None:
    text = "a 1111 b 2222 c"
    unsorted = [(9, 13, "<Y>"), (2, 6, "<X>")]
    assert splice(text, unsorted) == "a <X> b <Y> c"


def test_replacement_longer_than_the_span_does_not_corrupt_later_spans() -> None:
    text = "AA BB CC"
    result = splice(text, [(0, 2, "<LONGER>"), (3, 5, "<ALSO-LONGER>"), (6, 8, "<X>")])
    assert result == "<LONGER> <ALSO-LONGER> <X>"


def test_replacement_shorter_than_the_span() -> None:
    assert splice("hello world", [(0, 5, "x")]) == "x world"


def test_empty_replacement_deletes() -> None:
    assert splice("hello world", [(5, 11, "")]) == "hello"


def test_adjacent_spans_are_allowed() -> None:
    """Touching but not overlapping: end == next start is fine."""
    assert splice("abcd", [(0, 2, "<A>"), (2, 4, "<B>")]) == "<A><B>"


def test_span_at_the_start_and_end_of_the_text() -> None:
    assert splice("abc", [(0, 1, "<"), (2, 3, ">")]) == "<b>"


def test_full_text_replacement() -> None:
    assert splice("secret", [(0, 6, "<ALL>")]) == "<ALL>"


@pytest.mark.parametrize(
    "replacements",
    [
        [(0, 5, "<A>"), (3, 8, "<B>")],  # partial overlap
        [(0, 10, "<A>"), (2, 4, "<B>")],  # containment
        [(0, 5, "<A>"), (0, 5, "<B>")],  # duplicate span
    ],
)
def test_overlapping_replacements_raise(replacements: list[tuple[int, int, str]]) -> None:
    """Silently mangled output here means leaked characters, so this is loud."""
    with pytest.raises(ValueError, match="overlapping"):
        splice("0123456789abcdef", replacements)


@pytest.mark.parametrize("span", [(-1, 3), (0, 99), (5, 3)])
def test_out_of_range_replacements_raise(span: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        splice("short", [(span[0], span[1], "<X>")])


def test_arabic_text_splices_by_character_not_byte() -> None:
    text = "الرقم القومي ٢٨٥٠ نهاية"
    start = text.index("٢٨٥٠")
    result = splice(text, [(start, start + 4, "<EG_NATIONAL_ID>")])

    assert result == "الرقم القومي <EG_NATIONAL_ID> نهاية"
    assert "٢٨٥٠" not in result


def test_many_non_overlapping_replacements_hold_their_positions() -> None:
    rng = random.Random(11)
    text = "".join(rng.choice("abcdefghij") for _ in range(400))

    replacements = []
    cursor = 0
    while cursor + 6 < len(text):
        start = cursor + rng.randint(1, 5)
        end = start + rng.randint(1, 4)
        if end >= len(text):
            break
        replacements.append((start, end, f"<{len(replacements)}>"))
        cursor = end

    result = splice(text, replacements)
    for index in range(len(replacements)):
        assert f"<{index}>" in result
