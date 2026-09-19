"""Tests for the Egyptian identifier validators.

Every identifier here comes out of ``pii_service.synthetic``. Nothing is pasted.
"""

from __future__ import annotations

import random
from datetime import date, timedelta

import pytest

from pii_service.detect.eg_validators import (
    GOVERNORATE_CODES,
    compute_nid_check_digit,
    iban_is_valid,
    national_id_score,
    normalize_eg_mobile,
    parse_national_id,
    tax_id_is_plausible,
)
from pii_service.synthetic import (
    synthetic_iban,
    synthetic_mobile,
    synthetic_national_id,
    synthetic_tax_id,
)

TODAY = date(2026, 9, 19)


# ---------------------------------------------------------------------------
# National ID -- structure
# ---------------------------------------------------------------------------


def test_generated_id_parses_and_scores_one() -> None:
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=4821)
    parse = parse_national_id(nid, today=TODAY)

    assert parse is not None
    assert parse.birth_date == date(1985, 3, 12)
    assert parse.governorate_code == "21"
    assert parse.governorate_name == "Giza"
    assert parse.checksum_valid is True
    assert national_id_score(parse) == 1.0


def test_wrong_check_digit_still_parses_but_scores_zero_eight_five() -> None:
    nid = synthetic_national_id(
        birth_date=date(1985, 3, 12),
        governorate_code="21",
        serial=4821,
        valid_checksum=False,
    )
    parse = parse_national_id(nid, today=TODAY)

    # The brief is explicit: the checksum is a booster, never a gate. A bad
    # check digit must not make the ID vanish from the findings.
    assert parse is not None
    assert parse.checksum_valid is False
    assert national_id_score(parse) == 0.85


@pytest.mark.parametrize("bad", ["", "123", "2" * 13, "2" * 15, "abcdefghijklmn", "2850123456789 "])
def test_non_fourteen_digit_input_is_rejected(bad: str) -> None:
    assert parse_national_id(bad, today=TODAY) is None


@pytest.mark.parametrize("century", ["0", "1", "4", "5", "9"])
def test_century_digit_must_be_two_or_three(century: str) -> None:
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=1)
    assert parse_national_id(century + nid[1:], today=TODAY) is None


@pytest.mark.parametrize(
    "yymmdd",
    [
        "851332",  # month 13
        "850012",  # month 00
        "850332",  # day 32
        "850300",  # day 00
        "850229",  # 1985 is not a leap year
    ],
)
def test_implausible_birth_dates_are_rejected(yymmdd: str) -> None:
    nid = synthetic_national_id(birth_date=date(1985, 3, 12), governorate_code="21", serial=1)
    assert parse_national_id("2" + yymmdd + nid[7:], today=TODAY) is None


def test_leap_day_is_accepted_in_a_leap_year() -> None:
    nid = synthetic_national_id(birth_date=date(1984, 2, 29), governorate_code="21", serial=77)
    parse = parse_national_id(nid, today=TODAY)
    assert parse is not None
    assert parse.birth_date == date(1984, 2, 29)


def test_future_birth_date_is_rejected() -> None:
    tomorrow = TODAY + timedelta(days=1)
    nid = synthetic_national_id(birth_date=tomorrow, governorate_code="21", serial=5)
    assert parse_national_id(nid, today=TODAY) is None


def test_birth_date_today_is_accepted() -> None:
    nid = synthetic_national_id(birth_date=TODAY, governorate_code="21", serial=5)
    assert parse_national_id(nid, today=TODAY) is not None


def test_implausibly_old_birth_date_is_rejected() -> None:
    # Century digit 2 means 19xx, so "01" reads as 1901 -- over 120 years before
    # the reference date.
    nid = synthetic_national_id(birth_date=date(1901, 5, 5), governorate_code="21", serial=5)
    assert parse_national_id(nid, today=TODAY) is None


@pytest.mark.parametrize("code", sorted(GOVERNORATE_CODES))
def test_every_issued_governorate_code_parses_and_is_named(code: str) -> None:
    nid = synthetic_national_id(birth_date=date(1990, 6, 6), governorate_code=code, serial=101)
    parse = parse_national_id(nid, today=TODAY)

    assert parse is not None
    assert parse.governorate_known is True
    assert parse.governorate_name == GOVERNORATE_CODES[code]


@pytest.mark.parametrize("code", ["36", "50", "87", "99", "00"])
def test_out_of_range_governorate_codes_are_rejected(code: str) -> None:
    nid = synthetic_national_id(birth_date=date(1990, 6, 6), governorate_code="21", serial=101)
    assert parse_national_id(nid[:7] + code + nid[9:], today=TODAY) is None


@pytest.mark.parametrize("code", ["05", "06", "07", "08", "09", "10", "20", "30"])
def test_unissued_but_in_range_codes_still_parse(code: str) -> None:
    # Coverage beats precision here: an unissued code inside 01-35 is still
    # reported, just without a governorate name.
    nid = synthetic_national_id(birth_date=date(1990, 6, 6), governorate_code=code, serial=101)
    parse = parse_national_id(nid, today=TODAY)

    assert parse is not None
    assert parse.governorate_known is False


def test_gender_parity_reads_the_thirteenth_digit() -> None:
    nid = synthetic_national_id(birth_date=date(1990, 6, 6), governorate_code="21", serial=1234)
    parse = parse_national_id(nid, today=TODAY)
    assert parse is not None
    assert parse.is_female is (int(nid[12]) % 2 == 0)


# ---------------------------------------------------------------------------
# National ID -- check digit
# ---------------------------------------------------------------------------


def test_check_digit_rejects_malformed_input() -> None:
    assert compute_nid_check_digit("123") is None
    assert compute_nid_check_digit("a" * 13) is None


def test_check_digit_is_a_single_digit_or_none() -> None:
    rng = random.Random(4242)
    for _ in range(2000):
        stem = "".join(str(rng.randint(0, 9)) for _ in range(13))
        check = compute_nid_check_digit(stem)
        assert check is None or 0 <= check <= 9


def test_generated_ids_round_trip_through_the_checksum() -> None:
    rng = random.Random(99)
    for _ in range(500):
        nid = synthetic_national_id(rng=rng)
        assert compute_nid_check_digit(nid[:13]) == int(nid[13])


def test_random_generated_ids_all_parse() -> None:
    rng = random.Random(7)
    for _ in range(500):
        nid = synthetic_national_id(rng=rng)
        parse = parse_national_id(nid, today=TODAY)
        assert parse is not None, nid
        assert national_id_score(parse) == 1.0


# ---------------------------------------------------------------------------
# Mobile numbers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("prefix", ["010", "011", "012", "015"])
def test_local_mobile_normalizes_to_e164(prefix: str) -> None:
    number = synthetic_mobile(prefix=prefix, rng=random.Random(1))
    normalized = normalize_eg_mobile(number)
    assert normalized is not None
    assert normalized.startswith(f"+20{prefix[1:]}")
    assert len(normalized) == 13


def test_international_and_local_forms_agree() -> None:
    rng = random.Random(11)
    local = synthetic_mobile(prefix="010", rng=rng)
    as_international = f"+20{local[1:]}"
    assert normalize_eg_mobile(local) == normalize_eg_mobile(as_international)


@pytest.mark.parametrize(
    "template",
    ["{n}", "+20{s}", "0020{s}", "20{s}", "{a} {b} {c}", "{a}-{b}-{c}", "({a}) {b}.{c}"],
)
def test_separator_and_prefix_variants_all_normalize(template: str) -> None:
    number = synthetic_mobile(prefix="012", rng=random.Random(3))
    rendered = template.format(
        n=number, s=number[1:], a=number[:3], b=number[3:7], c=number[7:]
    )
    assert normalize_eg_mobile(rendered) == f"+20{number[1:]}"


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "0131234567",  # 013 is not an Egyptian mobile prefix
        "0141234567",
        "010123456",  # too short
        "0101234567890",  # too long
        "not a phone",
        "+1 415 555 0100",
    ],
)
def test_non_egyptian_mobiles_are_rejected(bad: str) -> None:
    assert normalize_eg_mobile(bad) is None


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------


def test_generated_iban_validates() -> None:
    rng = random.Random(5)
    for _ in range(200):
        assert iban_is_valid(synthetic_iban(rng=rng))


def test_iban_tolerates_spacing() -> None:
    iban = synthetic_iban(rng=random.Random(8))
    spaced = " ".join(iban[i : i + 4] for i in range(0, len(iban), 4))
    assert iban_is_valid(spaced)


def test_iban_with_a_corrupted_digit_fails_mod97() -> None:
    iban = synthetic_iban(rng=random.Random(12))
    corrupted = iban[:10] + str((int(iban[10]) + 1) % 10) + iban[11:]
    assert iban_is_valid(corrupted) is False


@pytest.mark.parametrize("bad", ["", "EG12", "GB82WEST12345698765432", "EG" + "1" * 26])
def test_malformed_ibans_are_rejected(bad: str) -> None:
    assert iban_is_valid(bad) is False


# ---------------------------------------------------------------------------
# Tax ID
# ---------------------------------------------------------------------------


def test_generated_tax_id_is_plausible() -> None:
    rng = random.Random(6)
    for _ in range(100):
        assert tax_id_is_plausible(synthetic_tax_id(rng=rng))


@pytest.mark.parametrize("bad", ["", "12345678", "1234567890", "111111111", "abcdefghi"])
def test_implausible_tax_ids_are_rejected(bad: str) -> None:
    assert tax_id_is_plausible(bad) is False
