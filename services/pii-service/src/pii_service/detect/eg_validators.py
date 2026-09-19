"""Structural validators for Egyptian identifiers.

Kept free of any Presidio import so the logic stays unit-testable on its own and
so ``scripts/benchmark.py`` can time it without building an analyzer.

The governing principle for every validator here (brief §4): **a wrong gate
means missed PII**. Where an authority disagrees with itself -- the national ID
check digit is the notorious case, published implementations genuinely differ --
the disagreement lowers the confidence score rather than rejecting the match.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from typing import Final

__all__ = [
    "GOVERNORATE_CODES",
    "MAX_PLAUSIBLE_AGE_YEARS",
    "NationalIdParse",
    "compute_nid_check_digit",
    "iban_is_valid",
    "national_id_score",
    "normalize_eg_mobile",
    "parse_national_id",
    "tax_id_is_plausible",
]

# Real Egyptian governorate codes. The brief specifies the accepted *range* as
# 01-35 or 88; this table is the subset actually issued. Membership is a score
# booster, never a gate -- see `national_id_score`.
GOVERNORATE_CODES: Final[dict[str, str]] = {
    "01": "Cairo",
    "02": "Alexandria",
    "03": "Port Said",
    "04": "Suez",
    "11": "Damietta",
    "12": "Dakahlia",
    "13": "Sharqia",
    "14": "Qalyubia",
    "15": "Kafr El Sheikh",
    "16": "Gharbia",
    "17": "Monufia",
    "18": "Beheira",
    "19": "Ismailia",
    "21": "Giza",
    "22": "Beni Suef",
    "23": "Fayoum",
    "24": "Minya",
    "25": "Asyut",
    "26": "Sohag",
    "27": "Qena",
    "28": "Aswan",
    "29": "Luxor",
    "31": "Red Sea",
    "32": "New Valley",
    "33": "Matrouh",
    "34": "North Sinai",
    "35": "South Sinai",
    "88": "Born abroad",
}

MAX_PLAUSIBLE_AGE_YEARS: Final = 120

# Weighted mod-11. This is the most widely deployed variant; others exist and
# disagree, which is exactly why a failure here costs 0.15 of score instead of
# discarding the match.
_NID_WEIGHTS: Final[tuple[int, ...]] = (2, 7, 6, 5, 4, 3, 2, 7, 6, 5, 4, 3, 2)

_IBAN_EG_RE: Final = re.compile(r"^EG\d{2}[A-Z0-9]{25}$")


@dataclass(frozen=True, slots=True)
class NationalIdParse:
    """A structurally valid 14-digit Egyptian national ID, decomposed."""

    digits: str
    birth_date: date
    governorate_code: str
    governorate_name: str | None
    serial: str
    check_digit: int
    checksum_valid: bool

    @property
    def governorate_known(self) -> bool:
        return self.governorate_name is not None

    @property
    def is_female(self) -> bool:
        """Parity of the 13th digit encodes gender: odd male, even female."""
        return int(self.digits[12]) % 2 == 0


def compute_nid_check_digit(first_thirteen: str) -> int | None:
    """Return the expected 14th digit, or ``None`` where the algorithm has no answer.

    A remainder of 1 leaves ``11 - 1 == 10``, which does not fit in one digit.
    Implementations in the wild handle that case inconsistently; we report
    "no answer" and let the caller fall back to the unverified score.
    """
    if len(first_thirteen) != 13 or not first_thirteen.isdigit():
        return None

    total = sum(int(d) * w for d, w in zip(first_thirteen, _NID_WEIGHTS, strict=True))
    candidate = (11 - total % 11) % 11
    return None if candidate == 10 else candidate


def parse_national_id(digits: str, *, today: date | None = None) -> NationalIdParse | None:
    """Parse a 14-digit string, returning ``None`` if the structure cannot hold.

    Structural gates, all of which are facts about the format rather than
    contested algorithms:

    * exactly 14 ASCII digits;
    * century digit is 2 (1900-1999) or 3 (2000-2099);
    * the embedded YYMMDD is a real calendar date;
    * that date is not in the future and not more than 120 years ago;
    * the governorate code is in 01-35 or 88.
    """
    if len(digits) != 14 or not digits.isdigit():
        return None

    century_digit = digits[0]
    if century_digit not in ("2", "3"):
        return None

    base_year = 1900 if century_digit == "2" else 2000
    year = base_year + int(digits[1:3])
    month = int(digits[3:5])
    day = int(digits[5:7])

    try:
        birth_date = date(year, month, day)
    except ValueError:
        return None

    reference = today or date.today()
    if birth_date > reference:
        return None
    if year < reference.year - MAX_PLAUSIBLE_AGE_YEARS:
        return None

    governorate_code = digits[7:9]
    if not _governorate_code_in_range(governorate_code):
        return None

    expected = compute_nid_check_digit(digits[:13])

    return NationalIdParse(
        digits=digits,
        birth_date=birth_date,
        governorate_code=governorate_code,
        governorate_name=GOVERNORATE_CODES.get(governorate_code),
        serial=digits[9:13],
        check_digit=int(digits[13]),
        checksum_valid=expected is not None and expected == int(digits[13]),
    )


def _governorate_code_in_range(code: str) -> bool:
    """01-35 or 88, per the brief. Wider than the issued set, deliberately."""
    if code == "88":
        return True
    return code.isdigit() and 1 <= int(code) <= 35


def national_id_score(parse: NationalIdParse) -> float:
    """Confidence for a structurally valid national ID.

    1.0 with a verified check digit, 0.85 without -- the brief's numbers. The
    checksum is a booster, not a gate, because a wrong gate means missed PII.
    """
    return 1.0 if parse.checksum_valid else 0.85


# ---------------------------------------------------------------------------
# Mobile numbers
# ---------------------------------------------------------------------------

_MOBILE_STRIP_RE: Final = re.compile(r"[\s\-().]")
_MOBILE_CORE_RE: Final = re.compile(r"^1[0125]\d{8}$")


def normalize_eg_mobile(raw: str) -> str | None:
    """Reduce an Egyptian mobile number to E.164 (``+201XXXXXXXXX``).

    Accepts the local ``01X`` form, the ``+20`` / ``0020`` / ``20`` international
    forms, and any of them written with spaces, hyphens, dots or parentheses.
    Returns ``None`` when the digits are not a valid Egyptian mobile.
    """
    compact = _MOBILE_STRIP_RE.sub("", raw)
    if not compact:
        return None

    if compact.startswith("+"):
        compact = compact[1:]
    if compact.startswith("0020"):
        compact = compact[4:]
    elif compact.startswith("20") and len(compact) == 12:
        compact = compact[2:]
    elif compact.startswith("0"):
        compact = compact[1:]

    if not compact.isdigit() or not _MOBILE_CORE_RE.match(compact):
        return None
    return f"+20{compact}"


# ---------------------------------------------------------------------------
# IBAN
# ---------------------------------------------------------------------------


def iban_is_valid(raw: str) -> bool:
    """Validate an Egyptian IBAN: ``EG`` + 2 check digits + 25 alphanumerics, mod-97.

    Unlike the national ID checksum, IBAN mod-97 is unambiguous and universally
    agreed, so here the check *is* a gate.
    """
    compact = _MOBILE_STRIP_RE.sub("", raw).upper()
    if not _IBAN_EG_RE.match(compact):
        return False

    rearranged = compact[4:] + compact[:4]
    numeric = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    return int(numeric) % 97 == 1


def iban_check_digits(country: str, bban: str) -> str:
    """Compute the two IBAN check digits for a country code and BBAN."""
    rearranged = f"{bban}{country}00"
    numeric = "".join(str(int(ch, 36)) if ch.isalpha() else ch for ch in rearranged)
    return f"{98 - int(numeric) % 97:02d}"


# ---------------------------------------------------------------------------
# Tax ID
# ---------------------------------------------------------------------------


def tax_id_is_plausible(digits: str) -> bool:
    """Egyptian tax registration numbers are 9 digits, commonly written 123-456-789.

    There is no public check digit, so this is a shape test only. It carries a
    low base score and leans on context words to clear the reporting threshold.
    """
    compact = _MOBILE_STRIP_RE.sub("", digits)
    if len(compact) != 9 or not compact.isdigit():
        return False
    # All-identical digits are placeholders in practice, not registrations.
    return len(set(compact)) > 1
