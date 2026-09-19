"""Synthetic Egyptian PII generators for tests, fixtures and the benchmark.

Brief §12: every PII fixture in this repository is generated, never pasted. A
real national ID in a test file is a real national ID in git history, in every
developer's clone, and in every CI log that prints a failing assertion.

Shipped inside the package rather than under ``tests/`` so that
``scripts/benchmark.py`` can build realistic load without a second copy of the
logic. Nothing in the request path imports it.
"""

from __future__ import annotations

import random
from datetime import date
from typing import Final

from pii_service.detect.eg_validators import (
    GOVERNORATE_CODES,
    compute_nid_check_digit,
    iban_check_digits,
)

__all__ = [
    "ARABIC_INDIC_DIGITS",
    "synthetic_iban",
    "synthetic_mobile",
    "synthetic_national_id",
    "synthetic_tax_id",
    "to_arabic_indic",
]

ARABIC_INDIC_DIGITS: Final = "٠١٢٣٤٥٦٧٨٩"

_MOBILE_PREFIXES: Final = ("010", "011", "012", "015")


def to_arabic_indic(text: str) -> str:
    """Render every ASCII digit in ``text`` as its Arabic-Indic counterpart."""
    return text.translate(str.maketrans("0123456789", ARABIC_INDIC_DIGITS))


def synthetic_national_id(
    *,
    birth_date: date | None = None,
    governorate_code: str = "21",
    serial: int | None = None,
    valid_checksum: bool = True,
    rng: random.Random | None = None,
) -> str:
    """Build a structurally valid 14-digit Egyptian national ID.

    With ``valid_checksum=False`` the first 13 digits still form a valid
    structure but the check digit is deliberately wrong -- the 0.85-score path.
    """
    rng = rng or random.Random()
    birth_date = birth_date or date(
        rng.randint(1950, 2005), rng.randint(1, 12), rng.randint(1, 28)
    )

    if not 1900 <= birth_date.year <= 2099:
        raise ValueError("birth year must fall in 1900-2099 to be encodable")

    century_digit = "2" if birth_date.year < 2000 else "3"
    stem = (
        f"{century_digit}"
        f"{birth_date.year % 100:02d}"
        f"{birth_date.month:02d}"
        f"{birth_date.day:02d}"
        f"{governorate_code}"
    )

    start = rng.randint(0, 9999) if serial is None else serial
    for attempt in range(10_000):
        candidate_serial = (start + attempt) % 10_000
        first_thirteen = f"{stem}{candidate_serial:04d}"
        check = compute_nid_check_digit(first_thirteen)
        if check is None:
            # Remainder 1 -- this stem has no single-digit answer. Try the next.
            continue
        if not valid_checksum:
            check = (check + 1) % 10
        return f"{first_thirteen}{check}"

    raise RuntimeError("no serial produced a computable check digit")


def synthetic_mobile(
    *,
    prefix: str | None = None,
    international: bool = False,
    rng: random.Random | None = None,
) -> str:
    """Build an Egyptian mobile number, local (``01XXXXXXXXX``) or ``+20`` form."""
    rng = rng or random.Random()
    chosen = prefix or rng.choice(_MOBILE_PREFIXES)
    if chosen not in _MOBILE_PREFIXES:
        raise ValueError(f"{chosen!r} is not an Egyptian mobile prefix")

    subscriber = "".join(str(rng.randint(0, 9)) for _ in range(8))
    return f"+20{chosen[1:]}{subscriber}" if international else f"{chosen}{subscriber}"


def synthetic_iban(*, rng: random.Random | None = None) -> str:
    """Build an Egyptian IBAN (EG + 2 check digits + 25) that passes mod-97."""
    rng = rng or random.Random()
    bban = "".join(str(rng.randint(0, 9)) for _ in range(25))
    return f"EG{iban_check_digits('EG', bban)}{bban}"


def synthetic_tax_id(*, rng: random.Random | None = None) -> str:
    """Build a 9-digit Egyptian tax registration number, hyphen-grouped."""
    rng = rng or random.Random()
    while True:
        digits = "".join(str(rng.randint(0, 9)) for _ in range(9))
        if len(set(digits)) > 1:
            return f"{digits[:3]}-{digits[3:6]}-{digits[6:]}"


def known_governorate_codes() -> tuple[str, ...]:
    """The issued governorate codes, for parametrized tests."""
    return tuple(GOVERNORATE_CODES)
