"""Correlation fingerprints for matched values.

``value_fp = HMAC-SHA256(canonical(value), pepper)[:16]`` (brief §6).

The fingerprint is what makes the admin view investigative rather than a
counter. It answers "is one person pasting a customer list, or did forty people
each paste one record" without a single digit of the value reaching the
database.

That only works if the *same* value fingerprints identically however it was
written. A national ID typed in Arabic-Indic digits and the same ID pasted in
ASCII must collide, or the investigation silently splits one person's activity
into two unrelated-looking streams. So canonicalization is entity-aware:
mobiles go through E.164, IBANs lose their spacing, names are folded. Getting
this wrong does not fail loudly -- it just makes the pivot quietly useless.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from typing import Final

from pii_service.detect.eg_validators import normalize_eg_mobile
from pii_service.detect.normalize import DIGITS, GAZETTEER, normalize

__all__ = ["FINGERPRINT_HEX_LENGTH", "canonicalize", "fingerprint"]

# 16 hex characters = 64 bits. Collisions are irrelevant at audit-table scale,
# and a short fingerprint is one less thing to stare at in a dense table.
FINGERPRINT_HEX_LENGTH: Final = 16

_WHITESPACE: Final = re.compile(r"\s+")
_NON_ALNUM: Final = re.compile(r"[^0-9a-z؀-ۿ]+")


def _digits_only(value: str) -> str:
    folded, _ = normalize(value, DIGITS)
    return "".join(ch for ch in folded if ch.isdigit())


def _canonical_mobile(value: str) -> str:
    """E.164, so 010..., +2010... and ٠١٠... all collapse to one form."""
    folded, _ = normalize(value, DIGITS)
    return normalize_eg_mobile(folded) or _digits_only(value)


def _canonical_iban(value: str) -> str:
    folded, _ = normalize(value, DIGITS)
    return _WHITESPACE.sub("", folded).upper()


def _canonical_text(value: str) -> str:
    """Fold orthography hard, so محمد and مُحَمَّد are one person.

    This over-merges slightly -- GAZETTEER folds ة to ه, so a few distinct
    names collide. For correlation that is the right side to err on: a false
    link is visible to an auditor looking at the spans, a missed link is not
    visible at all.
    """
    folded, _ = normalize(value, GAZETTEER)
    return _NON_ALNUM.sub(" ", folded.casefold()).strip()


_CANONICALIZERS: Final[dict[str, object]] = {
    "EG_NATIONAL_ID": _digits_only,
    "EG_TAX_ID": _digits_only,
    "CREDIT_CARD": _digits_only,
    "EG_MOBILE": _canonical_mobile,
    "EG_IBAN": _canonical_iban,
    "IBAN_CODE": _canonical_iban,
}


def canonicalize(entity_type: str, value: str) -> str:
    """Reduce a matched value to the form the fingerprint is computed over."""
    canonicalizer = _CANONICALIZERS.get(entity_type, _canonical_text)
    return canonicalizer(value)  # type: ignore[operator, no-any-return]


def fingerprint(entity_type: str, value: str, pepper: bytes) -> str | None:
    """Return the truncated HMAC for ``value``, or ``None`` if it canonicalizes away.

    Raises if the pepper is empty -- callers must not be able to produce an
    unkeyed fingerprint by accident, because an unkeyed one over an enumerable
    space is the same as storing the value.
    """
    if not pepper:
        raise ValueError("refusing to fingerprint with an empty pepper")

    canonical = canonicalize(entity_type, value)
    if not canonical:
        return None

    digest = hmac.new(pepper, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    return digest[:FINGERPRINT_HEX_LENGTH]
