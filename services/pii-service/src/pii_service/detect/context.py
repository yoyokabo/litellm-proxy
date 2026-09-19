"""Context-word boosting for tier-1 recognizers.

Presidio ships ``LemmaContextAwareEnhancer``, which we deliberately do not use.
It matches context words against ``token.lemma_``, and our pipeline has no
lemmatizer (see ``detect/nlp_engine.py``) -- nor would one help, because the
terms that matter here are Arabic: الرقم القومي, بطاقة, موبايل. An English
lemmatizer has nothing to say about those, and a blank pipeline leaves
``lemma_`` empty, so the stock enhancer would silently boost nothing at all.

Silently boosting nothing is the dangerous failure: a tax ID scored 0.3 that
was supposed to reach 0.65 with context never clears its threshold, and the PII
goes through unmasked with no error anywhere. So context matching is explicit
and tested here instead.

Matching runs against GAZETTEER-normalized text, so orthographic variation in
the term (أ vs ا, ة vs ه, diacritics, tatweel) does not cause a miss.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final

from pii_service.detect.normalize import GAZETTEER, normalize

__all__ = [
    "DEFAULT_CONTEXT_BOOST",
    "DEFAULT_CONTEXT_WINDOW",
    "find_context_term",
    "normalize_context_terms",
]

# Presidio's own default boost, kept for consistency with anything that does
# fall through to a stock recognizer.
DEFAULT_CONTEXT_BOOST: Final = 0.35

# Characters either side of the match to search. An Arabic label sits directly
# before its value ("الرقم القومي: 285..."), and a trailing label is common in
# tables, so the window is symmetric.
DEFAULT_CONTEXT_WINDOW: Final = 64


def _fold(text: str) -> str:
    """GAZETTEER-normalize and casefold. Offsets are irrelevant here -- only presence is."""
    normalized, _ = normalize(text, GAZETTEER)
    return normalized.casefold()


def normalize_context_terms(terms: Iterable[str]) -> frozenset[str]:
    """Fold a set of context terms once, at load time, for repeated matching."""
    return frozenset(folded for term in terms if (folded := _fold(term)))


def find_context_term(
    text: str,
    start: int,
    end: int,
    terms: frozenset[str],
    window: int = DEFAULT_CONTEXT_WINDOW,
) -> str | None:
    """Return the first context term found near ``text[start:end]``, or ``None``.

    ``start`` and ``end`` are offsets into ``text`` as the recognizer received
    it. The window either side is folded before matching, so the terms must
    already be folded -- pass the output of ``normalize_context_terms``.
    """
    if not terms:
        return None

    before = _fold(text[max(0, start - window) : start])
    after = _fold(text[end : end + window])
    haystack = f"{before} \x00 {after}"

    # Deterministic order, so an explanation naming the term is reproducible.
    for term in sorted(terms):
        if term in haystack:
            return term
    return None
