"""Arabic text normalization with an offset map back to the original string.

The single rule this module exists to enforce (brief §5): **findings are always
reported against original offsets**. A normalizer that returns a bare string is
a bug, because the caller then has no way to splice the original text when
anonymizing, and every reported span silently drifts by the number of
characters that were folded away.

So ``normalize`` returns ``(normalized, offset_map)`` where ``offset_map[i]`` is
the index in the *original* string of the character that produced
``normalized[i]``.

Three profiles, because the passes need different amounts of folding:

``DIGITS``
    What runs before any numeric regex. Folds Arabic-Indic, extended
    Arabic-Indic and (via NFKC) full-width digits to ASCII, and drops
    zero-width and bidi control characters, which are invisible but will
    happily sit in the middle of a national ID and break a ``\\d{14}`` match.
    Deliberately leaves diacritics and tatweel alone.

``NAMES``
    ``DIGITS`` plus tatweel and Arabic diacritic stripping, for name matching.

``GAZETTEER``
    ``NAMES`` plus lossy letter folding (alef variants, ta marbuta, alef
    maqsura, Farsi yeh/keheh) for gazetteer lookup only.

Transformations are 1:1, 1:0 (dropped) or 1:many (a ligature such as U+FEFB
expanding to two characters). The map is therefore non-decreasing rather than
strictly increasing; ``map_span_to_original`` handles all three cases.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Final, NamedTuple

__all__ = [
    "DIGITS",
    "GAZETTEER",
    "NAMES",
    "NormalizationProfile",
    "NormalizedText",
    "map_span_to_original",
    "normalize",
]

TATWEEL: Final = "ـ"

# Harakat, tanween, and the Quranic annotation marks. Explicit ranges rather
# than a `unicodedata.category(c) == "Mn"` test, so that Latin combining
# accents (José) survive untouched -- stripping those is not our business and
# would change what the Latin-script tier sees.
_DIACRITIC_RANGES: Final[tuple[tuple[int, int], ...]] = (
    (0x064B, 0x065F),  # fathatan .. wavy hamza below
    (0x0670, 0x0670),  # superscript alef
    (0x06D6, 0x06DC),  # small high Quranic marks
    (0x06DF, 0x06E8),  # small high/low marks
    (0x06EA, 0x06ED),  # empty centre low stop .. small low meem
)

ARABIC_DIACRITICS: Final[frozenset[str]] = frozenset(
    chr(cp) for lo, hi in _DIACRITIC_RANGES for cp in range(lo, hi + 1)
)

# Zero-width and bidirectional formatting characters. These are always removed,
# in every profile: they carry no content, they are invisible in every admin UI
# we will ever build, and left in place they defeat both numeric regexes and
# gazetteer lookups.
ZERO_WIDTH_AND_BIDI: Final[frozenset[str]] = frozenset(
    {
        "​",  # zero width space
        "‌",  # zero width non-joiner
        "‍",  # zero width joiner
        "‎",  # left-to-right mark
        "‏",  # right-to-left mark
        "؜",  # arabic letter mark
        "‪",  # left-to-right embedding
        "‫",  # right-to-left embedding
        "‬",  # pop directional formatting
        "‭",  # left-to-right override
        "‮",  # right-to-left override
        "⁠",  # word joiner
        "⁦",  # left-to-right isolate
        "⁧",  # right-to-left isolate
        "⁨",  # first strong isolate
        "⁩",  # pop directional isolate
        "﻿",  # zero width no-break space / BOM
    }
)

# NFKC does not touch these, so they need an explicit table.
_DIGIT_FOLD: Final[dict[str, str]] = {
    **{chr(0x0660 + n): str(n) for n in range(10)},  # Arabic-Indic
    **{chr(0x06F0 + n): str(n) for n in range(10)},  # extended Arabic-Indic (Persian)
}

_LETTER_FOLD: Final[dict[str, str]] = {
    "آ": "ا",  # alef with madda above
    "أ": "ا",  # alef with hamza above
    "إ": "ا",  # alef with hamza below
    "ٱ": "ا",  # alef wasla
    "ة": "ه",  # ta marbuta   -> heh
    "ى": "ي",  # alef maqsura -> yeh
    "ی": "ي",  # farsi yeh    -> yeh
    "ک": "ك",  # keheh        -> kaf
}


@dataclass(frozen=True, slots=True)
class NormalizationProfile:
    """Which folds a normalization pass applies.

    Zero-width and bidi stripping is unconditional and so is not represented
    here.
    """

    name: str
    fold_digits: bool = True
    apply_nfkc: bool = True
    strip_tatweel: bool = False
    strip_diacritics: bool = False
    fold_letters: bool = False


DIGITS: Final = NormalizationProfile(name="digits")
NAMES: Final = NormalizationProfile(
    name="names",
    strip_tatweel=True,
    strip_diacritics=True,
)
GAZETTEER: Final = NormalizationProfile(
    name="gazetteer",
    strip_tatweel=True,
    strip_diacritics=True,
    fold_letters=True,
)


class NormalizedText(NamedTuple):
    """``(normalized, offset_map)`` -- unpacks as the tuple the brief specifies."""

    text: str
    offset_map: list[int]


def normalize(text: str, profile: NormalizationProfile = DIGITS) -> NormalizedText:
    """Normalize ``text``, returning the result and its map back to the original.

    ``offset_map[i]`` is the index in ``text`` of the character that produced
    ``normalized[i]``. The map is non-decreasing and every entry is a valid
    index into ``text``.
    """
    out: list[str] = []
    offset_map: list[int] = []

    for index, char in enumerate(text):
        if char in ZERO_WIDTH_AND_BIDI:
            continue
        if profile.strip_tatweel and char == TATWEEL:
            continue
        if profile.strip_diacritics and char in ARABIC_DIACRITICS:
            continue

        # Per-character NFKC: folds presentation forms, ligatures and
        # full-width digits without ever composing across character
        # boundaries, which would make the expansion impossible to attribute
        # to a single source index.
        expanded = unicodedata.normalize("NFKC", char) if profile.apply_nfkc else char

        for produced in expanded:
            # NFKC can itself surface a combining mark (a presentation form
            # decomposing into letter + mark), so re-apply the filters.
            if produced in ZERO_WIDTH_AND_BIDI:
                continue
            if profile.strip_tatweel and produced == TATWEEL:
                continue
            if profile.strip_diacritics and produced in ARABIC_DIACRITICS:
                continue

            if profile.fold_digits:
                produced = _DIGIT_FOLD.get(produced, produced)
            if profile.fold_letters:
                produced = _LETTER_FOLD.get(produced, produced)

            out.append(produced)
            offset_map.append(index)

    return NormalizedText(text="".join(out), offset_map=offset_map)


def map_span_to_original(
    offset_map: list[int],
    start: int,
    end: int,
    original_len: int,
) -> tuple[int, int]:
    """Map a half-open span over normalized text back to original coordinates.

    The end is derived from the *last character inside* the span rather than
    from the first character after it, so a run of stripped characters
    immediately following the match is not swallowed into the reported span.
    A stripped run *inside* the span is included, which is what you want: the
    original substring has to be spliceable.
    """
    if not offset_map:
        raise ValueError("cannot map a span: the normalization produced no characters")
    if not 0 <= start < end <= len(offset_map):
        raise ValueError(
            f"span ({start}, {end}) is out of range for {len(offset_map)} normalized characters"
        )

    original_start = offset_map[start]
    original_end = min(offset_map[end - 1] + 1, original_len)
    return original_start, original_end
