"""Text splicing for masking.

One implementation, used by the router (which has values) and by the service on
a cache hit (which does not). Two implementations would eventually disagree,
and the way they would disagree is that one of them stops masking something.

Presidio ships an ``AnonymizerEngine`` that does this, but it wants
``RecognizerResult`` objects and re-derives its own conflict resolution, which
we have already done more carefully in ``router._deconflict``. Converting back
into its types to have it redo that work is more code, more dependency surface
in an air-gapped image, and a second opinion about overlaps that we do not want.
"""

from __future__ import annotations

from collections.abc import Sequence

__all__ = ["Replacement", "splice"]

# (start, end, replacement) in original-string coordinates.
Replacement = tuple[int, int, str]


def splice(text: str, replacements: Sequence[Replacement]) -> str:
    """Apply replacements to ``text``, right to left.

    Right to left so that earlier offsets stay valid as later ones are
    rewritten -- the alternative is tracking a running delta, which is the
    classic place an off-by-one turns into a half-masked identifier.

    Overlapping replacements are a programming error: the caller deconflicts
    first. This raises rather than silently producing mangled output, because
    mangled output here means leaked characters.
    """
    if not replacements:
        return text

    ordered = sorted(replacements, key=lambda item: (item[0], item[1]))

    for (start, end, _), (next_start, _, _) in zip(ordered, ordered[1:], strict=False):
        if end > next_start:
            raise ValueError(
                f"overlapping replacements: ({start}, {end}) overlaps ({next_start}, ...)"
            )

    result = text
    for start, end, replacement in reversed(ordered):
        if not 0 <= start <= end <= len(result):
            raise ValueError(f"replacement ({start}, {end}) is out of range for {len(result)}")
        result = f"{result[:start]}{replacement}{result[end:]}"
    return result
