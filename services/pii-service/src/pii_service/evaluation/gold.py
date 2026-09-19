"""The Arabic NER gold set: inline markup in, character spans out.

Gold spans are authored as ``[surface](TYPE)`` inside the sentence and parsed
into offsets here. Nobody hand-counts a character offset in Arabic text --
that is precisely the error this whole codebase exists to avoid, and a gold set
with drifted offsets silently scores a correct model as wrong.

The registers matter as much as the entities. The decision this set exists to
settle is *mix-ner vs msa-ner*, and those two differ on dialect: `msa` cases
are the formal register a model trained on Modern Standard Arabic should
handle, `egyptian` cases are the dialect a real prompt from a Cairo engineer
actually contains. A variant that wins on `msa` and loses on `egyptian` is the
wrong choice for this deployment whatever its headline score.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final, Self

import yaml

__all__ = [
    "GoldCase",
    "GoldSet",
    "GoldSpan",
    "Register",
    "parse_markup",
]

# [surface](TYPE) -- no nesting, validated on parse.
_MARKUP: Final = re.compile(r"\[([^\[\]]+)\]\(([A-Z_]+)\)")

VALID_TYPES: Final[frozenset[str]] = frozenset({"PER", "LOC", "ORG"})


class Register(StrEnum):
    MSA = "msa"
    EGYPTIAN = "egyptian"
    MIXED_SCRIPT = "mixed_script"


@dataclass(frozen=True, slots=True)
class GoldSpan:
    start: int
    end: int
    label: str

    def overlaps(self, start: int, end: int) -> bool:
        return start < self.end and self.start < end

    @property
    def length(self) -> int:
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class GoldCase:
    case_id: str
    register: Register
    text: str
    spans: tuple[GoldSpan, ...]
    note: str | None = None

    @property
    def is_negative(self) -> bool:
        """A sentence with no entities at all -- what false positives are measured on."""
        return not self.spans


def parse_markup(marked: str) -> tuple[str, tuple[GoldSpan, ...]]:
    """Strip ``[surface](TYPE)`` markup, returning plain text and its spans.

    Offsets are accumulated as the markup is removed, so they are correct by
    construction for any script, any direction, and any amount of diacritics.
    """
    plain: list[str] = []
    spans: list[GoldSpan] = []
    cursor = 0

    for match in _MARKUP.finditer(marked):
        plain.append(marked[cursor : match.start()])
        surface, label = match.group(1), match.group(2)

        if label not in VALID_TYPES:
            raise ValueError(
                f"unknown entity type {label!r}; expected one of {sorted(VALID_TYPES)}"
            )
        if "[" in surface or "]" in surface:
            raise ValueError(f"nested markup is not supported: {surface!r}")

        start = sum(len(part) for part in plain)
        plain.append(surface)
        spans.append(GoldSpan(start=start, end=start + len(surface), label=label))
        cursor = match.end()

    plain.append(marked[cursor:])
    text = "".join(plain)

    for span in spans:
        assert text[span.start : span.end], f"empty span in {marked!r}"

    return text, tuple(spans)


@dataclass(frozen=True, slots=True)
class GoldSet:
    version: int
    cases: tuple[GoldCase, ...]

    @classmethod
    def load(cls, path: Path | str) -> Self:
        payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a mapping at the top level")

        cases: list[GoldCase] = []
        seen: set[str] = set()

        for raw in payload.get("cases") or []:
            case_id = str(raw["id"])
            if case_id in seen:
                raise ValueError(f"duplicate gold case id: {case_id}")
            seen.add(case_id)

            text, spans = parse_markup(raw["text"])
            cases.append(
                GoldCase(
                    case_id=case_id,
                    register=Register(raw.get("register", "msa")),
                    text=text,
                    spans=spans,
                    note=raw.get("note"),
                )
            )

        if not cases:
            raise ValueError(f"{path}: gold set is empty")

        return cls(version=int(payload.get("version", 1)), cases=tuple(cases))

    def by_register(self, register: Register) -> tuple[GoldCase, ...]:
        return tuple(case for case in self.cases if case.register is register)

    @property
    def positives(self) -> tuple[GoldCase, ...]:
        return tuple(case for case in self.cases if not case.is_negative)

    @property
    def negatives(self) -> tuple[GoldCase, ...]:
        return tuple(case for case in self.cases if case.is_negative)

    @property
    def span_count(self) -> int:
        return sum(len(case.spans) for case in self.cases)

    def label_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for case in self.cases:
            for span in case.spans:
                counts[span.label] = counts.get(span.label, 0) + 1
        return counts
