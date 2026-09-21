"""Typed models for the YAML policy files.

Three files, deliberately separate because they change on different schedules
and by different people:

``entities.yaml``        what we detect and what we do about it -- compliance
``preview_policy.yaml``  how much of a value may reach the audit DB -- compliance
``gazetteer_eg.yaml``    Arabic terms and place names -- whoever tunes recall

All three are validated on load. A typo in a policy file must fail at startup
with a readable error, not at 3am as an entity that silently stopped masking.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Final, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from pii_service.policy.replacement import ReplacementRule

__all__ = [
    "EntitiesFile",
    "EntityAction",
    "EntityCategory",
    "EntityPolicy",
    "GazetteerFile",
    "Governorate",
    "PreviewPolicyFile",
    "PreviewRule",
    "PreviewStrategy",
    "ReplacementPolicyFile",
]


class _Strict(BaseModel):
    """Reject unknown keys. A misspelled policy key must not read as a default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EntityAction(StrEnum):
    MASK = "MASK"
    BLOCK = "BLOCK"
    ALLOW = "ALLOW"


class EntityCategory(StrEnum):
    """Drives the admin colour scale; kept separate from status colours (§9)."""

    ID = "id"
    PERSON = "person"
    CONTACT = "contact"
    LOCATION = "location"
    FINANCE = "finance"
    OTHER = "other"


Score = Annotated[float, Field(ge=0.0, le=1.0)]


class EntityDefaults(_Strict):
    action: EntityAction = EntityAction.MASK
    score_threshold: Score = 0.5
    category: EntityCategory = EntityCategory.OTHER


class EntityPolicy(_Strict):
    """Policy for one entity type, with defaults already applied."""

    entity_type: str
    category: EntityCategory
    action: EntityAction
    score_threshold: Score
    placeholder: str
    tier: int | None = None
    gliner_prompt: str | None = None
    """The natural-language label tier 3 is conditioned on.

    Lives with the entity rather than in a constant in the recognizer, so a
    label can be added from entities.yaml or from the admin overlay without
    touching code. Wording changes recall, so treat editing one as a model
    change and re-run scripts/eval_arabic_ner.py.
    """

    @property
    def is_masked(self) -> bool:
        return self.action is EntityAction.MASK

    @property
    def is_blocking(self) -> bool:
        return self.action is EntityAction.BLOCK


class _EntityEntry(_Strict):
    """One entry as written in the file, before defaults are applied."""

    category: EntityCategory | None = None
    action: EntityAction | None = None
    score_threshold: Score | None = None
    placeholder: str | None = None
    tier: int | None = None
    gliner_prompt: str | None = None


class EntitiesFile(_Strict):
    version: int
    defaults: EntityDefaults = EntityDefaults()
    entities: dict[str, _EntityEntry]

    def resolved(self) -> dict[str, EntityPolicy]:
        """Apply defaults, returning the policy actually used at runtime."""
        return {
            name: EntityPolicy(
                entity_type=name,
                category=entry.category or self.defaults.category,
                action=entry.action or self.defaults.action,
                score_threshold=(
                    self.defaults.score_threshold
                    if entry.score_threshold is None
                    else entry.score_threshold
                ),
                placeholder=entry.placeholder or f"<{name}>",
                tier=entry.tier,
                gliner_prompt=entry.gliner_prompt,
            )
            for name, entry in self.entities.items()
        }


# ---------------------------------------------------------------------------
# Preview policy
# ---------------------------------------------------------------------------


class PreviewStrategy(StrEnum):
    NONE = "none"
    FIRST_LAST = "first_last"
    LAST = "last"


class PreviewRule(_Strict):
    """How much of a matched value may be written to ``pii_events.preview``."""

    strategy: PreviewStrategy = PreviewStrategy.NONE
    first: int = Field(default=0, ge=0, le=8)
    last: int = Field(default=0, ge=0, le=8)
    mask_char: str = Field(default="*", min_length=1, max_length=1)

    @model_validator(mode="after")
    def _check_shape(self) -> Self:
        if self.strategy is PreviewStrategy.FIRST_LAST and self.first + self.last == 0:
            raise ValueError("first_last requires a non-zero first or last")
        if self.strategy is PreviewStrategy.LAST and self.last == 0:
            raise ValueError("last requires a non-zero last")
        if self.strategy is PreviewStrategy.NONE and (self.first or self.last):
            raise ValueError("strategy 'none' must not set first/last")
        return self

    def render(self, value: str) -> str | None:
        """Render the preview for ``value``, or ``None`` when none may be shown.

        The reconstruction budget is the policy author's call -- they set
        ``first`` and ``last`` per entity type knowing that entity's fixed
        length. This method does not second-guess that with a stricter rule of
        its own, because a guard that quietly returns ``None`` for a policy
        somebody deliberately wrote is just a silent policy change.

        What it does enforce is the one thing that is never a judgement call:
        a preview must hide at least one character. If the value is shorter
        than the rule expects, so that nothing would be masked, no preview is
        emitted at all.
        """
        if self.strategy is PreviewStrategy.NONE or not value:
            return None

        keep_first = self.first if self.strategy is PreviewStrategy.FIRST_LAST else 0
        keep_last = self.last

        hidden = len(value) - keep_first - keep_last
        if hidden < 1:
            return None

        return f"{value[:keep_first]}{self.mask_char * hidden}{value[len(value) - keep_last :]}"


class PreviewPolicyFile(_Strict):
    version: int
    default: PreviewRule = PreviewRule()
    rules: dict[str, PreviewRule] = Field(default_factory=dict)

    def rule_for(self, entity_type: str) -> PreviewRule:
        return self.rules.get(entity_type, self.default)


# ---------------------------------------------------------------------------
# Gazetteer
# ---------------------------------------------------------------------------


class Governorate(_Strict):
    code: str = Field(pattern=r"^\d{2}$")
    ar: str
    en: str


class GazetteerFile(_Strict):
    version: int
    governorates: list[Governorate]
    structural_markers: dict[str, list[str]]
    address_marker_groups: list[str] = Field(
        default_factory=lambda: ["street", "building", "unit", "postal"]
    )
    localities: list[str] = Field(default_factory=list)
    context_terms: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def all_markers(self) -> tuple[str, ...]:
        """Every structural marker, longest first so a greedy match wins."""
        seen: list[str] = [m for markers in self.structural_markers.values() for m in markers]
        return tuple(sorted(set(seen), key=len, reverse=True))

    @property
    def address_markers(self) -> tuple[str, ...]:
        """Only the markers strong enough to define an address on their own.

        Excludes the `area` group by default: those are ordinary nouns, and
        matching on one alone turns "Cairo is a big city" into an address.
        """
        seen: list[str] = [
            marker
            for group in self.address_marker_groups
            for marker in self.structural_markers.get(group, [])
        ]
        return tuple(sorted(set(seen), key=len, reverse=True))

    @property
    def governorate_names(self) -> tuple[str, ...]:
        return tuple(g.ar for g in self.governorates) + tuple(g.en for g in self.governorates)


class ReplacementPolicyFile(_Strict):
    """How each entity's spans are rewritten in the text sent to the model."""

    version: int
    default: ReplacementRule = ReplacementRule()
    rules: dict[str, ReplacementRule] = Field(default_factory=dict)

    def rule_for(self, entity_type: str) -> ReplacementRule:
        return self.rules.get(entity_type, self.default)

    @property
    def realistic_entities(self) -> tuple[str, ...]:
        """Entities whose masked output could be mistaken for real data.

        Surfaced by /health and the admin API so a deployment cannot drift into
        emitting realistic fakes without anyone noticing.
        """
        return tuple(sorted(name for name, rule in self.rules.items() if rule.is_realistic))


DEFAULT_ENTITIES_FILENAME: Final = "entities.yaml"
DEFAULT_PREVIEW_FILENAME: Final = "preview_policy.yaml"
DEFAULT_GAZETTEER_FILENAME: Final = "gazetteer_eg.yaml"
DEFAULT_REPLACEMENT_FILENAME: Final = "replacement_policy.yaml"
