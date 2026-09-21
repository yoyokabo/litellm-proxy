"""Load and validate the YAML policy bundle."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path
from typing import Final

import yaml

from pii_service.policy.models import (
    DEFAULT_ENTITIES_FILENAME,
    DEFAULT_GAZETTEER_FILENAME,
    DEFAULT_PREVIEW_FILENAME,
    DEFAULT_REPLACEMENT_FILENAME,
    EntitiesFile,
    EntityAction,
    EntityPolicy,
    GazetteerFile,
    PreviewPolicyFile,
    PreviewRule,
    ReplacementPolicyFile,
)
from pii_service.policy.overlay import EntityOverlay, apply_overlay
from pii_service.policy.replacement import ReplacementRule

__all__ = ["PolicyBundle", "PolicyError", "load_policy_bundle"]


class PolicyError(RuntimeError):
    """A policy file is missing, malformed, or internally inconsistent."""


# No `slots=True`: the cached_property members below need a __dict__.
@dataclass(frozen=True)
class PolicyBundle:
    """The three policy files, validated and cross-checked."""

    entities_file: EntitiesFile
    preview_file: PreviewPolicyFile
    gazetteer: GazetteerFile
    source_dir: Path
    replacement_file: ReplacementPolicyFile = field(
        default_factory=lambda: ReplacementPolicyFile(version=1)
    )
    # Administrator changes layered on the YAML baseline. Empty until the
    # service loads them from custom_entities at startup or after an edit.
    overlay: tuple[EntityOverlay, ...] = ()

    def with_overlay(self, overlays: Sequence[EntityOverlay]) -> PolicyBundle:
        """Return a new bundle with these overlays applied.

        A new instance rather than mutation: the cached properties below are
        the effective policy, and a bundle whose meaning changes under a
        request that already read it is a race nobody will debug.
        """
        return replace(self, overlay=tuple(overlays))

    @cached_property
    def entities(self) -> dict[str, EntityPolicy]:
        return apply_overlay(self.entities_file.resolved(), self.overlay)

    @cached_property
    def _overlay_by_type(self) -> dict[str, EntityOverlay]:
        return {o.entity_type: o for o in self.overlay}

    # -- replacement -------------------------------------------------------

    def replacement_rule_for(self, entity_type: str) -> ReplacementRule:
        """The overlay's rule wins, then the YAML rule, then the default."""
        overlay = self._overlay_by_type.get(entity_type)
        if overlay is not None and overlay.replacement is not None:
            return overlay.replacement
        return self.replacement_file.rule_for(entity_type)

    def render_replacement(self, entity_type: str, value_fp: str | None) -> str:
        """The text a masked span of this type becomes."""
        policy = self.policy_for(entity_type)
        placeholder = policy.placeholder if policy else f"<{entity_type}>"
        return self.replacement_rule_for(entity_type).render(
            placeholder=placeholder, value_fp=value_fp
        )

    @cached_property
    def _replacement_vocabulary(self) -> dict[str, frozenset[str]]:
        """Everything each entity's rule can emit, for the idempotency check."""
        vocabulary: dict[str, frozenset[str]] = {}
        for name, policy in self.entities.items():
            vocabulary[name] = self.replacement_rule_for(name).vocabulary(policy.placeholder)
        return vocabulary

    def is_replacement_value(self, entity_type: str, value: str) -> bool:
        """True when this value is already the output of masking this entity.

        Masking it again would replace a surrogate with a different surrogate,
        corrupting the text a little more on every hop -- and the architecture
        masks twice by design (brief §2). Suppressing these is what keeps a
        second pass a no-op.
        """
        return value.strip() in self._replacement_vocabulary.get(entity_type, frozenset())

    @cached_property
    def realistic_replacement_entities(self) -> tuple[str, ...]:
        """Entities whose masked output could be mistaken for real data."""
        return tuple(
            sorted(name for name in self.entities if self.replacement_rule_for(name).is_realistic)
        )

    # -- tier 3 labels -----------------------------------------------------

    @cached_property
    def gliner_prompts(self) -> dict[str, str]:
        """entity type -> the label tier 3 is conditioned on.

        Baseline entries and administrator-added ones are indistinguishable
        here, which is the point: a custom label is a first-class entity, not a
        second-class bolt-on.
        """
        return {
            name: policy.gliner_prompt
            for name, policy in sorted(self.entities.items())
            if policy.gliner_prompt
        }

    def policy_for(self, entity_type: str) -> EntityPolicy | None:
        return self.entities.get(entity_type)

    def preview_rule_for(self, entity_type: str) -> PreviewRule:
        return self.preview_file.rule_for(entity_type)

    def render_preview(self, entity_type: str, value: str) -> str | None:
        """The only place a matched value is turned into something storable."""
        return self.preview_rule_for(entity_type).render(value)

    @cached_property
    def blocking_entities(self) -> frozenset[str]:
        return frozenset(
            name for name, policy in self.entities.items() if policy.action is EntityAction.BLOCK
        )

    @cached_property
    def context_terms(self) -> dict[str, list[str]]:
        return self.gazetteer.context_terms

    @cached_property
    def known_entity_types(self) -> frozenset[str]:
        return frozenset(self.entities)


def _read_yaml(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise PolicyError(f"policy file not found: {path}")
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PolicyError(f"{path} is not valid YAML: {exc}") from exc
    if not isinstance(loaded, dict):
        raise PolicyError(f"{path} must contain a mapping at the top level")
    return loaded


def load_policy_bundle(config_dir: Path | str) -> PolicyBundle:
    """Load, validate and cross-check the policy bundle in ``config_dir``.

    Raises ``PolicyError`` on anything wrong. Called at startup, before the
    service binds a port: a policy typo should stop a deployment, not degrade
    one silently.
    """
    directory: Final = Path(config_dir)

    try:
        entities = EntitiesFile.model_validate(_read_yaml(directory / DEFAULT_ENTITIES_FILENAME))
        preview = PreviewPolicyFile.model_validate(_read_yaml(directory / DEFAULT_PREVIEW_FILENAME))
        gazetteer = GazetteerFile.model_validate(_read_yaml(directory / DEFAULT_GAZETTEER_FILENAME))
        # Optional, unlike the other three. It was added after the first
        # release, and its absence means "placeholders everywhere" -- which is
        # exactly the behaviour a deployment had before it existed. Making it
        # mandatory would break every existing config directory on upgrade to
        # restate a default they already had.
        replacement_path = directory / DEFAULT_REPLACEMENT_FILENAME
        replacement = (
            ReplacementPolicyFile.model_validate(_read_yaml(replacement_path))
            if replacement_path.is_file()
            else ReplacementPolicyFile(version=1)
        )
    except PolicyError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise PolicyError(f"invalid policy in {directory}: {exc}") from exc

    bundle = PolicyBundle(
        entities_file=entities,
        preview_file=preview,
        gazetteer=gazetteer,
        replacement_file=replacement,
        source_dir=directory,
    )
    _cross_check(bundle)
    return bundle


def _cross_check(bundle: PolicyBundle) -> None:
    """Catch the inconsistencies that span two files.

    A preview rule for an entity nobody detects is dead policy, and worse, it
    usually means a rename landed in one file and not the other -- which would
    quietly downgrade that entity to the default `none`... or, in the other
    direction, quietly start writing previews for a name.
    """
    known = bundle.known_entity_types

    orphan_previews = sorted(set(bundle.preview_file.rules) - known)
    if orphan_previews:
        raise PolicyError(
            "preview_policy.yaml has rules for entity types that entities.yaml "
            f"does not define: {', '.join(orphan_previews)}"
        )

    orphan_replacements = sorted(set(bundle.replacement_file.rules) - known)
    if orphan_replacements:
        raise PolicyError(
            "replacement_policy.yaml has rules for entity types that entities.yaml "
            f"does not define: {', '.join(orphan_replacements)}"
        )

    duplicate_codes = _duplicates(g.code for g in bundle.gazetteer.governorates)
    if duplicate_codes:
        raise PolicyError(f"duplicate governorate codes: {', '.join(sorted(duplicate_codes))}")

    if not bundle.entities:
        raise PolicyError("entities.yaml defines no entities")


def _duplicates(values: object) -> set[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:  # type: ignore[attr-defined]
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return duplicates
