"""Load and validate the YAML policy bundle."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Final

import yaml

from pii_service.policy.models import (
    DEFAULT_ENTITIES_FILENAME,
    DEFAULT_GAZETTEER_FILENAME,
    DEFAULT_PREVIEW_FILENAME,
    EntitiesFile,
    EntityAction,
    EntityPolicy,
    GazetteerFile,
    PreviewPolicyFile,
    PreviewRule,
)

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

    @cached_property
    def entities(self) -> dict[str, EntityPolicy]:
        return self.entities_file.resolved()

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
    except PolicyError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise PolicyError(f"invalid policy in {directory}: {exc}") from exc

    bundle = PolicyBundle(
        entities_file=entities,
        preview_file=preview,
        gazetteer=gazetteer,
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
