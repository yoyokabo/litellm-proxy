"""Policy loading, validation and preview rendering.

The preview tests carry most of the weight here. ``preview`` is the one column
in ``pii_events`` allowed to contain characters that came out of the user's
text, so every rule about it is a rule about how much PII ends up in the audit
database.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pii_service.policy.loader import PolicyError, load_policy_bundle
from pii_service.policy.models import (
    EntityAction,
    EntityCategory,
    PreviewRule,
    PreviewStrategy,
)


# ---------------------------------------------------------------------------
# The shipped bundle
# ---------------------------------------------------------------------------


def test_shipped_bundle_loads(policy: object) -> None:
    assert policy.entities  # type: ignore[attr-defined]
    assert len(policy.gazetteer.governorates) == 27  # type: ignore[attr-defined]


def test_every_entity_has_a_placeholder_and_threshold(policy: object) -> None:
    for name, entry in policy.entities.items():  # type: ignore[attr-defined]
        assert entry.placeholder, name
        assert 0.0 <= entry.score_threshold <= 1.0, name
        assert entry.category in set(EntityCategory), name


def test_national_id_threshold_admits_the_unverified_checksum_score(policy: object) -> None:
    """0.85 is a real, expected score -- the threshold must not exclude it."""
    assert policy.entities["EG_NATIONAL_ID"].score_threshold <= 0.85  # type: ignore[attr-defined]


def test_shape_only_entities_sit_below_their_own_threshold(policy: object) -> None:
    """Tax ID and passport must be unreachable without a context boost.

    Their base pattern scores are 0.3 and the default context boost is 0.35, so
    a threshold in (0.3, 0.65] is the band that makes context mandatory. If
    someone lowers one of these to 0.3, the entity silently starts firing on
    every nine-digit order number in the corpus.
    """
    for name in ("EG_TAX_ID", "EG_PASSPORT"):
        threshold = policy.entities[name].score_threshold  # type: ignore[attr-defined]
        assert 0.3 < threshold <= 0.65, name


def test_names_and_addresses_have_no_preview(policy: object) -> None:
    for name in ("AR_PERSON", "AR_LOCATION", "EG_ADDRESS", "PERSON", "LOCATION"):
        rule = policy.preview_rule_for(name)  # type: ignore[attr-defined]
        assert rule.strategy is PreviewStrategy.NONE, name
        assert policy.render_preview(name, "محمد علي") is None  # type: ignore[attr-defined]


def test_structured_ids_do_have_a_preview(policy: object) -> None:
    preview = policy.render_preview("EG_NATIONAL_ID", "28503122148219")  # type: ignore[attr-defined]
    assert preview == "2850******8219"


def test_context_terms_cover_every_family_the_recognizers_ask_for(policy: object) -> None:
    required = {"national_id", "mobile", "bank", "tax", "passport"}
    assert required <= set(policy.context_terms)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Preview rendering
# ---------------------------------------------------------------------------


def test_first_last_masks_the_middle() -> None:
    rule = PreviewRule(strategy=PreviewStrategy.FIRST_LAST, first=4, last=4)
    assert rule.render("28503122148219") == "2850******8219"


def test_last_only_keeps_the_tail() -> None:
    rule = PreviewRule(strategy=PreviewStrategy.LAST, last=4)
    assert rule.render("4111111111111111") == "************1111"


def test_none_strategy_renders_nothing() -> None:
    assert PreviewRule().render("anything") is None


def test_preview_refuses_when_nothing_would_be_hidden() -> None:
    rule = PreviewRule(strategy=PreviewStrategy.FIRST_LAST, first=4, last=4)
    assert rule.render("28501234") is None  # exactly 8 chars -- nothing masked
    assert rule.render("2850123") is None  # shorter still


def test_preview_always_hides_at_least_one_character() -> None:
    rule = PreviewRule(strategy=PreviewStrategy.FIRST_LAST, first=4, last=4)
    rendered = rule.render("285012345")
    assert rendered is not None
    assert rule.mask_char in rendered


def test_preview_of_empty_value_is_none() -> None:
    assert PreviewRule(strategy=PreviewStrategy.LAST, last=4).render("") is None


def test_preview_length_matches_value_length() -> None:
    rule = PreviewRule(strategy=PreviewStrategy.FIRST_LAST, first=4, last=4)
    value = "28503122148219"
    rendered = rule.render(value)
    assert rendered is not None
    assert len(rendered) == len(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": PreviewStrategy.FIRST_LAST, "first": 0, "last": 0},
        {"strategy": PreviewStrategy.LAST, "last": 0},
        {"strategy": PreviewStrategy.NONE, "first": 4},
        {"strategy": PreviewStrategy.FIRST_LAST, "first": 99, "last": 1},
        {"strategy": PreviewStrategy.FIRST_LAST, "first": 4, "mask_char": "xx"},
    ],
)
def test_incoherent_preview_rules_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PreviewRule(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Loader validation
# ---------------------------------------------------------------------------


def _write_bundle(
    directory: Path,
    *,
    entities: dict[str, object] | None = None,
    preview: dict[str, object] | None = None,
    gazetteer: dict[str, object] | None = None,
) -> Path:
    base_entities = {
        "version": 1,
        "entities": {"EG_NATIONAL_ID": {"category": "id", "action": "MASK"}},
    }
    base_preview = {"version": 1, "rules": {}}
    base_gazetteer = {
        "version": 1,
        "governorates": [{"code": "01", "ar": "القاهرة", "en": "Cairo"}],
        "structural_markers": {"street": ["شارع"]},
    }
    (directory / "entities.yaml").write_text(
        yaml.safe_dump(entities or base_entities, allow_unicode=True), encoding="utf-8"
    )
    (directory / "preview_policy.yaml").write_text(
        yaml.safe_dump(preview or base_preview, allow_unicode=True), encoding="utf-8"
    )
    (directory / "gazetteer_eg.yaml").write_text(
        yaml.safe_dump(gazetteer or base_gazetteer, allow_unicode=True), encoding="utf-8"
    )
    return directory


def test_minimal_bundle_loads(tmp_path: Path) -> None:
    bundle = load_policy_bundle(_write_bundle(tmp_path))
    assert bundle.entities["EG_NATIONAL_ID"].action is EntityAction.MASK


def test_missing_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(PolicyError, match="not found"):
        load_policy_bundle(tmp_path)


def test_malformed_yaml_is_an_error(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    (tmp_path / "entities.yaml").write_text("this: [is: not: valid", encoding="utf-8")
    with pytest.raises(PolicyError):
        load_policy_bundle(tmp_path)


def test_unknown_key_is_rejected(tmp_path: Path) -> None:
    """A misspelled key must fail loudly, not read as a default."""
    _write_bundle(
        tmp_path,
        entities={
            "version": 1,
            "entities": {"EG_NATIONAL_ID": {"catagory": "id"}},  # typo
        },
    )
    with pytest.raises(PolicyError):
        load_policy_bundle(tmp_path)


def test_unknown_action_is_rejected(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        entities={"version": 1, "entities": {"EG_NATIONAL_ID": {"action": "REDACT"}}},
    )
    with pytest.raises(PolicyError):
        load_policy_bundle(tmp_path)


def test_out_of_range_threshold_is_rejected(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        entities={"version": 1, "entities": {"EG_NATIONAL_ID": {"score_threshold": 1.5}}},
    )
    with pytest.raises(PolicyError):
        load_policy_bundle(tmp_path)


def test_preview_rule_for_an_undefined_entity_is_rejected(tmp_path: Path) -> None:
    """Catches a rename that landed in one file and not the other."""
    _write_bundle(
        tmp_path,
        preview={"version": 1, "rules": {"EG_NATIONAL_IDD": {"strategy": "none"}}},
    )
    with pytest.raises(PolicyError, match="does not define"):
        load_policy_bundle(tmp_path)


def test_duplicate_governorate_code_is_rejected(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        gazetteer={
            "version": 1,
            "governorates": [
                {"code": "01", "ar": "القاهرة", "en": "Cairo"},
                {"code": "01", "ar": "الجيزة", "en": "Giza"},
            ],
            "structural_markers": {},
        },
    )
    with pytest.raises(PolicyError, match="duplicate governorate"):
        load_policy_bundle(tmp_path)


def test_empty_entity_set_is_rejected(tmp_path: Path) -> None:
    _write_bundle(tmp_path, entities={"version": 1, "entities": {}})
    with pytest.raises(PolicyError, match="no entities"):
        load_policy_bundle(tmp_path)


def test_defaults_are_applied_to_sparse_entries(tmp_path: Path) -> None:
    _write_bundle(
        tmp_path,
        entities={
            "version": 1,
            "defaults": {"action": "ALLOW", "score_threshold": 0.9, "category": "finance"},
            "entities": {"THING": {}, "OTHER": {"action": "MASK"}},
        },
    )
    bundle = load_policy_bundle(tmp_path)

    thing = bundle.entities["THING"]
    assert thing.action is EntityAction.ALLOW
    assert thing.score_threshold == 0.9
    assert thing.category is EntityCategory.FINANCE
    assert thing.placeholder == "<THING>"
    assert bundle.entities["OTHER"].action is EntityAction.MASK
