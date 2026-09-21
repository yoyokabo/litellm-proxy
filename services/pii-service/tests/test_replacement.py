"""Replacement strategies and the overlay merge.

The load-bearing test in this file is the idempotency one. Realistic
replacement is the feature that can corrupt text silently: the architecture
masks twice (brief §2), and "John Doe" is a name, so without suppression the
second pass replaces it with a different name and the prompt degrades on every
hop. Nobody would see that in a unit test of the strategy alone.
"""

from __future__ import annotations

import pytest

from conftest import CONFIG_DIR
from pii_service.policy.loader import PolicyBundle, load_policy_bundle
from pii_service.policy.models import EntityAction, EntityCategory, EntityPolicy
from pii_service.policy.overlay import EntityOverlay, apply_overlay
from pii_service.policy.replacement import ReplacementRule, ReplacementStrategy

POOL = ("John Doe", "Jane Roe", "Sam Poe", "Alex Coe")
FP = "9f75871d3d222bcb"


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------


def test_placeholder_is_the_default() -> None:
    rule = ReplacementRule()
    assert rule.strategy is ReplacementStrategy.PLACEHOLDER
    assert rule.render(placeholder="<PERSON>", value_fp=FP) == "<PERSON>"
    assert rule.is_realistic is False


def test_constant_returns_the_configured_value() -> None:
    rule = ReplacementRule(strategy=ReplacementStrategy.CONSTANT, value="John Doe")
    assert rule.render(placeholder="<PERSON>", value_fp=FP) == "John Doe"
    assert rule.is_realistic is True


def test_redact_removes_the_span() -> None:
    rule = ReplacementRule(strategy=ReplacementStrategy.REDACT)
    assert rule.render(placeholder="<PERSON>", value_fp=FP) == ""


def test_labelled_fingerprint_keeps_distinct_values_distinct() -> None:
    rule = ReplacementRule(strategy=ReplacementStrategy.LABELLED_FINGERPRINT)
    first = rule.render(placeholder="<PERSON>", value_fp="9f75871d3d222bcb")
    second = rule.render(placeholder="<PERSON>", value_fp="ac90fef8651532d3")

    assert first == "<PERSON:9f75871d>"
    assert first != second
    assert rule.is_realistic is False  # still obviously masked


def test_labelled_fingerprint_without_a_fingerprint_falls_back() -> None:
    rule = ReplacementRule(strategy=ReplacementStrategy.LABELLED_FINGERPRINT)
    assert rule.render(placeholder="<PERSON>", value_fp=None) == "<PERSON>"


def test_surrogate_is_stable_for_the_same_value() -> None:
    """The same person must be the same fake name in every turn."""
    rule = ReplacementRule(strategy=ReplacementStrategy.SURROGATE, pool=POOL)
    assert rule.render(placeholder="<PERSON>", value_fp=FP) == rule.render(
        placeholder="<PERSON>", value_fp=FP
    )


def test_surrogate_keeps_distinct_values_distinct() -> None:
    rule = ReplacementRule(strategy=ReplacementStrategy.SURROGATE, pool=POOL)
    rendered = {rule.render(placeholder="<PERSON>", value_fp=f"{n:016x}") for n in range(200)}
    assert rendered == set(POOL), "every pool entry should be reachable"


def test_surrogate_is_uniform_over_the_pool() -> None:
    """A skewed pick would make one surrogate stand for most people."""
    rule = ReplacementRule(strategy=ReplacementStrategy.SURROGATE, pool=POOL)
    counts: dict[str, int] = {}
    for n in range(4000):
        name = rule.render(placeholder="<PERSON>", value_fp=f"{n * 2654435761:016x}")
        counts[name] = counts.get(name, 0) + 1

    assert all(700 < count < 1300 for count in counts.values()), counts


def test_surrogate_without_a_fingerprint_falls_back_to_the_placeholder() -> None:
    """No fingerprint means nothing stable to key on; do not pick arbitrarily."""
    rule = ReplacementRule(strategy=ReplacementStrategy.SURROGATE, pool=POOL)
    assert rule.render(placeholder="<PERSON>", value_fp=None) == "<PERSON>"


def test_a_rule_can_override_the_entity_placeholder() -> None:
    rule = ReplacementRule(placeholder="[redacted]")
    assert rule.render(placeholder="<PERSON>", value_fp=FP) == "[redacted]"


# ---------------------------------------------------------------------------
# Validation -- an incoherent rule must not reach a deployment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strategy": "constant"},  # no value
        {"strategy": "constant", "value": "   "},  # blank value
        {"strategy": "surrogate"},  # no pool
        {"strategy": "surrogate", "pool": ("John Doe", "")},  # blank entry
        {"strategy": "surrogate", "pool": ("John Doe", "John Doe")},  # duplicate
        {"strategy": "placeholder", "pool": ("John Doe",)},  # pool on wrong strategy
        {"strategy": "redact", "value": "x"},  # value on wrong strategy
    ],
)
def test_incoherent_rules_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ReplacementRule(**kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Idempotency -- the reason this feature is not just a string swap
# ---------------------------------------------------------------------------


def test_vocabulary_lists_everything_a_rule_can_emit() -> None:
    rule = ReplacementRule(strategy=ReplacementStrategy.SURROGATE, pool=POOL)
    assert rule.vocabulary("<PERSON>") == frozenset({*POOL, "<PERSON>"})


def test_constant_vocabulary_includes_its_value() -> None:
    rule = ReplacementRule(strategy=ReplacementStrategy.CONSTANT, value="John Doe")
    assert rule.vocabulary("<PERSON>") == frozenset({"John Doe", "<PERSON>"})


def test_policy_recognises_its_own_replacement_output(policy: PolicyBundle) -> None:
    """The check the router uses to suppress an already-masked value."""
    configured = policy.with_overlay(
        [
            EntityOverlay(
                entity_type="EG_MOBILE",
                replacement=ReplacementRule(strategy=ReplacementStrategy.SURROGATE, pool=POOL),
            )
        ]
    )

    assert configured.is_replacement_value("EG_MOBILE", "John Doe")
    assert configured.is_replacement_value("EG_MOBILE", "  John Doe  ")  # trimmed
    assert configured.is_replacement_value("EG_MOBILE", "<EG_MOBILE>")
    assert not configured.is_replacement_value("EG_MOBILE", "01001152449")
    # Scoped to the entity: another type's surrogate is not this one's output.
    assert not configured.is_replacement_value("EG_NATIONAL_ID", "John Doe")


def test_realistic_entities_are_reported(policy: PolicyBundle) -> None:
    """/health and the admin API warn on these; the warning needs a source."""
    assert policy.realistic_replacement_entities == ()

    configured = policy.with_overlay(
        [
            EntityOverlay(
                entity_type="PERSON",
                replacement=ReplacementRule(
                    strategy=ReplacementStrategy.CONSTANT, value="John Doe"
                ),
            )
        ]
    )
    assert configured.realistic_replacement_entities == ("PERSON",)


# ---------------------------------------------------------------------------
# Overlay merge
# ---------------------------------------------------------------------------


def _baseline() -> dict[str, EntityPolicy]:
    return {
        "PERSON": EntityPolicy(
            entity_type="PERSON",
            category=EntityCategory.PERSON,
            action=EntityAction.MASK,
            score_threshold=0.7,
            placeholder="<PERSON>",
            tier=3,
            gliner_prompt="person name",
        )
    }


def test_an_overlay_overrides_only_what_it_sets() -> None:
    """A replacement change must not reset a threshold somebody tuned."""
    merged = apply_overlay(
        _baseline(),
        (
            EntityOverlay(
                entity_type="PERSON",
                replacement=ReplacementRule(
                    strategy=ReplacementStrategy.CONSTANT, value="John Doe"
                ),
            ),
        ),
    )
    assert merged["PERSON"].score_threshold == 0.7
    assert merged["PERSON"].gliner_prompt == "person name"


def test_an_overlay_adds_a_new_tier3_label() -> None:
    merged = apply_overlay(
        _baseline(),
        (
            EntityOverlay(
                entity_type="PROJECT_CODENAME",
                gliner_prompt="internal project codename",
            ),
        ),
    )
    assert merged["PROJECT_CODENAME"].gliner_prompt == "internal project codename"
    assert merged["PROJECT_CODENAME"].tier == 3
    assert merged["PROJECT_CODENAME"].placeholder == "<PROJECT_CODENAME>"


def test_a_new_entity_without_a_prompt_is_rejected() -> None:
    """Nothing would ever detect it, so it is policy that silently does nothing."""
    with pytest.raises(ValueError, match="nothing would ever detect it"):
        apply_overlay(_baseline(), (EntityOverlay(entity_type="GHOST"),))


def test_disabling_removes_the_entity_from_the_effective_policy() -> None:
    merged = apply_overlay(_baseline(), (EntityOverlay(entity_type="PERSON", enabled=False),))
    assert "PERSON" not in merged


@pytest.mark.parametrize("bad", ["lowercase", "With-Dash", "1LEADING", "", "X"])
def test_malformed_entity_types_are_rejected(bad: str) -> None:
    with pytest.raises(ValueError):
        EntityOverlay(entity_type=bad, gliner_prompt="something")


def test_gliner_prompts_merge_baseline_and_overlay() -> None:
    bundle = load_policy_bundle(CONFIG_DIR).with_overlay(
        [EntityOverlay(entity_type="BADGE_NUMBER", gliner_prompt="employee badge number")]
    )
    prompts = bundle.gliner_prompts

    assert prompts["PERSON"] == "person name"  # baseline
    assert prompts["BADGE_NUMBER"] == "employee badge number"  # runtime


def test_the_overlay_does_not_mutate_the_baseline_bundle(policy: PolicyBundle) -> None:
    """A store swap must leave the previous bundle intact for in-flight requests."""
    before = dict(policy.entities)
    policy.with_overlay([EntityOverlay(entity_type="PERSON", enabled=False)])
    assert policy.entities == before
