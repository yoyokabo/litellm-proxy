"""Tests for the detection cache.

Two properties matter beyond "it caches":

* the cache holds no PII, so a heap dump of a long-running service is not a
  pile of national IDs; and
* a policy change invalidates it, so editing a threshold in ``entities.yaml``
  cannot be silently defeated by stale verdicts.
"""

from __future__ import annotations

import random

from conftest import CONFIG_DIR
from pii_service.detect.cache import DetectionCache, policy_fingerprint
from pii_service.policy.loader import PolicyBundle, load_policy_bundle
from pii_service.policy.models import EntityAction, EntityCategory
from pii_service.spans import PreparedSpan
from pii_service.synthetic import synthetic_national_id


def _span(entity_type: str = "EG_NATIONAL_ID") -> PreparedSpan:
    return PreparedSpan(
        text_index=0,
        entity_type=entity_type,
        category=str(EntityCategory.ID),
        recognizer="EgyptianNationalIdRecognizer",
        score=1.0,
        action=str(EntityAction.MASK),
        start=3,
        end=17,
        value_len=14,
        lang="en",
        value_fp="deadbeefdeadbeef",
        preview="2850******8219",
    )


def test_miss_then_hit(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy)
    assert cache.get("some text", "en") is None
    cache.put("some text", "en", [_span()])

    cached = cache.get("some text", "en")
    assert cached is not None
    assert len(cached) == 1
    assert cache.hits == 1
    assert cache.misses == 1


def test_language_is_part_of_the_key(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy)
    cache.put("text", "en", [_span()])
    assert cache.get("text", "ar") is None
    assert cache.get("text", "en") is not None


def test_different_text_is_a_miss(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy)
    cache.put("text a", "en", [_span()])
    assert cache.get("text b", "en") is None


def test_lru_evicts_the_oldest(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy, max_entries=3)
    for index in range(4):
        cache.put(f"text-{index}", "en", [_span()])

    assert cache.get("text-0", "en") is None  # evicted
    for index in (1, 2, 3):
        assert cache.get(f"text-{index}", "en") is not None


def test_a_hit_refreshes_recency(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy, max_entries=2)
    cache.put("a", "en", [_span()])
    cache.put("b", "en", [_span()])
    cache.get("a", "en")  # 'a' is now the most recent
    cache.put("c", "en", [_span()])

    assert cache.get("a", "en") is not None
    assert cache.get("b", "en") is None


def test_zero_size_cache_is_disabled(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy, max_entries=0)
    assert cache.enabled is False
    cache.put("text", "en", [_span()])
    assert cache.get("text", "en") is None


def test_empty_finding_list_is_cached_as_a_negative_result(policy: PolicyBundle) -> None:
    """Text with no PII is the common case; not caching it wastes the cache."""
    cache = DetectionCache(policy)
    cache.put("harmless", "en", [])

    cached = cache.get("harmless", "en")
    assert cached == ()
    assert cache.hits == 1


def test_clear_empties_the_cache(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy)
    cache.put("text", "en", [_span()])
    cache.clear()
    assert cache.get("text", "en") is None


def test_stats_report_hit_rate(policy: PolicyBundle) -> None:
    cache = DetectionCache(policy)
    cache.put("text", "en", [_span()])
    cache.get("text", "en")
    cache.get("other", "en")

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["hit_rate"] == 0.5
    assert stats["entries"] == 1


# ---------------------------------------------------------------------------
# The two properties that matter
# ---------------------------------------------------------------------------


def test_the_cache_key_is_not_the_text(policy: PolicyBundle) -> None:
    """Keys land in memory dumps and, if anyone logs stats, in logs."""
    nid = synthetic_national_id(rng=random.Random(3))
    cache = DetectionCache(policy)
    key = cache.key(f"my id is {nid}", "en")

    assert nid not in key
    assert "my id is" not in key


def test_cached_entries_hold_no_matched_value(policy: PolicyBundle) -> None:
    nid = synthetic_national_id(rng=random.Random(5))
    cache = DetectionCache(policy)
    cache.put(f"id {nid}", "en", [_span()])

    cached = cache.get(f"id {nid}", "en")
    assert cached is not None
    for span in cached:
        assert not hasattr(span, "value")
        assert nid not in repr(span)


def test_a_policy_change_invalidates_the_cache(tmp_path: object) -> None:
    """A threshold edit must not be defeated by a stale verdict."""
    import shutil
    from pathlib import Path

    directory = Path(str(tmp_path))
    for name in ("entities.yaml", "preview_policy.yaml", "gazetteer_eg.yaml"):
        shutil.copy(CONFIG_DIR / name, directory / name)

    before = load_policy_bundle(directory)
    cache_before = DetectionCache(before)
    key_before = cache_before.key("text", "en")

    # Raise one threshold, exactly as compliance would.
    entities = (directory / "entities.yaml").read_text(encoding="utf-8")
    entities = entities.replace("score_threshold: 0.8", "score_threshold: 0.95", 1)
    (directory / "entities.yaml").write_text(entities, encoding="utf-8")

    after = load_policy_bundle(directory)
    assert policy_fingerprint(before) != policy_fingerprint(after)
    assert DetectionCache(after).key("text", "en") != key_before


def test_policy_fingerprint_is_stable_for_an_unchanged_policy(policy: PolicyBundle) -> None:
    assert policy_fingerprint(policy) == policy_fingerprint(policy)
    assert len(policy_fingerprint(policy)) == 16
