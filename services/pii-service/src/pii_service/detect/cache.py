"""Detection cache.

The same PII reappears in every turn of a conversation, so an agentic workload
re-analyses the same system prompt and the same pasted record dozens of times.
Caching by content hash is where most of the latency saving lives (brief §7).

What the cache holds matters as much as that it exists. It stores
``PreparedSpan``s -- fingerprint and preview already computed, matched value
already discarded -- and never the text or the values. So the cache is exactly
as safe as the audit table, and a heap dump of a long-running service does not
contain a pile of national IDs.

The key is a SHA-256 of the text plus a policy fingerprint. Including the
policy means a threshold change in ``entities.yaml`` cannot be masked by stale
entries: edit the policy, and every cached verdict computed under the old one
is unreachable.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from typing import Final

from pii_service.policy.loader import PolicyBundle
from pii_service.spans import PreparedSpan

__all__ = ["DetectionCache"]


def policy_fingerprint(policy: PolicyBundle) -> str:
    """A short digest of everything that could change a cached answer.

    Covers the replacement rule as well as the detection policy, because the
    cache stores the *resolved* replacement text. An administrator switching
    PERSON from ``<PERSON>`` to a surrogate changes what a cached entry should
    say, and a digest that ignored it would serve the old masking until the
    entry aged out -- which is the kind of stale-policy bug nobody finds.

    The GLiNER prompt is covered too: it is the label tier 3 is conditioned on,
    so editing it changes what gets detected at all.
    """
    parts = [
        f"{name}:{p.action}:{p.score_threshold}:{p.placeholder}:{p.gliner_prompt}"
        f":{policy.replacement_rule_for(name).model_dump_json()}"
        for name, p in sorted(policy.entities.items())
    ]
    joined = "|".join(parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:16]


class DetectionCache:
    """A bounded LRU over value-free findings."""

    def __init__(self, policy: PolicyBundle, *, max_entries: int = 2048) -> None:
        self._entries: OrderedDict[str, tuple[PreparedSpan, ...]] = OrderedDict()
        self._max_entries: Final = max_entries
        self._policy_fp: Final = policy_fingerprint(policy)
        self.hits = 0
        self.misses = 0

    @property
    def enabled(self) -> bool:
        return self._max_entries > 0

    def key(self, text: str, language: str | None) -> str:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"{self._policy_fp}:{language or '-'}:{digest}"

    def get(self, text: str, language: str | None) -> tuple[PreparedSpan, ...] | None:
        if not self.enabled:
            return None
        key = self.key(text, language)
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return entry

    def put(self, text: str, language: str | None, spans: Sequence[PreparedSpan]) -> None:
        if not self.enabled:
            return
        key = self.key(text, language)
        self._entries[key] = tuple(spans)
        self._entries.move_to_end(key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()

    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "max_entries": self._max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 4) if total else 0.0,
        }
