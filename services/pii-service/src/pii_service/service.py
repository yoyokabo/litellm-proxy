"""The detection and audit authority.

Everything the two callers -- the LiteLLM guardrail and the chat backend -- need
goes through ``PiiService.analyze``. Having one entry point is what makes "one
audit writer either way" (brief §2) true rather than aspirational: there is no
second path that could detect without recording, or record with a different
fingerprint.
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Final, TypeVar

import structlog

from pii_service.api.schemas import AnalyzeRequest, AnalyzeResponse, Finding
from pii_service.audit.records import AuditRecord, RequestContext
from pii_service.audit.sink import AuditSink
from pii_service.detect.cache import DetectionCache
from pii_service.detect.mask import splice
from pii_service.detect.router import PiiRouter
from pii_service.policy.loader import PolicyBundle
from pii_service.policy.store import PolicyStore
from pii_service.settings import Settings
from pii_service.spans import PreparedSpan

__all__ = ["PiiService"]

logger: Final = structlog.get_logger(__name__)


class PiiService:
    def __init__(
        self,
        *,
        settings: Settings,
        policy: PolicyBundle,
        router: PiiRouter,
        sink: AuditSink,
        cache: DetectionCache,
        policy_store: PolicyStore | None = None,
    ) -> None:
        self._settings: Final = settings
        self._policy: Final = policy
        # When present, the effective policy comes from here, so an
        # administrator edit takes effect on the next request. Absent (tests,
        # standalone use) the constructor's bundle is the policy.
        self._policy_store: Final = policy_store
        self._router: Final = router
        self._sink: Final = sink
        self._cache: Final = cache

    @property
    def policy(self) -> PolicyBundle:
        """The effective policy. Read once per request, never mid-analysis."""
        store = self._policy_store
        return store.current if store is not None else self._policy

    def analyze(self, request: AnalyzeRequest) -> AnalyzeResponse:
        started = time.perf_counter()
        policy = self.policy

        masked_texts: list[str] = []
        all_spans: list[PreparedSpan] = []
        every_text_cached = True
        language: str | None = request.language

        for index, text in enumerate(request.texts):
            spans, from_cache, detected_lang = self._spans_for(text, index, request, policy)
            every_text_cached &= from_cache
            language = language or detected_lang

            all_spans.extend(spans)
            masked_texts.append(self._mask(text, spans))

        findings = [self._finding(span, request) for span in all_spans]
        counts = _counts(all_spans)
        blocking = _counts([s for s in all_spans if s.is_blocking])
        latency_ms = int((time.perf_counter() - started) * 1000)

        self._record(all_spans, request, latency_ms)

        return AnalyzeResponse(
            request_id=request.request_id,
            texts=masked_texts,
            findings=findings,
            entity_counts=counts,
            blocked=bool(blocking),
            block_reason=blocking or None,
            lang=language,
            latency_ms=latency_ms,
            cached=every_text_cached and bool(request.texts),
        )

    # -- internals ---------------------------------------------------------

    def _spans_for(
        self, text: str, index: int, request: AnalyzeRequest, policy: PolicyBundle
    ) -> tuple[tuple[PreparedSpan, ...], bool, str | None]:
        if not request.bypass_cache:
            cached = self._cache.get(text, request.language)
            if cached is not None:
                # The cache is keyed by text, and the same text can appear at
                # more than one position in a conversation, so the stored
                # index is meaningless -- re-stamp it for this call.
                return tuple(replace(span, text_index=index) for span in cached), True, None

        outcome = self._router.analyze(text, language=request.language, policy=policy)
        spans = tuple(
            PreparedSpan.from_detected(
                span,
                text_index=index,
                policy=policy,
                pepper=self._settings.pepper_bytes,
            )
            for span in outcome.spans
        )
        if not request.bypass_cache:
            self._cache.put(text, request.language, spans)
        return spans, False, outcome.lang

    def _mask(self, text: str, spans: tuple[PreparedSpan, ...]) -> str:
        """Splice each span's already-resolved replacement.

        The span carries the answer rather than the entity type, so a cache hit
        masks identically to a miss without needing the value or the pepper
        again.
        """
        return splice(
            text,
            [(span.start, span.end, span.replacement) for span in spans if span.is_masked],
        )

    def _finding(self, span: PreparedSpan, request: AnalyzeRequest) -> Finding:
        context: str | None = None
        context_offset: int | None = None

        if request.include_context and request.context_chars:
            # This snippet contains the matched value. It is opt-in, it goes
            # only to the caller, and it is never written to the audit row --
            # AuditRecord has no field that could hold it. The chat backend
            # turns it on to underline spans in the user's own message, which
            # is not a disclosure; the proxy guardrail leaves it off, because
            # its output reaches logs.
            text = request.texts[span.text_index]
            window = request.context_chars
            start = max(0, span.start - window)
            end = min(len(text), span.end + window)
            context = text[start:end]
            context_offset = span.start - start

        return Finding(
            text_index=span.text_index,
            entity_type=span.entity_type,
            category=span.category,
            recognizer=span.recognizer,
            score=span.score,
            action=span.action,
            start=span.start,
            end=span.end,
            value_len=span.value_len,
            value_fp=span.value_fp,
            preview=span.preview,
            context_term=span.context_term,
            lang=span.lang,
            context=context,
            context_offset=context_offset,
        )

    def _record(self, spans: list[PreparedSpan], request: AnalyzeRequest, latency_ms: int) -> None:
        if not spans:
            return

        records = [
            AuditRecord.from_prepared(
                span,
                RequestContext(
                    request_id=request.request_id,
                    user_id=request.identity.user_id,
                    team_id=request.identity.team_id,
                    key_alias=request.identity.key_alias,
                    key_hash=request.identity.key_hash,
                    end_user_id=request.identity.end_user_id,
                    model=request.model,
                    message_index=_at(request.message_indices, span.text_index),
                    message_role=_at(request.message_roles, span.text_index),
                    field_name=_at(request.fields, span.text_index),
                ),
                latency_ms=latency_ms,
            )
            for span in spans
        ]
        self._sink.submit(records)

    def cache_stats(self) -> dict[str, int | float]:
        return self._cache.stats()

    def invalidate_cache(self) -> None:
        """Drop cached verdicts after a policy change.

        The cache key already includes the policy fingerprint, so stale entries
        are unreachable rather than wrong. Clearing them anyway keeps the
        hit-rate statistic honest after an edit, and means memory is not held
        by verdicts no key will ever match again.
        """
        self._cache.clear()


def _counts(spans: list[PreparedSpan]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for span in spans:
        counts[span.entity_type] = counts.get(span.entity_type, 0) + 1
    return counts


_T = TypeVar("_T")


def _at(values: list[_T] | None, index: int) -> _T | None:
    """Read a parallel array safely -- a short one must degrade, not raise."""
    if values is None or index >= len(values):
        return None
    return values[index]
