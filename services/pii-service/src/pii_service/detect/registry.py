"""Presidio analyzer assembly.

Presidio is the orchestration layer here, not the detector (brief §4): it owns
the recognizer registry, result conflict resolution and the anonymizer. Every
recognizer that actually finds anything is ours, apart from three deterministic
built-ins (email, credit card, generic IBAN) that would be silly to rewrite.

Tier 2 and tier 3 are injected as factories rather than imported directly, so
that this module -- and therefore the service's startup path -- has no import
dependency on ``onnxruntime`` or ``gliner``. A deployment that runs tier 1 only
does not need those wheels present at all.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import date
from typing import Final

from presidio_analyzer import AnalyzerEngine, EntityRecognizer, RecognizerRegistry
from presidio_analyzer.predefined_recognizers import (
    CreditCardRecognizer,
    EmailRecognizer,
    IbanRecognizer,
)

from pii_service.detect.nlp_engine import DEFAULT_LANGUAGES, BlankSpacyNlpEngine
from pii_service.detect.tier1_patterns import build_tier1_recognizers
from pii_service.policy.loader import PolicyBundle

__all__ = ["RecognizerFactory", "build_analyzer"]

# A factory is called once per supported language and returns the recognizers
# to register for it. Returning an empty list is how a tier declines to load.
RecognizerFactory = Callable[[str], Sequence[EntityRecognizer]]

_BUILTIN_RECOGNIZERS: Final[dict[str, type[EntityRecognizer]]] = {
    "EMAIL_ADDRESS": EmailRecognizer,
    "CREDIT_CARD": CreditCardRecognizer,
    "IBAN_CODE": IbanRecognizer,
}


def build_analyzer(
    policy: PolicyBundle,
    *,
    languages: Sequence[str] = DEFAULT_LANGUAGES,
    today: date | None = None,
    tier2_factory: RecognizerFactory | None = None,
    tier3_factory: RecognizerFactory | None = None,
) -> AnalyzerEngine:
    """Build the analyzer for the configured policy.

    A built-in recognizer is registered only if ``entities.yaml`` declares its
    entity type. Detecting something the policy has no opinion about would
    produce findings nobody can act on and audit rows nobody can interpret.
    """
    nlp_engine = BlankSpacyNlpEngine(languages)
    nlp_engine.load()

    registry = RecognizerRegistry(supported_languages=list(languages))
    declared = policy.known_entity_types

    for language in languages:
        for recognizer in build_tier1_recognizers(
            supported_language=language,
            context_terms=policy.context_terms,
            today=today,
        ):
            if recognizer.supported_entities[0] in declared:
                registry.add_recognizer(recognizer)

        for entity_type, recognizer_cls in _BUILTIN_RECOGNIZERS.items():
            if entity_type in declared:
                registry.add_recognizer(recognizer_cls(supported_language=language))

        # Tiers 2 and 3 are registered for *every* language, not just the one
        # matching their script. Document-level language is a poor router: a
        # mostly-English prompt containing one Arabic name is language "en",
        # and registering the Arabic tier under "ar" alone would mean that name
        # is never looked at. Each tier selects the runs it can read from
        # `router.script_segments` instead.
        for factory in (tier2_factory, tier3_factory):
            if factory is None:
                continue
            for recognizer in factory(language):
                registry.add_recognizer(recognizer)

    return AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=list(languages),
        # Our own thresholds are per-entity and applied in the router, after
        # context boosting. A global floor here would silently drop a finding
        # that policy says to keep.
        default_score_threshold=0.0,
    )
