"""Tier 3: GLiNER2 for Latin-script text.

Schema-conditioned: the entity labels are passed at inference time in a single
forward pass, so adding a type is a line in ``entities.yaml`` rather than a
retrain. Supports EN/FR/ES/DE/IT/PT/NL and **not Arabic**, which is why this
recognizer filters itself to the Latin runs from ``router.script_segments``
rather than trusting document-level language.

OFF BY DEFAULT, AND THAT IS A MEASURED DECISION, NOT A PRECAUTION.

The brief (§4) flags the vendor latency claim as disputed: the paper reports
50-200 ms/document on 8-16 core CPU, an independent benchmark measured ~2.3 s
per chat-sized message on an M1. That is a 10-40x spread, and this guardrail
runs ``pre_call``, so its latency is added to every single request through the
gateway. At the high end it is unshippable.

``scripts/benchmark.py`` measures it per tier on the target CPU. Do not turn
this on, and do not let anyone design around a number, until that benchmark has
been run on hardware that resembles production.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final

import structlog
from presidio_analyzer import EntityRecognizer, RecognizerResult

from pii_service.detect.router import Script, script_segments

if TYPE_CHECKING:
    from pii_service.policy.loader import PolicyBundle
    from pii_service.settings import Settings

__all__ = ["GlinerRecognizer", "Tier3Unavailable", "build_tier3_factory"]

logger: Final = structlog.get_logger(__name__)

# Fallback prompts, used only when a caller builds the recognizer without a
# policy. The real source is ``PolicyBundle.gliner_prompts`` -- every entity
# with a ``gliner_prompt``, whether it came from entities.yaml or from an
# administrator adding a label at runtime.
#
# That indirection is the whole feature. GLiNER2 is schema-conditioned: labels
# are prompts passed in the forward pass, not trained classes, so a new entity
# type costs a config row rather than a retrain.
LABEL_PROMPTS: Final[dict[str, str]] = {
    "PERSON": "person name",
    "LOCATION": "location",
    "ORGANIZATION": "organization",
}

# Latin runs shorter than this are punctuation and stray words between Arabic
# clauses; running a transformer over them is pure cost.
_RECOGNIZER_NAME: Final = "GlinerRecognizer"

MIN_SEGMENT_CHARS: Final = 12


class Tier3Unavailable(RuntimeError):
    """Tier 3 was enabled but the gliner package is not installed."""


class GlinerRecognizer(EntityRecognizer):
    """GLiNER2 over Latin-script runs only."""

    def __init__(
        self,
        *,
        model_name: str,
        supported_language: str,
        supported_entities: Sequence[str] | None = None,
        prompts: Mapping[str, str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        # Set before super().__init__(): Presidio's EntityRecognizer.__init__
        # calls self.load(), which reads self._model_name. See the equivalent
        # comment in tier2_arabic_ner.py.
        label_prompts = dict(prompts if prompts is not None else LABEL_PROMPTS)
        if supported_entities is None:
            supported_entities = tuple(label_prompts)

        self._model_name: Final = model_name
        self._threshold: Final = threshold
        self._model: object | None = None
        # Reverse map, because a prediction comes back labelled with the prompt
        # we sent. Two entities sharing a prompt would make that ambiguous, so
        # the policy loader keeps prompts distinct.
        self._prompt_to_entity: dict[str, str] = {
            prompt: entity
            for entity, prompt in label_prompts.items()
            if entity in supported_entities
        }
        super().__init__(
            supported_entities=list(supported_entities),
            supported_language=supported_language,
            name=_RECOGNIZER_NAME,
        )

    def load(self) -> None:
        try:
            from gliner import GLiNER
        except ImportError as exc:
            raise Tier3Unavailable(
                "tier 3 is enabled but gliner is not installed. "
                "Install the 'gliner' extra: pip install 'pii-service[gliner]'."
            ) from exc

        # local_files_only: an air-gapped host must fail with a clear error
        # rather than hanging on a HuggingFace fetch that cannot succeed.
        self._model = GLiNER.from_pretrained(self._model_name, local_files_only=True)
        logger.info("tier3.loaded", model=self._model_name)

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: object | None = None,
    ) -> list[RecognizerResult]:
        if self._model is None or not text:
            return []

        prompts = [
            prompt for prompt, entity in self._prompt_to_entity.items() if entity in entities
        ]
        if not prompts:
            return []

        results: list[RecognizerResult] = []
        for segment in script_segments(text, min_length=MIN_SEGMENT_CHARS):
            if segment.script != Script.LATIN:
                continue
            chunk = text[segment.start : segment.end]

            for prediction in self._model.predict_entities(  # type: ignore[attr-defined]
                chunk, prompts, threshold=self._threshold
            ):
                entity_type = self._prompt_to_entity.get(prediction["label"])
                if entity_type is None:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=segment.start + int(prediction["start"]),
                        end=segment.start + int(prediction["end"]),
                        score=float(prediction.get("score", self._threshold)),
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: _RECOGNIZER_NAME,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                        },
                    )
                )
        return results


def build_tier3_factory(
    settings: Settings,
    *,
    model_name: str = "urchade/gliner_multi_pii-v1",
    policy: PolicyBundle | None = None,
) -> object:
    """Build the tier-3 factory, conditioned on the policy's labels.

    ``policy`` supplies ``gliner_prompts``: the baseline labels from
    entities.yaml plus anything an administrator added at runtime. Passing None
    falls back to the built-in three, which is only useful in isolation.
    """
    prompts = dict(policy.gliner_prompts) if policy is not None else dict(LABEL_PROMPTS)

    def factory(language: str) -> list[EntityRecognizer]:
        # Constructing it loads the model: EntityRecognizer.__init__ calls
        # load(), which is what raises Tier3Unavailable when gliner is missing.
        return [
            GlinerRecognizer(model_name=model_name, supported_language=language, prompts=prompts)
        ]

    return factory
