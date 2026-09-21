"""Tier 2: Arabic NER (ONNX int8) and the Egyptian address gazetteer.

Two recognizers, and the brief is explicit that they are different kinds of
thing (§4):

``ArabicNerRecognizer`` wraps a CAMeLBERT-family token-classification model
exported to ONNX int8 and run through ``onnxruntime`` with the Rust
``tokenizers`` binding. **No PyTorch and no transformers at inference time** --
those belong in the builder stage that produces the ONNX artifact (phase 4),
not in the runtime image. It maps PER -> AR_PERSON, LOC -> AR_LOCATION,
ORG -> AR_ORG.

``EgyptianAddressRecognizer`` does *not* use the model. Arabic addresses are
structural -- a marker word, a name, a number, a governorate -- and a gazetteer
plus pattern gets better recall than a LOC tagger, is auditable, and is
extended by editing ``gazetteer_eg.yaml`` rather than by retraining.

Both restrict themselves to the Arabic runs reported by
``router.script_segments``, so neither ever scores Latin text.

MODEL ARTIFACT REQUIRED. The exact CAMeLBERT variant to quantize is one of the
three things the brief says to flag rather than guess, and it has not been
decided -- see the README. Until a model directory is provided,
``enable_tier2_arabic_ner`` must stay false; turning it on without one raises at
startup rather than silently running with reduced coverage.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import structlog
from presidio_analyzer import EntityRecognizer, Pattern, RecognizerResult

from pii_service.detect.context import normalize_context_terms
from pii_service.detect.normalize import GAZETTEER, normalize
from pii_service.detect.router import Script, script_segments
from pii_service.detect.tier1_patterns import ContextBoostedPatternRecognizer
from pii_service.policy.loader import PolicyBundle

if TYPE_CHECKING:
    from pii_service.settings import Settings

__all__ = [
    "ArabicNerRecognizer",
    "EgyptianAddressRecognizer",
    "Tier2Unavailable",
    "build_tier2_factory",
]

logger: Final = structlog.get_logger(__name__)

# CAMeLBERT NER label -> our entity type.
LABEL_MAP: Final[dict[str, str]] = {
    "PER": "AR_PERSON",
    "PERS": "AR_PERSON",
    "LOC": "AR_LOCATION",
    "ORG": "AR_ORG",
}

_NER_RECOGNIZER_NAME: Final = "ArabicNerRecognizer"
_MODEL_FILENAME: Final = "model.onnx"
_TOKENIZER_FILENAME: Final = "tokenizer.json"
_LABELS_FILENAME: Final = "labels.json"


class Tier2Unavailable(RuntimeError):
    """Tier 2 was enabled but its model artifact or runtime is missing."""


@dataclass(frozen=True, slots=True)
class _Entity:
    label: str
    start: int
    end: int
    score: float


class ArabicNerRecognizer(EntityRecognizer):
    """ONNX int8 Arabic NER, scoped to Arabic script runs."""

    def __init__(
        self,
        *,
        model_dir: Path,
        supported_language: str,
        supported_entities: Sequence[str] = tuple(dict.fromkeys(LABEL_MAP.values())),
        score_floor: float = 0.5,
    ) -> None:
        # Every attribute load() touches must exist BEFORE super().__init__(),
        # because Presidio's EntityRecognizer.__init__ calls self.load() itself.
        # Assigning them afterwards raises AttributeError on construction, which
        # is how tier 2 would have failed the moment anyone enabled it.
        self._model_dir: Final = model_dir
        self._score_floor: Final = score_floor
        self._session: object | None = None
        self._tokenizer: object | None = None
        self._labels: list[str] = []
        super().__init__(
            supported_entities=list(supported_entities),
            supported_language=supported_language,
            name=_NER_RECOGNIZER_NAME,
        )

    def load(self) -> None:
        """Load the ONNX session and tokenizer. Called once by Presidio."""
        try:
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise Tier2Unavailable(
                "tier 2 is enabled but its runtime is not installed. "
                "Install the 'ner' extra: pip install 'pii-service[ner]'."
            ) from exc

        model_path = self._model_dir / _MODEL_FILENAME
        tokenizer_path = self._model_dir / _TOKENIZER_FILENAME
        labels_path = self._model_dir / _LABELS_FILENAME

        missing = [p for p in (model_path, tokenizer_path, labels_path) if not p.is_file()]
        if missing:
            raise Tier2Unavailable(
                "tier 2 is enabled but its model artifact is incomplete; missing: "
                + ", ".join(str(p) for p in missing)
                + ". See the README section 'Tier 2 model artifact'."
            )

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1  # CPU-only service; avoid thread storms
        self._session = onnxruntime.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._labels = json.loads(labels_path.read_text(encoding="utf-8"))

        logger.info("tier2.loaded", model_dir=str(self._model_dir), labels=len(self._labels))

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: object | None = None,
    ) -> list[RecognizerResult]:
        if self._session is None or not text:
            return []

        results: list[RecognizerResult] = []
        for segment in script_segments(text):
            if segment.script != Script.ARABIC:
                continue
            chunk = text[segment.start : segment.end]
            for entity in self._infer(chunk):
                entity_type = LABEL_MAP.get(entity.label)
                if entity_type is None or entity_type not in entities:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=entity_type,
                        start=segment.start + entity.start,
                        end=segment.start + entity.end,
                        score=entity.score,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: _NER_RECOGNIZER_NAME,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                        },
                    )
                )
        return results

    def _infer(self, chunk: str) -> list[_Entity]:
        """Run the model over one Arabic run and decode BIO spans.

        Offsets come from the tokenizer's own ``offset_mapping``, which is why
        the model is fed the raw chunk rather than a normalized form: a second
        offset translation here would be a second place to get it wrong.
        """
        import numpy as np

        encoding = self._tokenizer.encode(chunk)  # type: ignore[union-attr]
        input_ids = np.array([encoding.ids], dtype=np.int64)
        attention_mask = np.array([encoding.attention_mask], dtype=np.int64)

        feeds = {"input_ids": input_ids, "attention_mask": attention_mask}
        expected = {i.name for i in self._session.get_inputs()}  # type: ignore[union-attr]
        if "token_type_ids" in expected:
            feeds["token_type_ids"] = np.zeros_like(input_ids)
        feeds = {name: value for name, value in feeds.items() if name in expected}

        logits: Any = self._session.run(None, feeds)[0][0]  # type: ignore[union-attr]
        scores: Any = _softmax(logits)
        predictions = scores.argmax(axis=-1)

        return _tidy_spans(
            _decode_bio(
                labels=[self._labels[int(p)] for p in predictions],
                scores=[float(scores[i, int(p)]) for i, p in enumerate(predictions)],
                offsets=encoding.offsets,
                score_floor=self._score_floor,
            ),
            chunk,
        )


def _is_word_char(char: str) -> bool:
    """A character that must not be split away from the rest of its word.

    Letters, plus the combining marks that sit on them -- leaving a stray
    fatha behind a placeholder is the same bug as leaving a letter.

    Digits are deliberately excluded: "عمارة 12" must not pull the building
    number into a name span.
    """
    return char.isalpha() or unicodedata.category(char) == "Mn"


def _tidy_spans(entities: list[_Entity], text: str) -> list[_Entity]:
    """Snap span edges out to word boundaries, then merge what now touches.

    The model is a WordPiece tokenizer over Arabic, so one orthographic word
    is usually several tokens and the BIO tags over them are not always
    consistent. Two failure modes come out of that, both observed on the
    sentence "أنا رايح لمنى في المعادي بكرة":

    * the final sub-word of المعادي tagged O, so the span stopped a
      character short and masking left a ي stranded after the placeholder --
      a fragment of the matched value surviving in what we forward;
    * لمنى tagged B-LOC twice rather than B-LOC I-LOC, so the decoder
      emitted two spans and the output read <AR_LOCATION><AR_LOCATION>.

    The evaluation harness could not catch either one. It scores *coverage* --
    the share of gold spans overlapped by any prediction -- so a span truncated
    by one character still counts as a pass. That is the right metric for
    deciding whether a model finds things; it says nothing about whether the
    masking built on top of it is clean.

    Merging is deliberately limited to spans that touch or overlap **after**
    snapping. Merging across whitespace would join two adjacent but
    distinct places into a single span, which over-masks.
    """
    if not entities:
        return []

    snapped: list[_Entity] = []
    for entity in entities:
        start, end = entity.start, entity.end
        while start > 0 and _is_word_char(text[start - 1]):
            start -= 1
        while end < len(text) and _is_word_char(text[end]):
            end += 1
        snapped.append(_Entity(label=entity.label, start=start, end=end, score=entity.score))

    snapped.sort(key=lambda entity: (entity.start, entity.end))

    merged: list[_Entity] = [snapped[0]]
    for entity in snapped[1:]:
        previous = merged[-1]
        # Same label and touching or overlapping. Different labels over the
        # same characters are left alone: router._deconflict resolves those by
        # length and score, and it does it for every recognizer at once.
        if entity.label == previous.label and entity.start <= previous.end:
            merged[-1] = _Entity(
                label=previous.label,
                start=previous.start,
                end=max(previous.end, entity.end),
                score=min(previous.score, entity.score),
            )
        else:
            merged.append(entity)

    return merged


def _softmax(logits: Any) -> Any:
    import numpy as np

    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exponentiated = np.exp(shifted)
    return exponentiated / np.sum(exponentiated, axis=-1, keepdims=True)


def _decode_bio(
    *,
    labels: list[str],
    scores: list[float],
    offsets: list[tuple[int, int]],
    score_floor: float,
) -> list[_Entity]:
    """Merge BIO tags into spans, scoring each span by its weakest token.

    Weakest rather than mean: a span is only as trustworthy as its least
    confident token, and averaging lets one certain token drag an otherwise
    doubtful span over the threshold.
    """
    entities: list[_Entity] = []
    current: _Entity | None = None

    for label, score, (start, end) in zip(labels, scores, offsets, strict=True):
        if start == end:  # special token
            continue

        tag, _, raw_label = label.partition("-")
        if tag not in ("B", "I") or not raw_label:
            if current is not None:
                entities.append(current)
                current = None
            continue

        if tag == "B" or current is None or current.label != raw_label:
            if current is not None:
                entities.append(current)
            current = _Entity(label=raw_label, start=start, end=end, score=score)
        else:
            current = _Entity(
                label=current.label,
                start=current.start,
                end=end,
                score=min(current.score, score),
            )

    if current is not None:
        entities.append(current)

    return [entity for entity in entities if entity.score >= score_floor]


class EgyptianAddressRecognizer(ContextBoostedPatternRecognizer):
    """Gazetteer + structural pattern for Egyptian addresses.

    Deliberately not the NER model. The pattern requires a structural marker
    (شارع, عمارة, شقة, ...) because a governorate name alone is a place, not an
    address -- "I live in Cairo" is not PII worth masking, "12 شارع X، المعادي,
    القاهرة" is.
    """

    ENTITY: Final = "EG_ADDRESS"

    def __init__(
        self,
        *,
        policy: PolicyBundle,
        supported_language: str,
    ) -> None:
        gazetteer = policy.gazetteer
        markers = gazetteer.address_markers
        places = tuple(gazetteer.governorate_names) + tuple(gazetteer.localities)

        marker_alternation = "|".join(_escape(m) for m in markers)
        place_alternation = "|".join(_escape(p) for p in sorted(places, key=len, reverse=True))

        # An optional house number, then a marker at a word boundary, then up
        # to ~6 words, optionally running on to a known place.
        #
        # The \b is load-bearing. `ش` is the near-universal abbreviation for
        # شارع and is a single letter, so without a boundary it matches the
        # final letter of any word ending in shin -- "أنا عايش في القاهرة"
        # ("I live in Cairo") was reported as an address, which is exactly the
        # non-PII sentence the brief says must not match.
        pattern = (
            rf"(?:\d{{1,4}}\s+)?"
            rf"\b(?:{marker_alternation})\s+"
            rf"[^\n،,.;]{{2,60}}"
            rf"(?:[،,]\s*(?:{place_alternation}))?"
        )

        super().__init__(
            supported_entity=self.ENTITY,
            name="EgyptianAddressRecognizer",
            supported_language=supported_language,
            patterns=[Pattern(name="eg_address_structural", regex=pattern, score=0.45)],
            context_terms=policy.context_terms.get("address", ()),
            context_window=48,
        )
        self._places: Final = normalize_context_terms(places)

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: object | None = None,
        regex_flags: int | None = None,
    ) -> list[RecognizerResult]:
        results = super().analyze(text, entities, nlp_artifacts, regex_flags)

        # A recognized governorate or locality inside the span is strong
        # corroboration that this is an address rather than a sentence that
        # happens to start with "شارع".
        for result in results:
            folded, _ = normalize(text[result.start : result.end], GAZETTEER)
            haystack = folded.casefold()
            if any(place in haystack for place in self._places):
                result.score = min(1.0, result.score + 0.2)

        return [r for r in results if _spans_arabic(text, r.start, r.end)]


def _spans_arabic(text: str, start: int, end: int) -> bool:
    return any(
        segment.script == Script.ARABIC and segment.start < end and start < segment.end
        for segment in script_segments(text)
    )


def _escape(term: str) -> str:
    import re

    return re.escape(term)


def build_tier2_factory(settings: Settings) -> object:
    """Return a factory the registry calls once per language.

    The address recognizer needs no model and is always included. The NER
    recognizer is included only when a model directory is configured, and
    raises if that directory is not usable -- a tier the operator switched on
    must not silently do nothing.
    """
    from pii_service.policy.loader import load_policy_bundle

    policy = load_policy_bundle(settings.config_dir)

    if settings.tier2_model_dir is None:
        raise Tier2Unavailable(
            "PII_ENABLE_TIER2_ARABIC_NER is true but PII_TIER2_MODEL_DIR is unset. "
            "Set it to a directory containing model.onnx, tokenizer.json and "
            "labels.json, or disable tier 2. See the README section "
            "'Tier 2 model artifact'."
        )

    model_dir = Path(settings.tier2_model_dir)

    def factory(language: str) -> list[EntityRecognizer]:
        recognizers: list[EntityRecognizer] = [
            EgyptianAddressRecognizer(policy=policy, supported_language=language)
        ]
        # Presidio's EntityRecognizer.__init__ calls load(), so constructing
        # this is what loads the ONNX session -- and what raises
        # Tier2Unavailable if the artifact is missing. No second load() call.
        recognizers.append(ArabicNerRecognizer(model_dir=model_dir, supported_language=language))
        return recognizers

    return factory
