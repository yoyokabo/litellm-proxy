"""Tier 3: GLiNER2 over Latin-script text, on ONNX Runtime.

Schema-conditioned: the labels travel in the forward pass as natural-language
prompts, so adding an entity type is a config row rather than a retrain. That
is what ``gliner_prompt`` in entities.yaml -- and the Entities screen -- are
for. Supports EN/FR/ES/DE/IT/PT/NL and **not Arabic**, which is why this
recognizer filters itself to the Latin runs from ``router.script_segments``
rather than trusting document-level language.

**No PyTorch.** The ``gliner`` package declares torch and transformers as core
dependencies and its own ONNX adapter imports torch anyway, so installing it
would put a multi-gigabyte training stack in a CPU inference image to run a
graph onnxruntime can execute alone. Instead ``scripts/export_gliner_onnx.py``
exports the graph at build time and this module reimplements the two ends in
numpy. Both ends were verified against the torch model's own output: identical
spans, scores matching to three decimals.

The model's contract, from the exported graph:

    input_ids       [1, T]            [CLS] <<ENT>> label.. <<ENT>> label.. <<SEP>> words.. [SEP]
    attention_mask  [1, T]
    words_mask      [1, T]            1-based word index on each word's FIRST sub-token, else 0
    text_lengths    [1, 1]            number of words
    span_idx        [1, W*max_width, 2]  (start_word, end_word) for every candidate span
    span_mask       [1, W*max_width]     which of those are in range
    -> logits       [1, W, max_width, num_labels]

Words come from GLiNER's own splitter, ``\\w+(?:[-_]\\w+)*|\\S``, which keeps
character offsets: a span is decoded back to ``(first_word.start,
last_word.end)`` in the original string, never re-derived from sub-tokens.

LATENCY, MEASURED. On an i7-10700K, one thread, fp32: p50 71 ms and p95 85 ms
on chat-sized text, and **1.4 s on a 1.7 kB document**. The brief (§4) called
the vendor's 50-200 ms and an independent ~2.3 s irreconcilable and said to
measure; chat-sized messages land at the good end, documents do not. This
runs ``pre_call``, so that cost is on every request carrying Latin text.
Default off, and the document figure is the one to design around.

Dynamic int8 was measured and is **not usable for this model**: mdeberta-v3's
disentangled attention does not survive it. The export's smoke check refuses a
degraded artifact.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import structlog
from presidio_analyzer import EntityRecognizer, RecognizerResult

from pii_service.detect.router import Script, script_segments

if TYPE_CHECKING:
    from pii_service.policy.loader import PolicyBundle
    from pii_service.settings import Settings

__all__ = ["GlinerRecognizer", "Tier3Unavailable", "build_tier3_factory"]

logger: Final = structlog.get_logger(__name__)

# Fallback prompts, used only when a caller builds the recognizer without a
# policy. The real source is ``PolicyBundle.gliner_prompts``.
LABEL_PROMPTS: Final[dict[str, str]] = {
    "PERSON": "person name",
    "LOCATION": "location",
    "ORGANIZATION": "organization",
}

_RECOGNIZER_NAME: Final = "GlinerRecognizer"
_MODEL_FILENAME: Final = "model.onnx"
_TOKENIZER_FILENAME: Final = "tokenizer.json"
_CONFIG_FILENAME: Final = "gliner.json"

# GLiNER's own word splitter. Reproduced rather than imported, because
# importing it would pull in the package this module exists to avoid.
_WORD_RE: Final = re.compile(r"\w+(?:[-_]\w+)*|\S")

# Latin runs shorter than this are punctuation and stray words between Arabic
# clauses; running a transformer over them is pure cost.
MIN_SEGMENT_CHARS: Final = 12

# The graph accepts 384 sub-tokens. Windowing is by words, conservatively, so
# a long paste is chunked instead of silently truncated -- a truncated document
# is undetected PII with nothing in the logs. The overlap is max_width words so
# that every candidate span sits wholly inside at least one window.
_WINDOW_WORDS: Final = 160


class Tier3Unavailable(RuntimeError):
    """Tier 3 was enabled but its model artifact or runtime is missing."""


@dataclass(frozen=True, slots=True)
class _Word:
    text: str
    start: int
    end: int


@dataclass(frozen=True, slots=True)
class _Span:
    label: str
    start: int
    end: int
    score: float


def _split_words(text: str) -> list[_Word]:
    return [_Word(m.group(), m.start(), m.end()) for m in _WORD_RE.finditer(text)]


class GlinerRecognizer(EntityRecognizer):
    """GLiNER2 over Latin-script runs, via onnxruntime."""

    def __init__(
        self,
        *,
        model_dir: Path,
        supported_language: str,
        supported_entities: Sequence[str] | None = None,
        prompts: Mapping[str, str] | None = None,
        threshold: float = 0.5,
    ) -> None:
        # Every attribute load() touches must exist BEFORE super().__init__(),
        # because Presidio's EntityRecognizer.__init__ calls self.load().
        label_prompts = dict(prompts if prompts is not None else LABEL_PROMPTS)
        if supported_entities is None:
            supported_entities = tuple(label_prompts)

        self._model_dir: Final = Path(model_dir)
        self._threshold: Final = threshold
        self._session: Any | None = None
        self._tokenizer: Any | None = None
        self._config: dict[str, int] = {}
        # Entity order is fixed here: it is the order the prompts are written
        # into the input, and therefore the order of the logits' last axis.
        self._entities: list[str] = [e for e in supported_entities if e in label_prompts]
        self._prompts: list[str] = [label_prompts[e] for e in self._entities]

        super().__init__(
            supported_entities=list(self._entities),
            supported_language=supported_language,
            name=_RECOGNIZER_NAME,
        )

    def load(self) -> None:
        try:
            import onnxruntime
            from tokenizers import Tokenizer
        except ImportError as exc:
            raise Tier3Unavailable(
                "tier 3 is enabled but its runtime is not installed. "
                "Install the 'ner' extra: pip install 'pii-service[ner]'."
            ) from exc

        model_path = self._model_dir / _MODEL_FILENAME
        tokenizer_path = self._model_dir / _TOKENIZER_FILENAME
        config_path = self._model_dir / _CONFIG_FILENAME

        missing = [p for p in (model_path, tokenizer_path, config_path) if not p.is_file()]
        if missing:
            raise Tier3Unavailable(
                "tier 3 is enabled but its model artifact is incomplete; missing: "
                + ", ".join(str(p) for p in missing)
                + ". Run scripts/export_gliner_onnx.py on a machine with internet."
            )

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1  # CPU-only service; avoid thread storms
        self._session = onnxruntime.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._config = json.loads(config_path.read_text(encoding="utf-8"))

        logger.info(
            "tier3.loaded",
            model_dir=str(self._model_dir),
            labels=len(self._entities),
            max_width=self._config.get("max_width"),
        )

    def analyze(
        self,
        text: str,
        entities: list[str],
        nlp_artifacts: object | None = None,
    ) -> list[RecognizerResult]:
        if self._session is None or not text or not self._entities:
            return []

        wanted = [entity for entity in self._entities if entity in entities]
        if not wanted:
            return []

        results: list[RecognizerResult] = []
        for segment in script_segments(text, min_length=MIN_SEGMENT_CHARS):
            if segment.script != Script.LATIN:
                continue
            chunk = text[segment.start : segment.end]
            for span in self._infer(chunk):
                if span.label not in wanted:
                    continue
                results.append(
                    RecognizerResult(
                        entity_type=span.label,
                        start=segment.start + span.start,
                        end=segment.start + span.end,
                        score=span.score,
                        recognition_metadata={
                            RecognizerResult.RECOGNIZER_NAME_KEY: _RECOGNIZER_NAME,
                            RecognizerResult.RECOGNIZER_IDENTIFIER_KEY: self.id,
                        },
                    )
                )
        return results

    def set_prompts(self, prompts: Mapping[str, str]) -> None:
        """Swap the label set without reloading the model.

        This is the whole point of schema conditioning: labels are prompts in
        the forward pass, not trained classes, so an administrator adding an
        entity type costs a re-prompt rather than a re-export. ``main.py``
        subscribes this to the policy store.

        Order matters and is set here. The prompts are written into the input
        in this order, and the last axis of the logits is in the same order, so
        the two lists must be rebuilt together or every label shifts by one.
        """
        entities = [entity for entity in prompts if prompts[entity]]
        self._entities = entities
        self._prompts = [prompts[entity] for entity in entities]
        self.supported_entities = list(entities)
        logger.info("tier3.reprompted", labels=len(entities))

    # -- inference ---------------------------------------------------------

    def _encode(self, words: Sequence[_Word]) -> dict[str, Any]:
        import numpy as np

        config = self._config
        ids: list[int] = [config["cls_token_id"]]
        words_mask: list[int] = [0]

        for prompt in self._prompts:
            ids.append(config["ent_token_id"])
            words_mask.append(0)
            for piece in self._tokenizer.encode(prompt, add_special_tokens=False).ids:  # type: ignore[union-attr]
                ids.append(piece)
                words_mask.append(0)

        ids.append(config["sep_token_id"])
        words_mask.append(0)

        for index, word in enumerate(words, start=1):
            pieces = self._tokenizer.encode(word.text, add_special_tokens=False).ids  # type: ignore[union-attr]
            for position, piece in enumerate(pieces):
                ids.append(piece)
                # Only the first sub-token carries the word index. That is how
                # the model gathers one vector per word out of the sequence.
                words_mask.append(index if position == 0 else 0)

        ids.append(config["eos_token_id"])
        words_mask.append(0)

        count = len(words)
        width = config["max_width"]
        span_idx = [[i, i + w] for i in range(count) for w in range(width)]
        span_mask = [(i + w) < count for i in range(count) for w in range(width)]

        return {
            "input_ids": np.array([ids], dtype=np.int64),
            "attention_mask": np.ones((1, len(ids)), dtype=np.int64),
            "words_mask": np.array([words_mask], dtype=np.int64),
            "text_lengths": np.array([[count]], dtype=np.int64),
            "span_idx": np.array([span_idx], dtype=np.int64),
            "span_mask": np.array([span_mask], dtype=bool),
        }

    def _infer(self, chunk: str) -> list[_Span]:
        import numpy as np

        words = _split_words(chunk)
        if not words:
            return []

        width = self._config["max_width"]
        stride = max(1, _WINDOW_WORDS - width)
        spans: list[_Span] = []

        for offset in range(0, len(words), stride):
            window = words[offset : offset + _WINDOW_WORDS]
            if not window:
                break
            feeds = self._encode(window)
            logits = self._session.run(None, feeds)[0]  # type: ignore[union-attr]
            probabilities = 1.0 / (1.0 + np.exp(-logits[0]))

            count = len(window)
            for start_index in range(min(probabilities.shape[0], count)):
                for offset_width in range(min(probabilities.shape[1], width)):
                    end_index = start_index + offset_width
                    if end_index >= count:
                        break
                    for label_index, entity in enumerate(self._entities):
                        score = float(probabilities[start_index, offset_width, label_index])
                        if score >= self._threshold:
                            spans.append(
                                _Span(
                                    label=entity,
                                    start=window[start_index].start,
                                    end=window[end_index].end,
                                    score=score,
                                )
                            )
            if offset + _WINDOW_WORDS >= len(words):
                break

        return _greedy_flat(spans)

    def _infer_words(self, chunk: str) -> list[_Word]:
        """Exposed for tests: the word split a span's offsets are derived from."""
        return _split_words(chunk)


def _greedy_flat(spans: list[_Span]) -> list[_Span]:
    """Highest score first, keeping only non-overlapping spans.

    Flat NER, matching GLiNER's own default. Windowing can propose the same
    span twice with slightly different scores; taking the strongest first and
    rejecting anything that overlaps it de-duplicates as a side effect.
    """
    kept: list[_Span] = []
    for span in sorted(spans, key=lambda s: -s.score):
        if all(span.end <= other.start or span.start >= other.end for other in kept):
            kept.append(span)
    return sorted(kept, key=lambda s: s.start)


def build_tier3_factory(
    settings: Settings,
    *,
    model_dir: Path | None = None,
    policy: PolicyBundle | None = None,
) -> object:
    """Build the tier-3 factory, conditioned on the policy's labels.

    ``policy`` supplies ``gliner_prompts``: the baseline labels from
    entities.yaml plus anything an administrator added at runtime.
    """
    prompts = dict(policy.gliner_prompts) if policy is not None else dict(LABEL_PROMPTS)
    directory = model_dir or settings.tier3_model_dir
    if directory is None:
        raise Tier3Unavailable(
            "tier 3 is enabled but PII_TIER3_MODEL_DIR is unset. Run "
            "scripts/export_gliner_onnx.py and point it at the output."
        )

    def factory(language: str) -> list[EntityRecognizer]:
        # Constructing it loads the model: EntityRecognizer.__init__ calls
        # load(), which raises Tier3Unavailable when the artifact is missing.
        return [
            GlinerRecognizer(
                model_dir=Path(directory), supported_language=language, prompts=prompts
            )
        ]

    return factory
