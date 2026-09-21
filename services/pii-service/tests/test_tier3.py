"""Tier 3: GLiNER2 over Latin-script runs.

The gliner wheel is not installed (the tier is off by default and its latency
is unresolved), so these tests inject a stub in place of the loaded model. That
still exercises everything this module is actually responsible for: which runs
get sent to the model, how labels map back to our entity types, and how offsets
are rebased onto the full text. The model's own accuracy is not ours to test.
"""

from __future__ import annotations

from typing import Any

import pytest

from pii_service.detect.router import Script, script_segments
from pii_service.detect.tier3_gliner import (
    LABEL_PROMPTS,
    MIN_SEGMENT_CHARS,
    GlinerRecognizer,
    Tier3Unavailable,
)


class _StubModel:
    """Records what it was asked and returns canned predictions."""

    def __init__(self, predictions: list[dict[str, Any]] | None = None) -> None:
        self.predictions = predictions or []
        self.calls: list[tuple[str, list[str]]] = []

    def predict_entities(
        self, text: str, labels: list[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        self.calls.append((text, labels))
        return [p for p in self.predictions if p["_chunk"] == text]


def _recognizer(model: _StubModel | None) -> GlinerRecognizer:
    """Build without triggering load(), then attach the stub."""
    recognizer = GlinerRecognizer.__new__(GlinerRecognizer)
    recognizer._model_name = "stub"
    recognizer._threshold = 0.5
    recognizer._model = model
    # Normally set by EntityRecognizer.__init__, which we bypass here.
    recognizer._id = "stub-gliner"
    recognizer.name = "GlinerRecognizer"
    recognizer._prompt_to_entity = {prompt: entity for entity, prompt in LABEL_PROMPTS.items()}
    recognizer.supported_entities = list(LABEL_PROMPTS)
    return recognizer


ALL_ENTITIES = list(LABEL_PROMPTS)


# ---------------------------------------------------------------------------
# Script scoping
# ---------------------------------------------------------------------------


def test_arabic_runs_are_never_sent_to_the_model() -> None:
    """GLiNER2 does not support Arabic; fed it anyway it invents entities."""
    model = _StubModel()
    recognizer = _recognizer(model)

    recognizer.analyze("محمد علي يسكن في القاهرة ويعمل هناك", ALL_ENTITIES)

    assert model.calls == []


def test_latin_runs_are_sent() -> None:
    model = _StubModel()
    recognizer = _recognizer(model)

    recognizer.analyze("Please contact Jane Doe about the contract", ALL_ENTITIES)

    assert len(model.calls) == 1
    assert "Jane Doe" in model.calls[0][0]


def test_only_the_latin_run_of_a_mixed_message_is_sent() -> None:
    model = _StubModel()
    recognizer = _recognizer(model)
    text = "محمد علي وايضا هنا Please contact Jane Doe today"

    recognizer.analyze(text, ALL_ENTITIES)

    assert len(model.calls) == 1
    sent = model.calls[0][0]
    assert "Jane Doe" in sent
    assert "محمد" not in sent


def test_short_latin_runs_are_skipped() -> None:
    """A stray word between Arabic clauses is not worth a transformer pass."""
    model = _StubModel()
    recognizer = _recognizer(model)

    recognizer.analyze("محمد علي ok وايضا هنا", ALL_ENTITIES)

    assert model.calls == []


def test_min_segment_length_is_the_documented_threshold() -> None:
    text = "x" * MIN_SEGMENT_CHARS
    assert [s.script for s in script_segments(text, min_length=MIN_SEGMENT_CHARS)] == [Script.LATIN]


# ---------------------------------------------------------------------------
# Label mapping and offsets
# ---------------------------------------------------------------------------


def test_predictions_map_to_our_entity_types() -> None:
    chunk = "Please contact Jane Doe about the contract"
    model = _StubModel(
        [{"_chunk": chunk, "label": "person name", "start": 15, "end": 23, "score": 0.9}]
    )
    results = _recognizer(model).analyze(chunk, ALL_ENTITIES)

    assert len(results) == 1
    assert results[0].entity_type == "PERSON"
    assert chunk[results[0].start : results[0].end] == "Jane Doe"


def test_offsets_are_rebased_onto_the_full_text() -> None:
    """A segment offset that is not added back masks the wrong characters."""
    prefix = "محمد علي وايضا هنا "
    latin = "Please contact Jane Doe today"
    text = prefix + latin

    model = _StubModel(
        [{"_chunk": latin, "label": "person name", "start": 15, "end": 23, "score": 0.9}]
    )
    results = _recognizer(model).analyze(text, ALL_ENTITIES)

    assert len(results) == 1
    assert text[results[0].start : results[0].end] == "Jane Doe"


def test_an_unknown_label_is_ignored_rather_than_guessed() -> None:
    chunk = "Please contact Jane Doe about the contract"
    model = _StubModel(
        [{"_chunk": chunk, "label": "shoe size", "start": 15, "end": 23, "score": 0.9}]
    )
    assert _recognizer(model).analyze(chunk, ALL_ENTITIES) == []


def test_only_requested_entities_are_prompted_for() -> None:
    model = _StubModel()
    _recognizer(model).analyze("Please contact Jane Doe today", ["PERSON"])

    assert model.calls[0][1] == ["person name"]


def test_no_requested_entities_means_no_model_call() -> None:
    model = _StubModel()
    _recognizer(model).analyze("Please contact Jane Doe today", ["EG_NATIONAL_ID"])
    assert model.calls == []


def test_score_is_carried_through() -> None:
    chunk = "Please contact Jane Doe about the contract"
    model = _StubModel(
        [{"_chunk": chunk, "label": "person name", "start": 15, "end": 23, "score": 0.77}]
    )
    assert _recognizer(model).analyze(chunk, ALL_ENTITIES)[0].score == pytest.approx(0.77)


def test_recognizer_name_is_recorded_for_the_audit_row() -> None:
    chunk = "Please contact Jane Doe about the contract"
    model = _StubModel(
        [{"_chunk": chunk, "label": "person name", "start": 15, "end": 23, "score": 0.9}]
    )
    metadata = _recognizer(model).analyze(chunk, ALL_ENTITIES)[0].recognition_metadata
    assert metadata["recognizer_name"] == "GlinerRecognizer"


# ---------------------------------------------------------------------------
# Degenerate input and availability
# ---------------------------------------------------------------------------


def test_empty_text_and_no_model_are_both_inert() -> None:
    assert _recognizer(_StubModel()).analyze("", ALL_ENTITIES) == []
    assert _recognizer(None).analyze("Please contact Jane Doe today", ALL_ENTITIES) == []


def test_enabling_tier3_without_the_wheel_raises() -> None:
    """A tier switched on must fail loudly, not detect nothing.

    Only exercisable in an image built without the tier-3 extra, which is the
    default. Where the extra IS installed this skips rather than pretending.
    """
    try:
        import gliner  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("gliner is installed; the unavailable path cannot be exercised")

    with pytest.raises(Tier3Unavailable, match="gliner is not installed"):
        GlinerRecognizer(model_name="whatever", supported_language="en")


def test_set_prompts_swaps_labels_without_reloading_the_model() -> None:
    """Regression: main.py subscribed this to the policy store before it existed.

    The subscription filters on hasattr(), so a missing method meant runtime
    label changes silently never reached the model. Tier 3 being off by
    default meant nobody could have noticed.
    """
    recognizer = _recognizer(_StubModel())
    recognizer.supported_entities = list(LABEL_PROMPTS)

    recognizer.set_prompts({"PERSON": "person name", "PROJECT_CODENAME": "project codename"})

    assert recognizer.supported_entities == ["PERSON", "PROJECT_CODENAME"]
    # The reverse map is what a prediction's label is looked up in.
    assert recognizer._prompt_to_entity == {
        "person name": "PERSON",
        "project codename": "PROJECT_CODENAME",
    }


def test_set_prompts_ignores_an_entity_with_no_prompt() -> None:
    """An entity nothing can detect is policy that silently does nothing."""
    recognizer = _recognizer(_StubModel())
    recognizer.set_prompts({"PERSON": "person name", "NO_PROMPT": ""})

    assert recognizer.supported_entities == ["PERSON"]


def test_label_prompts_are_natural_language_not_identifiers() -> None:
    """Wording changes recall, so a change here is a model change."""
    assert LABEL_PROMPTS["PERSON"] == "person name"
    assert LABEL_PROMPTS["LOCATION"] == "location"
    assert LABEL_PROMPTS["ORGANIZATION"] == "organization"
