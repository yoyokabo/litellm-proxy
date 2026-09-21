"""Tier 3: GLiNER2 over Latin-script runs, on ONNX Runtime.

Two layers, deliberately separated.

The stubbed tests replace ``_infer`` -- the one method that touches the model --
and exercise everything this module is actually responsible for: which runs get
sent, how labels map to our entity types, and how offsets are rebased onto the
full text. They run everywhere and need no artifact.

The artifact tests at the bottom load the real exported graph and assert on
real predictions. They skip unless ``models/gliner-multi-pii`` is present,
because it is a 1.2 GB build output and not something CI should download. They
are what proves the numpy reimplementation of GLiNER's pre- and
post-processing still matches the model it was verified against.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_service.detect.router import Script, script_segments
from pii_service.detect.tier3_gliner import (
    LABEL_PROMPTS,
    MIN_SEGMENT_CHARS,
    GlinerRecognizer,
    Tier3Unavailable,
    _greedy_flat,
    _Span,
    _split_words,
)

ALL_ENTITIES = list(LABEL_PROMPTS)

MODEL_DIR = Path(__file__).resolve().parents[3] / "models" / "gliner-multi-pii"
_artifact = pytest.mark.skipif(
    not (MODEL_DIR / "model.onnx").is_file(),
    reason="run scripts/export_gliner_onnx.py to exercise the real graph",
)


class _StubRecognizer(GlinerRecognizer):
    """Real routing and offset logic, canned inference."""

    def __init__(self, spans_by_chunk: dict[str, list[_Span]] | None = None) -> None:
        self.chunks: list[str] = []
        self._spans_by_chunk = spans_by_chunk or {}
        self._model_dir = Path("stub")
        self._threshold = 0.5
        self._session = object()  # not None, so analyze() proceeds
        self._tokenizer = None
        self._config = {"max_width": 12}
        self._entities = list(LABEL_PROMPTS)
        self._prompts = [LABEL_PROMPTS[e] for e in self._entities]
        self._id = "stub-gliner"
        self.name = "GlinerRecognizer"
        self.supported_entities = list(self._entities)

    def _infer(self, chunk: str) -> list[_Span]:
        self.chunks.append(chunk)
        return self._spans_by_chunk.get(chunk, [])


# ---------------------------------------------------------------------------
# Script scoping -- GLiNER does not support Arabic
# ---------------------------------------------------------------------------


def test_arabic_runs_are_never_sent_to_the_model() -> None:
    """Feeding it Arabic produces confident nonsense rather than an error."""
    recognizer = _StubRecognizer()
    recognizer.analyze("محمد علي يسكن في القاهرة ويعمل هناك", ALL_ENTITIES)
    assert recognizer.chunks == []


def test_only_the_latin_run_of_a_mixed_message_is_sent() -> None:
    text = "محمد علي يسكن في القاهرة and works at Contoso Limited"
    recognizer = _StubRecognizer()
    recognizer.analyze(text, ALL_ENTITIES)

    assert recognizer.chunks
    assert all("محمد" not in chunk for chunk in recognizer.chunks)
    assert any("Contoso" in chunk for chunk in recognizer.chunks)


def test_short_latin_runs_are_skipped() -> None:
    """Punctuation and stray words between Arabic clauses are not worth a pass."""
    recognizer = _StubRecognizer()
    recognizer.analyze("القاهرة ok القاهرة", ALL_ENTITIES)
    assert recognizer.chunks == []


def test_min_segment_length_is_the_documented_threshold() -> None:
    long_enough = "x" * MIN_SEGMENT_CHARS
    assert any(
        segment.script == Script.LATIN
        for segment in script_segments(long_enough, min_length=MIN_SEGMENT_CHARS)
    )


# ---------------------------------------------------------------------------
# Mapping back
# ---------------------------------------------------------------------------


def test_offsets_are_rebased_onto_the_full_text() -> None:
    """A span's offsets are into the segment; findings must be into the text."""
    prefix = "القاهرة القاهرة "
    chunk = "Call Jane Doe tomorrow"
    text = prefix + chunk

    recognizer = _StubRecognizer({chunk: [_Span(label="PERSON", start=5, end=13, score=0.9)]})
    results = recognizer.analyze(text, ALL_ENTITIES)

    assert len(results) == 1
    assert text[results[0].start : results[0].end] == "Jane Doe"


def test_only_requested_entities_are_returned() -> None:
    chunk = "Call Jane Doe tomorrow"
    recognizer = _StubRecognizer({chunk: [_Span(label="PERSON", start=5, end=13, score=0.9)]})

    assert recognizer.analyze(chunk, ["LOCATION"]) == []
    assert len(recognizer.analyze(chunk, ["PERSON"])) == 1


def test_no_requested_entities_means_no_model_call() -> None:
    recognizer = _StubRecognizer()
    recognizer.analyze("Call Jane Doe tomorrow", ["EG_NATIONAL_ID"])
    assert recognizer.chunks == []


def test_recognizer_name_is_recorded_for_the_audit_row() -> None:
    chunk = "Call Jane Doe tomorrow"
    recognizer = _StubRecognizer({chunk: [_Span(label="PERSON", start=5, end=13, score=0.9)]})
    result = recognizer.analyze(chunk, ALL_ENTITIES)[0]

    assert result.recognition_metadata["recognizer_name"] == "GlinerRecognizer"
    assert result.score == pytest.approx(0.9)


def test_empty_text_and_no_session_are_both_inert() -> None:
    assert _StubRecognizer().analyze("", ALL_ENTITIES) == []
    recognizer = _StubRecognizer()
    recognizer._session = None
    assert recognizer.analyze("Please contact Jane Doe today", ALL_ENTITIES) == []


# ---------------------------------------------------------------------------
# The pieces reimplemented from gliner, which is where a silent break would be
# ---------------------------------------------------------------------------


def test_word_splitter_matches_gliners_own_regex() -> None:
    """Punctuation is its own word. Getting this wrong shifts every offset.

    The exported model was traced with this split; "Berlin." is two words, not
    one, and a span's character offsets come from the word list.
    """
    words = _split_words("Contact Sarah Mitchell at Acme Corp in Berlin.")

    assert [w.text for w in words] == [
        "Contact",
        "Sarah",
        "Mitchell",
        "at",
        "Acme",
        "Corp",
        "in",
        "Berlin",
        ".",
    ]
    assert (words[1].start, words[1].end) == (8, 13)
    # Hyphenated and underscored words stay whole.
    assert [w.text for w in _split_words("well-known x_y")] == ["well-known", "x_y"]


def test_greedy_flat_keeps_the_strongest_of_overlapping_spans() -> None:
    kept = _greedy_flat(
        [
            _Span(label="PERSON", start=0, end=10, score=0.6),
            _Span(label="ORGANIZATION", start=5, end=15, score=0.9),
            _Span(label="LOCATION", start=20, end=25, score=0.7),
        ]
    )

    assert [(s.label, s.start) for s in kept] == [("ORGANIZATION", 5), ("LOCATION", 20)]


def test_greedy_flat_deduplicates_a_span_proposed_by_two_windows() -> None:
    """Overlapping windows re-propose spans; the stronger one wins silently."""
    kept = _greedy_flat(
        [
            _Span(label="PERSON", start=3, end=9, score=0.81),
            _Span(label="PERSON", start=3, end=9, score=0.80),
        ]
    )

    assert len(kept) == 1
    assert kept[0].score == pytest.approx(0.81)


def test_set_prompts_rebuilds_labels_and_prompts_together() -> None:
    """They index the same axis of the logits; drifting apart shifts every label."""
    recognizer = _StubRecognizer()
    recognizer.set_prompts({"PERSON": "person name", "PROJECT_CODENAME": "project codename"})

    assert recognizer._entities == ["PERSON", "PROJECT_CODENAME"]
    assert recognizer._prompts == ["person name", "project codename"]
    assert recognizer.supported_entities == ["PERSON", "PROJECT_CODENAME"]


def test_a_missing_artifact_raises_rather_than_detecting_nothing() -> None:
    with pytest.raises(Tier3Unavailable, match="incomplete"):
        GlinerRecognizer(model_dir=Path("/nonexistent/gliner"), supported_language="en")


def test_label_prompts_are_natural_language_not_identifiers() -> None:
    """Wording changes recall, so a change here is a model change."""
    assert LABEL_PROMPTS["PERSON"] == "person name"
    assert LABEL_PROMPTS["LOCATION"] == "location"
    assert LABEL_PROMPTS["ORGANIZATION"] == "organization"


# ---------------------------------------------------------------------------
# The real graph
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def real_recognizer() -> GlinerRecognizer:
    return GlinerRecognizer(model_dir=MODEL_DIR, supported_language="en")


@_artifact
def test_the_onnx_path_reproduces_the_torch_reference(
    real_recognizer: GlinerRecognizer,
) -> None:
    """The numbers this pins are the torch model's own, to four decimals.

    Measured from PyTorch with the same label order this recognizer uses, then
    reproduced by the numpy reimplementation from the exported graph. If this
    drifts, the pre- or post-processing has broken, not the model.
    """
    text = "Contact Sarah Mitchell at Acme Corp in Berlin."
    found = {
        r.entity_type: (text[r.start : r.end], r.score)
        for r in real_recognizer.analyze(text, ALL_ENTITIES)
    }

    assert found["PERSON"][0] == "Sarah Mitchell"
    assert found["PERSON"][1] == pytest.approx(0.7056, abs=0.005)
    assert found["ORGANIZATION"][0] == "Acme Corp"
    assert found["ORGANIZATION"][1] == pytest.approx(0.9981, abs=0.005)
    assert found["LOCATION"][0] == "Berlin"
    assert found["LOCATION"][1] == pytest.approx(0.9823, abs=0.005)


@_artifact
def test_scores_depend_on_which_labels_were_asked_for(
    real_recognizer: GlinerRecognizer,
) -> None:
    """A property of schema conditioning, and an operational trap.

    The labels are written into the input, so the label *set and its order*
    change the scores of every entity. Asking for the same sentence with
    person/organization/location scores Sarah Mitchell at 0.757; with
    person/location/organization it scores 0.706. Both were confirmed against
    PyTorch.

    That matters because thresholds live in entities.yaml: adding an entity
    type through the Entities screen moves the scores of the ones already
    there. It is not large, and it is not zero.
    """
    text = "Contact Sarah Mitchell at Acme Corp in Berlin."
    baseline = {r.entity_type: r.score for r in real_recognizer.analyze(text, ALL_ENTITIES)}

    reordered = GlinerRecognizer(
        model_dir=MODEL_DIR,
        supported_language="en",
        prompts={
            "PERSON": "person name",
            "ORGANIZATION": "organization",
            "LOCATION": "location",
        },
    )
    shifted = {r.entity_type: r.score for r in reordered.analyze(text, ALL_ENTITIES)}

    assert shifted["PERSON"] == pytest.approx(0.7568, abs=0.005)
    assert shifted["PERSON"] != pytest.approx(baseline["PERSON"], abs=0.005)


@_artifact
def test_arabic_is_still_never_sent_to_the_real_model(
    real_recognizer: GlinerRecognizer,
) -> None:
    assert real_recognizer.analyze("محمد علي يسكن في القاهرة ويعمل هناك", ALL_ENTITIES) == []


@_artifact
def test_a_long_document_is_windowed_not_truncated(
    real_recognizer: GlinerRecognizer,
) -> None:
    """The graph takes 384 sub-tokens; a longer paste must still be scanned.

    Truncation here would be undetected PII with nothing in the logs, so the
    entity is planted well past the first window on purpose.

    Worth recording: the gliner package itself finds *nothing* in this text --
    it truncates to the model's limit and the tail is never seen. Windowing is
    the reason this passes, and it is behaviour this module adds rather than
    reproduces.
    """
    filler = "The quarterly report covers the period and the usual operating costs. "
    text = filler * 40 + "Please forward it to Ingrid Bergqvist at Northwind Traders."
    found = real_recognizer.analyze(text, ALL_ENTITIES)

    surfaces = {text[r.start : r.end] for r in found}
    assert "Northwind Traders" in surfaces, surfaces
