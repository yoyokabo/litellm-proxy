"""Tier 2: the Egyptian address recognizer and the NER BIO decoder.

The address recognizer needs no model, so it is tested for real. The NER
recognizer's ONNX session cannot be exercised without a model artifact, but its
BIO decoding is a pure function and carries most of the logic that could
silently mis-span an entity, so that is tested directly.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pii_service.detect.tier2_arabic_ner import (
    LABEL_MAP,
    ArabicNerRecognizer,
    EgyptianAddressRecognizer,
    Tier2Unavailable,
    _decode_bio,
    build_tier2_factory,
)
from pii_service.policy.loader import PolicyBundle


@pytest.fixture(scope="module")
def address_recognizer(request: pytest.FixtureRequest) -> EgyptianAddressRecognizer:
    from conftest import CONFIG_DIR
    from pii_service.policy.loader import load_policy_bundle

    return EgyptianAddressRecognizer(policy=load_policy_bundle(CONFIG_DIR), supported_language="ar")


def _spans(recognizer: EgyptianAddressRecognizer, text: str) -> list[str]:
    return [text[r.start : r.end] for r in recognizer.analyze(text, ["EG_ADDRESS"])]


# ---------------------------------------------------------------------------
# Addresses that must be found
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected_fragment"),
    [
        ("العنوان ١٢ شارع الهرم، الجيزة", "شارع الهرم"),
        ("ساكن في عمارة 5 الدور الثالث شقة 12، المعادي", "عمارة 5"),
        ("العنوان ٧ ش الجمهورية، القاهرة", "ش الجمهورية"),
        ("العنوان: برج النيل الدور 7", "برج النيل"),
    ],
)
def test_addresses_are_detected(
    address_recognizer: EgyptianAddressRecognizer, text: str, expected_fragment: str
) -> None:
    found = _spans(address_recognizer, text)
    assert found, f"no address found in {text!r}"
    assert any(expected_fragment in span for span in found)


def test_a_leading_house_number_is_included_in_the_span() -> None:
    from conftest import CONFIG_DIR
    from pii_service.policy.loader import load_policy_bundle

    recognizer = EgyptianAddressRecognizer(
        policy=load_policy_bundle(CONFIG_DIR), supported_language="ar"
    )
    text = "العنوان ١٢ شارع الهرم، الجيزة"
    found = _spans(recognizer, text)
    assert found[0].startswith("١٢")


def test_a_known_governorate_in_the_span_raises_the_score(
    address_recognizer: EgyptianAddressRecognizer,
) -> None:
    with_place = address_recognizer.analyze("١٢ شارع الهرم، الجيزة", ["EG_ADDRESS"])
    without_place = address_recognizer.analyze("١٢ شارع النصر الجديد", ["EG_ADDRESS"])

    assert with_place and without_place
    assert with_place[0].score > without_place[0].score


# ---------------------------------------------------------------------------
# Things that must NOT be addresses
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        # The regression that prompted the word-boundary fix: `ش` is the
        # abbreviation for شارع, and without \b it matched the final letter of
        # any word ending in shin.
        "أنا عايش في القاهرة",
        "هو عايش هناك",
        "المعاش الشهري",
        # A place name alone is a place, not an address.
        "أنا من القاهرة",
        # An area word is an ordinary noun: "Cairo is a big city" is not an
        # address, and neither is a district name with no street or building.
        "القاهرة مدينة كبيرة",
        "مدينة نصر، الحي العاشر",
        "أسكن في منطقة هادئة",
        "القاهرة مدينة كبيرة",
        # A bare marker with nothing after it.
        "شارع",
        "عمارة",
        # Latin script is tier 3's problem, never this recognizer's.
        "I live in Cairo",
        "123 Main Street, Springfield",
        "",
    ],
)
def test_non_addresses_are_not_reported(
    address_recognizer: EgyptianAddressRecognizer, text: str
) -> None:
    assert _spans(address_recognizer, text) == []


def test_spans_stay_inside_the_arabic_run(
    address_recognizer: EgyptianAddressRecognizer,
) -> None:
    text = "Ship it to ١٢ شارع الهرم، الجيزة please"
    for result in address_recognizer.analyze(text, ["EG_ADDRESS"]):
        assert 0 <= result.start < result.end <= len(text)


def test_recognizer_declares_its_entity(
    address_recognizer: EgyptianAddressRecognizer,
) -> None:
    assert address_recognizer.supported_entities == ["EG_ADDRESS"]


# ---------------------------------------------------------------------------
# BIO decoding
# ---------------------------------------------------------------------------


def test_decode_single_token_entity() -> None:
    entities = _decode_bio(
        labels=["O", "B-PER", "O"],
        scores=[0.9, 0.95, 0.9],
        offsets=[(0, 0), (5, 10), (11, 15)],
        score_floor=0.5,
    )
    assert len(entities) == 1
    assert (entities[0].label, entities[0].start, entities[0].end) == ("PER", 5, 10)


def test_decode_merges_b_then_i() -> None:
    entities = _decode_bio(
        labels=["B-PER", "I-PER", "O"],
        scores=[0.9, 0.8, 0.99],
        offsets=[(0, 4), (5, 9), (10, 14)],
        score_floor=0.5,
    )
    assert len(entities) == 1
    assert (entities[0].start, entities[0].end) == (0, 9)


def test_span_score_is_the_weakest_token_not_the_mean() -> None:
    """Averaging lets one certain token drag a doubtful span over the line."""
    entities = _decode_bio(
        labels=["B-PER", "I-PER"],
        scores=[0.99, 0.55],
        offsets=[(0, 4), (5, 9)],
        score_floor=0.5,
    )
    assert entities[0].score == pytest.approx(0.55)


def test_a_new_b_tag_starts_a_new_entity() -> None:
    entities = _decode_bio(
        labels=["B-PER", "B-PER"],
        scores=[0.9, 0.9],
        offsets=[(0, 4), (5, 9)],
        score_floor=0.5,
    )
    assert len(entities) == 2


def test_a_label_change_starts_a_new_entity() -> None:
    entities = _decode_bio(
        labels=["B-PER", "I-LOC"],
        scores=[0.9, 0.9],
        offsets=[(0, 4), (5, 9)],
        score_floor=0.5,
    )
    assert [e.label for e in entities] == ["PER", "LOC"]


def test_special_tokens_with_zero_width_offsets_are_skipped() -> None:
    entities = _decode_bio(
        labels=["O", "B-PER", "O"],
        scores=[1.0, 0.9, 1.0],
        offsets=[(0, 0), (0, 5), (0, 0)],
        score_floor=0.5,
    )
    assert len(entities) == 1
    assert entities[0].start == 0


def test_entities_below_the_floor_are_dropped() -> None:
    entities = _decode_bio(labels=["B-PER"], scores=[0.3], offsets=[(0, 4)], score_floor=0.5)
    assert entities == []


def test_an_entity_running_to_the_end_of_the_sequence_is_emitted() -> None:
    entities = _decode_bio(
        labels=["O", "B-PER", "I-PER"],
        scores=[0.9, 0.9, 0.9],
        offsets=[(0, 2), (3, 7), (8, 12)],
        score_floor=0.5,
    )
    assert len(entities) == 1
    assert entities[0].end == 12


def test_all_outside_yields_nothing() -> None:
    assert (
        _decode_bio(labels=["O", "O"], scores=[0.9, 0.9], offsets=[(0, 2), (3, 5)], score_floor=0.5)
        == []
    )


def test_label_map_covers_the_camelbert_tagset() -> None:
    assert LABEL_MAP["PER"] == "AR_PERSON"
    assert LABEL_MAP["LOC"] == "AR_LOCATION"
    assert LABEL_MAP["ORG"] == "AR_ORG"
    # CAMeLBERT variants use PERS as well as PER.
    assert LABEL_MAP["PERS"] == "AR_PERSON"


# ---------------------------------------------------------------------------
# Availability: a tier switched on must never silently do nothing
# ---------------------------------------------------------------------------


def test_missing_model_artifact_raises_at_construction(tmp_path: Path) -> None:
    """Presidio's EntityRecognizer.__init__ calls load(), so this is startup.

    Failing here is the point: a tier the operator switched on must not come up
    inert and silently detect nothing.
    """
    with pytest.raises(Tier2Unavailable, match="model artifact is incomplete"):
        ArabicNerRecognizer(model_dir=tmp_path, supported_language="ar")


def test_the_error_names_every_missing_file(tmp_path: Path) -> None:
    (tmp_path / "model.onnx").write_bytes(b"not really a model")

    with pytest.raises(Tier2Unavailable) as caught:
        ArabicNerRecognizer(model_dir=tmp_path, supported_language="ar")

    message = str(caught.value)
    assert "tokenizer.json" in message
    assert "labels.json" in message
    assert "model.onnx" not in message  # this one was present


def test_enabling_tier2_without_a_model_dir_raises(policy: PolicyBundle) -> None:
    from pii_service.settings import Settings

    settings = Settings(
        audit_pepper="a" * 32,  # type: ignore[arg-type]
        enable_tier2_arabic_ner=True,
        tier2_model_dir=None,
    )
    with pytest.raises(Tier2Unavailable, match="PII_TIER2_MODEL_DIR is unset"):
        build_tier2_factory(settings)


def test_analyze_is_inert_without_a_session(tmp_path: Path) -> None:
    """Defence in depth for the guard in analyze().

    Construction can no longer produce an unloaded recognizer, so this bypasses
    __init__ to reach the guard directly -- it is the last thing standing
    between a half-initialised recognizer and an AttributeError inside the
    request path.
    """
    recognizer = ArabicNerRecognizer.__new__(ArabicNerRecognizer)
    recognizer._session = None

    assert recognizer.analyze("محمد علي", ["AR_PERSON"]) == []


# ---------------------------------------------------------------------------
# Real ONNX inference, against a deterministic stand-in model
# ---------------------------------------------------------------------------
#
# These execute the code that actually runs in production -- onnxruntime
# session, tokenizer offsets, BIO decoding, script segmentation, offset
# rebasing -- without downloading a 400MB checkpoint. See tests/fake_ner_model.py.


def _require_onnx() -> None:
    """Skip visibly rather than pretend tier 2 is covered.

    A silent gap here is the worst outcome: it would look like the ONNX path is
    tested when it is not. CI installs the extras, so these always run there.
    """
    pytest.importorskip("onnx", reason="pip install -e '.[dev,ner]' to cover ONNX inference")
    pytest.importorskip("onnxruntime", reason="pip install -e '.[dev,ner]'")


@pytest.fixture(scope="module")
def ner(tmp_path_factory: pytest.TempPathFactory) -> ArabicNerRecognizer:
    _require_onnx()
    from fake_ner_model import build_fake_ner_model

    directory = build_fake_ner_model(
        tmp_path_factory.mktemp("fake-ner"),
        word_labels={
            "محمد": "B-PER",
            "علي": "I-PER",
            "أحمد": "B-PER",
            "القاهرة": "B-LOC",
            "الجيزة": "B-LOC",
            "وزارة": "B-ORG",
            "الصحة": "I-ORG",
        },
    )
    return ArabicNerRecognizer(model_dir=directory, supported_language="ar")


NER_ENTITIES = ["AR_PERSON", "AR_LOCATION", "AR_ORG"]


def _found(recognizer: ArabicNerRecognizer, text: str) -> list[tuple[str, str]]:
    return [
        (r.entity_type, text[r.start : r.end])
        for r in sorted(recognizer.analyze(text, NER_ENTITIES), key=lambda r: r.start)
    ]


def test_a_single_token_entity_is_found(ner: ArabicNerRecognizer) -> None:
    assert _found(ner, "زار أحمد المكان") == [("AR_PERSON", "أحمد")]


def test_a_multi_token_entity_merges_into_one_span(ner: ArabicNerRecognizer) -> None:
    """B-PER followed by I-PER is one person, not two."""
    assert _found(ner, "اجتمع محمد علي هناك") == [("AR_PERSON", "محمد علي")]


def test_labels_map_to_our_entity_types(ner: ArabicNerRecognizer) -> None:
    found = _found(ner, "محمد في القاهرة مع وزارة الصحة")
    assert found == [
        ("AR_PERSON", "محمد"),
        ("AR_LOCATION", "القاهرة"),
        ("AR_ORG", "وزارة الصحة"),
    ]


def test_clean_text_produces_nothing(ner: ArabicNerRecognizer) -> None:
    assert _found(ner, "لا يوجد شيء مهم هنا") == []


def test_offsets_are_rebased_across_a_latin_prefix(ner: ArabicNerRecognizer) -> None:
    """The Arabic run starts partway in; its offsets must be shifted back."""
    text = "Please review this record for محمد علي today"
    found = ner.analyze(text, NER_ENTITIES)

    assert len(found) == 1
    assert text[found[0].start : found[0].end] == "محمد علي"


def test_two_arabic_runs_around_latin_both_report(ner: ArabicNerRecognizer) -> None:
    text = "محمد wrote the patch and القاهرة is the site"
    assert {surface for _, surface in _found(ner, text)} == {"محمد", "القاهرة"}


def test_latin_only_text_is_never_sent_to_the_model(ner: ArabicNerRecognizer) -> None:
    assert _found(ner, "Please review this pull request today") == []


def test_entities_not_requested_are_not_returned(ner: ArabicNerRecognizer) -> None:
    found = ner.analyze("محمد في القاهرة", ["AR_LOCATION"])
    assert [r.entity_type for r in found] == ["AR_LOCATION"]


def test_empty_text_is_inert(ner: ArabicNerRecognizer) -> None:
    assert ner.analyze("", NER_ENTITIES) == []


def test_every_span_slices_back_to_real_text(ner: ArabicNerRecognizer) -> None:
    """The invariant the whole offset design exists for, through the model path."""
    for text in (
        "محمد علي في القاهرة",
        "record: محمد، الجيزة",
        "أحمد wrote it, القاهرة hosted it",
        "  محمد  ",
    ):
        for result in ner.analyze(text, NER_ENTITIES):
            assert 0 <= result.start < result.end <= len(text)
            assert text[result.start : result.end].strip()


def test_the_recognizer_name_reaches_the_audit_row(ner: ArabicNerRecognizer) -> None:
    result = ner.analyze("زار أحمد المكان", NER_ENTITIES)[0]
    assert result.recognition_metadata["recognizer_name"] == "ArabicNerRecognizer"


def test_a_low_confidence_model_is_filtered_by_the_score_floor(
    tmp_path: Path,
) -> None:
    _require_onnx()
    from fake_ner_model import build_fake_ner_model

    directory = build_fake_ner_model(
        tmp_path / "unsure", word_labels={"محمد": "B-PER"}, confidence=0.05
    )
    recognizer = ArabicNerRecognizer(model_dir=directory, supported_language="ar", score_floor=0.9)
    assert recognizer.analyze("زار محمد المكان", NER_ENTITIES) == []
