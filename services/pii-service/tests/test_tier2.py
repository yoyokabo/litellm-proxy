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
