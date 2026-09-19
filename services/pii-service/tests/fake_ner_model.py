"""Build a tiny, deterministic ONNX token-classification model for tests.

Tier 2's inference path -- ONNX session, tokenizer offsets, BIO decoding,
segment rebasing -- is the part most likely to mis-span an entity, and until
now none of it was executed by a test, because exercising it seemed to require
downloading a 400MB CAMeLBERT checkpoint.

It does not. A token classifier is a function from token id to logits, so a
single ONNX ``Gather`` over a lookup table is a complete, honest stand-in: the
recognizer cannot tell it from a real model, and the labels it emits are
whatever the test decides. That makes the real code path testable offline, in
CI, in about a millisecond.

What this deliberately does not test is model *accuracy*. That is what
``scripts/eval_arabic_ner.py`` and the gold set are for, against real weights.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer, models, pre_tokenizers

__all__ = ["DEFAULT_LABELS", "build_fake_ner_model"]

# The BIO tagset CAMeLBERT NER uses, in the id order its config declares.
DEFAULT_LABELS: tuple[str, ...] = (
    "O",
    "B-PER",
    "I-PER",
    "B-LOC",
    "I-LOC",
    "B-ORG",
    "I-ORG",
)


def build_fake_ner_model(
    directory: Path,
    *,
    word_labels: Mapping[str, str],
    labels: Sequence[str] = DEFAULT_LABELS,
    confidence: float = 8.0,
) -> Path:
    """Write model.onnx, tokenizer.json and labels.json into ``directory``.

    ``word_labels`` maps a whitespace-delimited word to the BIO tag the model
    should emit for it. Anything absent is tagged ``O``.

    ``confidence`` is the logit assigned to the chosen tag; softmax over the
    tagset turns it into the score the recognizer thresholds on, so a low value
    is how a test drives the score floor.
    """
    directory.mkdir(parents=True, exist_ok=True)
    label_index = {label: index for index, label in enumerate(labels)}

    unknown = "[UNK]"
    vocabulary: dict[str, int] = {unknown: 0}
    for word in word_labels:
        vocabulary.setdefault(word, len(vocabulary))

    tokenizer = Tokenizer(models.WordLevel(vocab=vocabulary, unk_token=unknown))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))

    # logits[token] = table[input_id], so each token's tag is a pure function of
    # its id -- deterministic, and trivially controllable from a test.
    table = np.zeros((len(vocabulary), len(labels)), dtype=np.float32)
    for word, identifier in vocabulary.items():
        tag = word_labels.get(word, "O")
        if tag not in label_index:
            raise ValueError(f"unknown tag {tag!r}; expected one of {list(labels)}")
        table[identifier, label_index[tag]] = confidence

    graph = helper.make_graph(
        nodes=[
            helper.make_node(
                "Gather", inputs=["label_table", "input_ids"], outputs=["logits"], axis=0
            )
        ],
        name="fake_token_classifier",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "sequence"]),
            helper.make_tensor_value_info(
                "attention_mask", TensorProto.INT64, ["batch", "sequence"]
            ),
        ],
        outputs=[
            helper.make_tensor_value_info(
                "logits", TensorProto.FLOAT, ["batch", "sequence", len(labels)]
            )
        ],
        initializer=[numpy_helper.from_array(table, name="label_table")],
    )

    model = helper.make_model(
        graph,
        producer_name="pii-service-tests",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 9
    onnx.checker.check_model(model)
    onnx.save(model, str(directory / "model.onnx"))

    (directory / "labels.json").write_text(json.dumps(list(labels)), encoding="utf-8")
    return directory
