#!/usr/bin/env python3
"""Fetch and prepare model artifacts at deploy time.

For a host that has internet. It runs once before pii-service starts, writes
into the shared models volume, and exits. It lives in the package rather than
in scripts/ so it is inside the image without a second copy. Both artifacts are skipped if they
are already there, so a restart costs nothing and a half-finished download is
retried rather than half-used.

    python -m pii_service.bootstrap --output /models

This is the deploy-time counterpart to the build-time export scripts. An
air-gapped site does not run it: it bakes the artifacts into the image or
carries them in, which is what scripts/export_camelbert_onnx.py is for. A
workstation with internet runs it and waits a few minutes once.

What it produces:

    <output>/camelbert-msa-int8/     tier 2, ONNX int8   (~106 MB)
    <output>/gliner-multi-pii/       tier 3, torch        (~1.2 GB)

Tier 2's artifact is converted here rather than downloaded, because there is
no published ONNX build of CAMeLBERT. That needs the conversion toolchain --
the [bootstrap] extra -- which is why this runs in its own container rather
than inside the service.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

# The variant the evaluation chose. See the README, "Result -- camelbert-msa-int8".
CAMELBERT_REPO = "CAMeL-Lab/bert-base-arabic-camelbert-msa-ner"
CAMELBERT_DIR = "camelbert-msa-int8"

GLINER_REPO = "urchade/gliner_multi_pii-v1"
GLINER_DIR = "gliner-multi-pii"


def _done(marker: Path) -> bool:
    return marker.is_file() and marker.stat().st_size > 0


def fetch_camelbert(output: Path) -> None:
    """Export CAMeLBERT to ONNX int8, quantized for avx2.

    avx2 rather than avx512_vnni, and this is not a performance choice:
    quantizing for a CPU feature the host does not have produces a model that
    loads, runs, reports no error and predicts nothing. That shipped once at
    0.0% coverage. avx2 is correct on every x86-64 CPU.
    """
    destination = output / CAMELBERT_DIR
    if _done(destination / "model.onnx"):
        print(f"[tier2] {destination} already present, skipping")
        return

    from onnxruntime.quantization import QuantType, quantize_dynamic
    from optimum.onnxruntime import ORTModelForTokenClassification
    from transformers import AutoConfig, AutoTokenizer

    print(f"[tier2] downloading and converting {CAMELBERT_REPO}")
    destination.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as staging_name:
        staging = Path(staging_name)
        model = ORTModelForTokenClassification.from_pretrained(CAMELBERT_REPO, export=True)
        model.save_pretrained(staging)
        AutoTokenizer.from_pretrained(CAMELBERT_REPO).save_pretrained(staging)

        source = next(iter(sorted(staging.glob("*.onnx"))), None)
        if source is None:
            raise SystemExit("[tier2] conversion produced no .onnx")

        print("[tier2] quantizing (dynamic, avx2, reduce_range=True)")
        quantize_dynamic(
            model_input=str(source),
            model_output=str(destination / "model.onnx"),
            weight_type=QuantType.QInt8,
            reduce_range=True,
            extra_options={"MatMulConstBOnly": True},
        )
        shutil.copy(staging / "tokenizer.json", destination / "tokenizer.json")

        config = AutoConfig.from_pretrained(CAMELBERT_REPO)
        labels = [config.id2label[index] for index in sorted(config.id2label)]
        (destination / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    _verify_tier2(destination)


def _verify_tier2(destination: Path) -> None:
    """Load it exactly as the service will, and fail here if it is inert.

    An artifact quantized for the wrong target loads cleanly and predicts "O"
    for every token. Finding that out now, on a host that can retry, is the
    entire point of this check.
    """
    from pii_service.detect.tier2_arabic_ner import ArabicNerRecognizer

    recognizer = ArabicNerRecognizer(model_dir=destination, supported_language="ar")
    found = recognizer.analyze("محمد علي يسكن في القاهرة", ["AR_PERSON", "AR_LOCATION"])
    print(f"[tier2] smoke: {len(found)} entities on a two-entity sentence")
    if not found:
        raise SystemExit(
            f"[tier2] {destination} predicted nothing on an obvious sentence. "
            "The artifact is inert; delete it and re-run rather than serving it."
        )


def fetch_gliner(output: Path) -> None:
    destination = output / GLINER_DIR
    if _done(destination / "pytorch_model.bin"):
        print(f"[tier3] {destination} already present, skipping")
        return

    try:
        from gliner import GLiNER
    except ImportError:
        print("[tier3] gliner is not installed in this image; skipping tier 3 weights")
        return

    print(f"[tier3] downloading {GLINER_REPO} (~1.2 GB)")
    GLiNER.from_pretrained(GLINER_REPO).save_pretrained(destination)

    # Same discipline: prove it loads offline, the way the service loads it.
    model = GLiNER.from_pretrained(str(destination), local_files_only=True)
    found = model.predict_entities(
        "Contact Sarah Mitchell at Acme Corp in Berlin.",
        ["person name", "organization", "location"],
    )
    print(f"[tier3] smoke: {len(found)} entities on a three-entity sentence")
    if len(found) < 3:
        raise SystemExit(f"[tier3] {destination} is degraded; delete it and re-run.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("/models"))
    parser.add_argument("--skip-tier2", action="store_true")
    parser.add_argument("--skip-tier3", action="store_true")
    arguments = parser.parse_args()

    arguments.output.mkdir(parents=True, exist_ok=True)
    if not arguments.skip_tier2:
        fetch_camelbert(arguments.output)
    if not arguments.skip_tier3:
        fetch_gliner(arguments.output)

    print("[bootstrap] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
