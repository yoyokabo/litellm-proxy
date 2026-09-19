#!/usr/bin/env python3
"""Export a CAMeLBERT NER variant to ONNX, optionally int8-quantized.

Run this on a machine with internet access, then ship the output directory to
the air-gapped host. **This script is the only place PyTorch is allowed**: it
is a build-time tool, and its dependencies never enter the runtime image (brief
§4). The service loads the artifact with onnxruntime and the Rust tokenizers
binding, and nothing else.

    pip install "torch --index-url https://download.pytorch.org/whl/cpu"
    pip install transformers "optimum[onnxruntime]"

    # Both variants, fp32 and int8, ready for scripts/eval_arabic_ner.py
    python scripts/export_camelbert_onnx.py --all

    # One variant
    python scripts/export_camelbert_onnx.py --variant mix --precision int8

Output layout, which is what ``ArabicNerRecognizer`` expects:

    models/camelbert-<variant>-<precision>/
        model.onnx
        tokenizer.json
        labels.json      # class id -> BIO tag, ordered by id
        export.json      # provenance: repo, revision, precision, versions

Export both precisions and evaluate both. Dynamic int8 usually costs little,
but "usually" is not a measurement, and on this task a point of coverage is a
leak rather than a rounding error.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "models"


@dataclass(frozen=True)
class Variant:
    key: str
    repo: str
    description: str


VARIANTS: dict[str, Variant] = {
    "mix": Variant(
        key="mix",
        repo="CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
        description="pre-trained on MSA + dialectal + classical Arabic",
    ),
    "msa": Variant(
        key="msa",
        repo="CAMeL-Lab/bert-base-arabic-camelbert-msa-ner",
        description="pre-trained on Modern Standard Arabic only",
    ),
}


def export(variant: Variant, precision: str, output_root: Path, revision: str | None) -> Path:
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - build-time tool
        raise SystemExit(
            "missing build-time dependencies. Install them with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            '  pip install transformers "optimum[onnxruntime]"'
        ) from exc

    destination = output_root / f"camelbert-{variant.key}-{precision}"
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / ".staging"

    print(f"[{variant.key}/{precision}] exporting {variant.repo}")
    model = ORTModelForTokenClassification.from_pretrained(
        variant.repo, export=True, revision=revision
    )
    model.save_pretrained(staging)

    tokenizer = AutoTokenizer.from_pretrained(variant.repo, revision=revision)
    tokenizer.save_pretrained(staging)

    if precision == "int8":
        print(f"[{variant.key}/{precision}] quantizing (dynamic, avx512_vnni)")
        quantizer = ORTQuantizer.from_pretrained(staging)
        quantizer.quantize(
            save_dir=staging,
            quantization_config=AutoQuantizationConfig.avx512_vnni(
                is_static=False, per_channel=True
            ),
        )

    _collect(staging, destination, precision)

    config = AutoConfig.from_pretrained(variant.repo, revision=revision)
    labels = [config.id2label[index] for index in sorted(config.id2label)]
    (destination / "labels.json").write_text(json.dumps(labels, indent=2), encoding="utf-8")

    (destination / "export.json").write_text(
        json.dumps(
            {
                "variant": variant.key,
                "repo": variant.repo,
                "revision": revision or "main",
                "precision": precision,
                "labels": len(labels),
                "description": variant.description,
                "exported_by": Path(__file__).name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    shutil.rmtree(staging, ignore_errors=True)
    _verify(destination)
    print(f"[{variant.key}/{precision}] -> {destination}")
    return destination


def _collect(staging: Path, destination: Path, precision: str) -> None:
    """Move the files the service needs, under the names it expects."""
    candidates = sorted(staging.glob("*.onnx"))
    if not candidates:
        raise SystemExit(f"no .onnx produced in {staging}")

    quantized = [p for p in candidates if "quantized" in p.name or "int8" in p.name]
    chosen = (quantized or candidates)[0] if precision == "int8" else candidates[0]
    if precision == "int8" and not quantized:
        raise SystemExit(f"asked for int8 but quantization produced no quantized file in {staging}")

    shutil.copy(chosen, destination / "model.onnx")

    tokenizer_json = staging / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise SystemExit(
            f"{tokenizer_json} is missing. The runtime uses the Rust tokenizers binding, "
            "which needs tokenizer.json specifically -- a vocab.txt alone will not do."
        )
    shutil.copy(tokenizer_json, destination / "tokenizer.json")


def _verify(destination: Path) -> None:
    """Load the artifact exactly as the service will, and fail here if it cannot."""
    sys.path.insert(0, str(REPO_ROOT / "services" / "pii-service" / "src"))
    from pii_service.detect.tier2_arabic_ner import ArabicNerRecognizer

    recognizer = ArabicNerRecognizer(model_dir=destination, supported_language="ar")
    found = recognizer.analyze("محمد علي يسكن في القاهرة", ["AR_PERSON", "AR_LOCATION"])
    print(f"    smoke: {len(found)} entities on a two-entity sentence")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=sorted(VARIANTS), help="which variant to export")
    parser.add_argument(
        "--precision", choices=["fp32", "int8"], default="int8", help="default: int8"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="export every variant in both precisions (what the comparison needs)",
    )
    parser.add_argument("--revision", help="pin a model revision rather than main")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()

    if not arguments.all and not arguments.variant:
        parser.error("pass --variant, or --all to export everything")

    jobs = (
        [(v, p) for v in VARIANTS.values() for p in ("fp32", "int8")]
        if arguments.all
        else [(VARIANTS[arguments.variant], arguments.precision)]
    )

    for variant, precision in jobs:
        export(variant, precision, arguments.output, arguments.revision)

    print(f"\n{len(jobs)} artifact(s) written to {arguments.output}")
    print("Next:  python scripts/eval_arabic_ner.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
