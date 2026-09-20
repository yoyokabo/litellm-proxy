#!/usr/bin/env python3
"""Export a CAMeLBERT NER variant to ONNX, optionally int8-quantized.

Run this on a machine with internet access, then ship the output directory to
the air-gapped host. **This script is the only place PyTorch is allowed**: it
is a build-time tool, and its dependencies never enter the runtime image (brief
§4). The service loads the artifact with onnxruntime and the Rust tokenizers
binding, and nothing else.

    pip install "torch --index-url https://download.pytorch.org/whl/cpu"
    pip install transformers "optimum[onnxruntime]"
    pip install -e "services/pii-service[ner]"

The third line is not optional: every export ends by loading the artifact
through the service's own ArabicNerRecognizer, so that a directory which cannot
be loaded fails here, on a machine with internet, rather than at startup on an
air-gapped host where nothing can be re-downloaded.

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


# Quantization target -> (AutoQuantizationConfig factory name, reduce_range).
#
# THIS IS NOT A PERFORMANCE KNOB. Quantizing for a target the runtime CPU does
# not have produces a model that loads, runs, reports no error, and predicts
# nothing -- u8s8 matmuls saturate without VNNI, the logits collapse to roughly
# +/-1, and every token decodes as "O". A tier-2 recognizer that silently
# returns zero spans is total Arabic name-detection loss with nothing in the
# logs to say so, which is the exact failure mode brief §13 says to flag rather
# than risk.
#
# Hence the default is avx2 with reduce_range=True: correct on every x86-64
# CPU, including those that do have VNNI, at a small throughput cost. Opt into
# avx512_vnni only when the *deployment* host is known to have it -- check
# `grep -o 'avx512_vnni' /proc/cpuinfo` on that machine, not on the build one.
QUANTIZATION_TARGETS: dict[str, tuple[str, bool]] = {
    "avx2": ("avx2", True),
    "avx512": ("avx512", True),
    "avx512_vnni": ("avx512_vnni", False),
    "arm64": ("arm64", False),
}

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


def export(
    variant: Variant,
    precision: str,
    output_root: Path,
    revision: str | None,
    quantization_target: str = "avx2",
) -> Path:
    try:
        from optimum.onnxruntime import ORTModelForTokenClassification, ORTQuantizer
        from optimum.onnxruntime.configuration import AutoQuantizationConfig
        from transformers import AutoConfig, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - build-time tool
        raise SystemExit(
            "missing build-time dependencies. Install them with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            '  pip install transformers "optimum[onnxruntime]"\n'
            '  pip install -e "services/pii-service[ner]"'
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
        factory_name, reduce_range = QUANTIZATION_TARGETS[quantization_target]
        print(
            f"[{variant.key}/{precision}] quantizing "
            f"(dynamic, {quantization_target}, reduce_range={reduce_range})"
        )
        quantizer = ORTQuantizer.from_pretrained(staging)
        quantizer.quantize(
            save_dir=staging,
            quantization_config=getattr(AutoQuantizationConfig, factory_name)(
                is_static=False, per_channel=True, reduce_range=reduce_range
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
                "quantization_target": quantization_target if precision == "int8" else None,
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
    """Load the artifact exactly as the service will, and fail here if it cannot.

    The entity count is asserted, not merely printed. A quantized model built
    for the wrong CPU loads cleanly and predicts "O" for every token, so
    "it exported without an error" is not evidence that it works -- and the
    place to find that out is here, on the build machine, rather than on the
    air-gapped host where the artifact cannot be rebuilt.

    This is a tripwire, not an evaluation. Two entities in one sentence says
    the artifact is not inert; whether it is *good enough to ship* is decided
    by scripts/eval_arabic_ner.py against the gold set.
    """
    sys.path.insert(0, str(REPO_ROOT / "services" / "pii-service" / "src"))
    from pii_service.detect.tier2_arabic_ner import ArabicNerRecognizer

    recognizer = ArabicNerRecognizer(model_dir=destination, supported_language="ar")
    found = recognizer.analyze("محمد علي يسكن في القاهرة", ["AR_PERSON", "AR_LOCATION"])
    print(f"    smoke: {len(found)} entities on a two-entity sentence")
    if not found:
        raise SystemExit(
            f"{destination} predicted nothing on a sentence with an obvious person and "
            "place name. The artifact is inert -- do not ship it.\n"
            "The usual cause is quantizing for a CPU feature the machine does not have: "
            "u8s8 matmuls saturate without VNNI and every token decodes as 'O'. "
            "Re-export with --quantization-target avx2, which is correct everywhere."
        )


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
    parser.add_argument(
        "--quantization-target",
        choices=sorted(QUANTIZATION_TARGETS),
        default="avx2",
        help="CPU feature set to quantize int8 for. Default avx2, which is correct on "
        "every x86-64 CPU. Only pick avx512_vnni if the DEPLOYMENT host has VNNI -- "
        "a mismatch yields a model that silently predicts nothing.",
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
        export(
            variant,
            precision,
            arguments.output,
            arguments.revision,
            arguments.quantization_target,
        )

    print(f"\n{len(jobs)} artifact(s) written to {arguments.output}")
    print("Next:  python scripts/eval_arabic_ner.py --all")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
