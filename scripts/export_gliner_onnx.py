#!/usr/bin/env python3
"""Export GLiNER2 to ONNX so tier 3 runs without PyTorch.

Run on a machine with internet, ship the output directory to the air-gapped
host. **This script and scripts/export_camelbert_onnx.py are the only places
PyTorch is allowed**; the runtime image has neither torch nor transformers, and
tests/test_deployment.py fails the build if either appears.

    pip install torch --index-url https://download.pytorch.org/whl/cpu
    pip install "gliner>=0.2.29" onnx onnxruntime tiktoken protobuf sentencepiece

    python scripts/export_gliner_onnx.py

Why this exists rather than `pip install gliner` in the image: the gliner
package declares torch and transformers as *core* dependencies, and its own
ONNX runtime adapter imports torch anyway. Installing it would put a
multi-gigabyte deep-learning stack in a CPU inference image to run a graph that
onnxruntime can execute alone. So the graph is exported here and
detect/tier3_gliner.py reimplements the pre- and post-processing in numpy --
about eighty lines, all of it verified against the torch model's own output.

Output layout, which is what ``GlinerRecognizer`` expects:

    models/gliner-multi-pii/
        model.onnx
        tokenizer.json
        gliner.json      # max_width, the two marker token ids, provenance
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "models" / "gliner-multi-pii"
DEFAULT_REPO = "urchade/gliner_multi_pii-v1"

# The sentence the artifact must not be silent on. Same tripwire as the
# CAMeLBERT export: an inert model loads cleanly and predicts nothing, and the
# place to find that out is here rather than on a host that cannot rebuild it.
SMOKE_TEXT = "Contact Sarah Mitchell at Acme Corp in Berlin."
SMOKE_LABELS = ("person name", "organization", "location")
SMOKE_MIN_ENTITIES = 3


def export(repo: str, output: Path, *, quantize: bool) -> Path:
    try:
        from gliner import GLiNER
    except ImportError as exc:  # pragma: no cover - build-time tool
        raise SystemExit(
            "missing build-time dependencies. Install them with:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            '  pip install "gliner>=0.2.29" onnx onnxruntime tiktoken protobuf sentencepiece'
        ) from exc

    output.mkdir(parents=True, exist_ok=True)
    staging = output / ".staging"
    staging.mkdir(parents=True, exist_ok=True)

    print(f"[gliner] loading {repo}")
    model = GLiNER.from_pretrained(repo)
    config = model.config

    if getattr(config, "span_mode", None) != "markerV0":
        raise SystemExit(
            f"unsupported span_mode {getattr(config, 'span_mode', None)!r}. "
            "detect/tier3_gliner.py decodes markerV0 span logits; another mode "
            "needs a matching decoder, not just a re-export."
        )

    print("[gliner] reference prediction (torch), for comparison after export:")
    for span in model.predict_entities(SMOKE_TEXT, list(SMOKE_LABELS)):
        print(f"    {span['label']:<14} {span['score']:.3f}  {span['text']!r}")

    print("[gliner] exporting to ONNX")
    model.export_to_onnx(staging, quantize=quantize)
    model.save_pretrained(staging)

    shutil.copy(staging / "model.onnx", output / "model.onnx")
    tokenizer_json = staging / "tokenizer.json"
    if not tokenizer_json.is_file():
        raise SystemExit(
            f"{tokenizer_json} is missing. The runtime uses the Rust tokenizers "
            "binding, which needs tokenizer.json specifically."
        )
    shutil.copy(tokenizer_json, output / "tokenizer.json")

    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(output / "tokenizer.json"))
    sep_id = tokenizer.token_to_id(config.sep_token)
    if sep_id is None:
        raise SystemExit(f"tokenizer has no id for sep_token {config.sep_token!r}")

    (output / "gliner.json").write_text(
        json.dumps(
            {
                "repo": repo,
                "span_mode": config.span_mode,
                "max_width": int(config.max_width),
                "max_len": int(getattr(config, "max_len", 384)),
                "ent_token_id": int(config.class_token_index),
                "sep_token_id": int(sep_id),
                "cls_token_id": int(tokenizer.token_to_id("[CLS]")),
                "eos_token_id": int(tokenizer.token_to_id("[SEP]")),
                "exported_by": Path(__file__).name,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    shutil.rmtree(staging, ignore_errors=True)
    _verify(output)
    print(f"[gliner] -> {output}")
    return output


def _verify(destination: Path) -> None:
    """Load the artifact exactly as the service will, and fail here if it cannot.

    The entity count is asserted, not printed. A quantized graph that saturates
    loads cleanly and returns almost nothing -- which is precisely how the
    CAMeLBERT int8 artifacts shipped at 0.0% coverage before this check existed.
    """
    sys.path.insert(0, str(REPO_ROOT / "services" / "pii-service" / "src"))
    from pii_service.detect.tier3_gliner import GlinerRecognizer

    recognizer = GlinerRecognizer(
        model_dir=destination,
        supported_language="en",
        prompts={"PERSON": "person name", "ORGANIZATION": "organization", "LOCATION": "location"},
    )
    found = recognizer.analyze(SMOKE_TEXT, ["PERSON", "ORGANIZATION", "LOCATION"])
    print(f"    smoke: {len(found)} entities on a three-entity sentence")
    if len(found) < SMOKE_MIN_ENTITIES:
        raise SystemExit(
            f"{destination} found {len(found)} of {SMOKE_MIN_ENTITIES} obvious entities. "
            "The artifact is degraded -- do not ship it.\n"
            "Dynamic int8 quantization of this model is known to destroy it: mdeberta-v3's "
            "disentangled attention does not survive it, and --quantize measured 0.68 on the "
            "one entity it still found while losing the other two entirely. Export fp32."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Also emit int8. Measured to destroy this model -- the smoke check "
        "will refuse the result. Kept so the finding can be reproduced.",
    )
    arguments = parser.parse_args()

    export(arguments.repo, arguments.output, quantize=arguments.quantize)
    print("\nNext:  set PII_ENABLE_TIER3_GLINER=true and PII_TIER3_MODEL_DIR")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
