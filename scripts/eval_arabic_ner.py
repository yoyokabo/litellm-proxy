#!/usr/bin/env python3
"""Compare Arabic NER variants on the gold set and pick one.

Settles the open decision from the brief (§13): which CAMeLBERT variant to
quantize and ship. Run it after ``scripts/export_camelbert_onnx.py --all``:

    python scripts/eval_arabic_ner.py --all
    python scripts/eval_arabic_ner.py --model models/camelbert-mix-int8 \\
                                      --model models/camelbert-msa-int8

Ranking (see ``pii_service.evaluation.scoring``): coverage first, because a
missed span is a leak; then false positives on entity-free text, because a
model that masks اليوم makes the gateway unusable; then p95 latency, because
this runs pre_call on every request.

Two things this deliberately does *not* do. It does not use exact-match F1 --
the brief is explicit that a sloppy boundary covering the name is a pass. And
it does not drive the model directly: it goes through the same
``ArabicNerRecognizer`` the service loads, so what is measured is the deployed
path including BIO decoding, script segmentation and offset rebasing, not a
parallel implementation that could be right where the service is wrong.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SERVICE_ROOT = REPO_ROOT / "services" / "pii-service"
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from pii_service.detect.tier2_arabic_ner import (  # noqa: E402
    LABEL_MAP,
    ArabicNerRecognizer,
)
from pii_service.evaluation.gold import GoldSet  # noqa: E402
from pii_service.evaluation.scoring import (  # noqa: E402
    MAX_ACCEPTABLE_FP_PER_SENTENCE,
    MIN_ACCEPTABLE_COVERAGE,
    Prediction,
    compare,
    register_breakdown,
    score_variant,
)

GOLD = SERVICE_ROOT / "eval" / "arabic_ner_gold.yaml"
DEFAULT_MODEL_ROOT = REPO_ROOT / "models"

# Our entity type -> the gold set's label.
ENTITY_TO_GOLD = {"AR_PERSON": "PER", "AR_LOCATION": "LOC", "AR_ORG": "ORG"}
ALL_ENTITIES = sorted(set(LABEL_MAP.values()))


def run_variant(model_dir: Path, gold: GoldSet, repeats: int) -> tuple[dict, dict]:
    """Run one model over every gold case. Returns (predictions, latencies)."""
    recognizer = ArabicNerRecognizer(model_dir=model_dir, supported_language="ar")

    predictions: dict[str, list[Prediction]] = {}
    latencies: dict[str, float] = {}

    # One untimed pass so warm-up does not land in the p95.
    for case in gold.cases[: min(5, len(gold.cases))]:
        recognizer.analyze(case.text, ALL_ENTITIES)

    for case in gold.cases:
        best = float("inf")
        found: list = []
        for _ in range(repeats):
            started = time.perf_counter()
            found = recognizer.analyze(case.text, ALL_ENTITIES)
            best = min(best, (time.perf_counter() - started) * 1000)

        predictions[case.case_id] = [
            Prediction(
                start=result.start,
                end=result.end,
                label=ENTITY_TO_GOLD.get(result.entity_type, result.entity_type),
                score=float(result.score),
            )
            for result in found
        ]
        latencies[case.case_id] = best

    return predictions, latencies


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, action="append", default=[], dest="models")
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"evaluate every camelbert-* directory under {DEFAULT_MODEL_ROOT}",
    )
    parser.add_argument("--gold", type=Path, default=GOLD)
    parser.add_argument("--repeats", type=int, default=3, help="timing repeats per case")
    parser.add_argument("--json", type=Path, help="write the full result as JSON")
    arguments = parser.parse_args()

    models = list(arguments.models)
    if arguments.all:
        models += sorted(
            p for p in DEFAULT_MODEL_ROOT.glob("camelbert-*") if (p / "model.onnx").is_file()
        )
    if not models:
        parser.error(
            "no models to evaluate. Run scripts/export_camelbert_onnx.py --all first, "
            "then pass --all or --model <dir>."
        )

    gold = GoldSet.load(arguments.gold)

    print("=" * 94)
    print("Arabic NER variant comparison")
    print("=" * 94)
    print(f"  gold set   {arguments.gold.relative_to(REPO_ROOT)}")
    print(
        f"  cases      {len(gold.cases)}  "
        f"({len(gold.positives)} with entities, {len(gold.negatives)} without)"
    )
    print(f"  spans      {gold.span_count}  {gold.label_counts()}")
    print(
        f"  gate       coverage >= {MIN_ACCEPTABLE_COVERAGE:.0%}, "
        f"false positives <= {MAX_ACCEPTABLE_FP_PER_SENTENCE:.2f}/clean sentence"
    )
    print()

    scores = []
    for model_dir in models:
        name = model_dir.name
        print(f"running {name} ...", flush=True)
        predictions, latencies = run_variant(model_dir, gold, arguments.repeats)
        scores.append(score_variant(name, gold, predictions, latencies))

    print()
    _print_table(scores)
    print()
    _print_registers(scores)

    verdict = compare(scores)
    print()
    print("=" * 94)
    if verdict.winner:
        print(f"VERDICT: {verdict.winner}")
        print(f"         {verdict.reason}")
        print()
        print("Set in .env:")
        print("  PII_ENABLE_TIER2_ARABIC_NER=true")
        print(f"  PII_TIER2_MODEL_DIR=/models/{verdict.winner}")
        print("and record the decision in the README's 'Tier 2 model artifact' section.")
    else:
        print("VERDICT: none — do not enable tier 2")
        print(f"         {verdict.reason}")
    print("=" * 94)

    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {"gold": str(arguments.gold), "cases": len(gold.cases), **verdict.as_dict()},
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        print(f"\nwrote {arguments.json}")

    return 0 if verdict.winner else 1


def _print_table(scores: list) -> None:
    header = (
        f"  {'variant':<26} {'coverage':>9} {'typed':>7} {'char rec':>9} "
        f"{'prec':>7} {'FP/clean':>9} {'p50 ms':>8} {'p95 ms':>8}  gate"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))
    for score in sorted(scores, key=lambda s: -s.coverage_any):
        gate = "pass" if score.is_acceptable else "FAIL"
        print(
            f"  {score.name:<26} {score.coverage_any:>8.1%} {score.coverage_typed:>7.1%} "
            f"{score.char_recall:>8.1%} {score.span_precision:>7.1%} "
            f"{score.false_positives_per_clean_sentence:>9.2f} "
            f"{score.p50_ms:>8.1f} {score.p95_ms:>8.1f}  {gate}"
        )
    for score in scores:
        for reason in score.rejection_reasons():
            print(f"    ! {score.name}: {reason}")


def _print_registers(scores: list) -> None:
    print("  coverage by register (Egyptian first — it is what this deployment sees)")
    print(f"  {'variant':<26} " + " ".join(f"{r:>16}" for r in ("egyptian", "mixed_script", "msa")))
    print("  " + "-" * 80)
    for score in sorted(scores, key=lambda s: -s.coverage_any):
        cells = dict((name, cov) for name, cov, _ in register_breakdown(score))
        row = " ".join(f"{cells.get(r, 0.0):>15.1%} " for r in ("egyptian", "mixed_script", "msa"))
        print(f"  {score.name:<26} {row}")


if __name__ == "__main__":
    raise SystemExit(main())
