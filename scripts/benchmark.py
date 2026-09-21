#!/usr/bin/env python3
"""Detection latency benchmark: p50/p95 per recognizer tier, on CPU.

Done-criterion 5 of the brief, and the input to a decision the brief
deliberately refuses to make in advance: whether tier 3 (GLiNER2) is shippable.
The vendor reports 50-200 ms/document on 8-16 core CPU; an independent
measurement found ~2.3 s per chat-sized message on an M1. That is more than an
order of magnitude, and the guardrail runs pre_call, so whatever it is lands on
every request through the gateway. Measure before anyone designs around a
number.

Run it on hardware that resembles production. A number from a laptop is not
evidence about a GPU host's CPU allocation.

    python scripts/benchmark.py
    python scripts/benchmark.py --iterations 500 --json results.json

Tiers 2 and 3 are included only when enabled and their artifacts are present;
otherwise the run says so rather than quietly reporting tier-1 numbers under a
"full pipeline" heading.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "services" / "pii-service" / "src"))

from pii_service.detect.normalize import DIGITS, normalize  # noqa: E402
from pii_service.detect.registry import build_analyzer  # noqa: E402
from pii_service.detect.router import PiiRouter  # noqa: E402
from pii_service.policy.loader import load_policy_bundle  # noqa: E402
from pii_service.synthetic import (  # noqa: E402
    synthetic_iban,
    synthetic_mobile,
    synthetic_national_id,
    synthetic_tax_id,
    to_arabic_indic,
)

CONFIG_DIR = REPO_ROOT / "services" / "pii-service" / "config"


@dataclass
class Measurement:
    name: str
    samples: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    max_ms: float
    throughput_per_s: float

    def render(self) -> str:
        return (
            f"  {self.name:<34} "
            f"p50 {self.p50_ms:7.2f}ms  "
            f"p95 {self.p95_ms:7.2f}ms  "
            f"p99 {self.p99_ms:7.2f}ms  "
            f"max {self.max_ms:7.2f}ms  "
            f"{self.throughput_per_s:8.1f}/s"
        )


def measure(name: str, operation: Callable[[str], object], corpus: list[str]) -> Measurement:
    # One untimed pass: the first call compiles patterns and warms caches, and
    # including it would put a one-off cost into the p99 of every tier.
    for text in corpus[: min(10, len(corpus))]:
        operation(text)

    timings: list[float] = []
    started = time.perf_counter()
    for text in corpus:
        begin = time.perf_counter()
        operation(text)
        timings.append((time.perf_counter() - begin) * 1000)
    wall = time.perf_counter() - started

    ordered = sorted(timings)
    return Measurement(
        name=name,
        samples=len(timings),
        p50_ms=round(statistics.median(ordered), 3),
        p95_ms=round(_quantile(ordered, 0.95), 3),
        p99_ms=round(_quantile(ordered, 0.99), 3),
        mean_ms=round(statistics.fmean(ordered), 3),
        max_ms=round(ordered[-1], 3),
        throughput_per_s=round(len(timings) / wall, 1) if wall else 0.0,
    )


def _quantile(ordered: list[float], q: float) -> float:
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, round(q * (len(ordered) - 1)))
    return ordered[index]


def build_corpus(size: int, seed: int = 20260919) -> dict[str, list[str]]:
    """Messages shaped like real traffic, with synthetic PII at a realistic rate.

    Not every message contains PII. A corpus where everything matches measures
    the wrong thing: most of the gateway's traffic is ordinary prompts, and the
    cost of *not* finding anything is what dominates the average.
    """
    rng = random.Random(seed)

    def nid() -> str:
        return synthetic_national_id(rng=rng)

    arabic_templates = [
        "عايز أعرف تفاصيل الحساب ده",
        "من فضلك راجع الطلب وقولي رأيك",
        "الرقم القومي {nid} لو سمحت سجله",
        "موبايل العميل {mobile} والعنوان شارع الهرم، الجيزة",
        "البيانات: الاسم محمد علي، الرقم القومي {nid}، موبايل {mobile}",
        "اكتبلي إيميل رسمي للعميل بخصوص الشكوى",
        "حساب بنكي {iban} والرقم الضريبي {tax}",
    ]
    latin_templates = [
        "Can you refactor this function to use a generator?",
        "Summarise the attached quarterly report in five bullets",
        "The customer's national ID is {nid}, please verify",
        "Contact: {mobile}, account {iban}",
        "Write a unit test for the retry logic",
        "tax id {tax} for the invoice",
        "Explain the difference between a mutex and a semaphore",
    ]
    mixed_templates = [
        "Please translate: الرقم القومي {nid}",
        "Customer محمد wants a refund, mobile {mobile}",
        "Log entry: user=ahmed id={nid} status=ok",
    ]

    def render(template: str) -> str:
        text = template.format(
            nid=nid(),
            mobile=synthetic_mobile(rng=rng),
            iban=synthetic_iban(rng=rng),
            tax=synthetic_tax_id(rng=rng),
        )
        # Half of Arabic numerals in the wild are written Arabic-Indic.
        return to_arabic_indic(text) if rng.random() < 0.4 and "ا" in text else text

    def build(templates: list[str]) -> list[str]:
        return [render(rng.choice(templates)) for _ in range(size)]

    long_document = " ".join(render(rng.choice(arabic_templates)) for _ in range(40))

    return {
        "arabic (chat-sized)": build(arabic_templates),
        "latin (chat-sized)": build(latin_templates),
        "mixed script": build(mixed_templates),
        "arabic (long document)": [long_document] * max(1, size // 10),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200, help="messages per corpus")
    parser.add_argument("--json", type=Path, help="also write results as JSON")
    parser.add_argument("--seed", type=int, default=20260919)
    arguments = parser.parse_args()

    policy = load_policy_bundle(CONFIG_DIR)
    corpora = build_corpus(arguments.iterations, arguments.seed)

    print("=" * 96)
    print("pii-service detection latency")
    print("=" * 96)
    print(f"  python     {platform.python_version()}  ({platform.machine()})")
    print(f"  cpu        {_cpu_description()}")
    print(f"  iterations {arguments.iterations} per corpus")
    print(f"  entities   {len(policy.entities)}")
    print()

    tiers = _available_tiers(policy)
    results: dict[str, list[Measurement]] = {}

    for corpus_name, corpus in corpora.items():
        print(
            f"{corpus_name}  ({len(corpus)} messages, "
            f"mean {statistics.fmean(len(t) for t in corpus):.0f} chars)"
        )
        measurements = [measure("normalize (DIGITS) only", lambda t: normalize(t, DIGITS), corpus)]
        for tier_name, router in tiers:
            measurements.append(measure(tier_name, lambda t, r=router: r.analyze(t), corpus))

        for measurement in measurements:
            print(measurement.render())
        print()
        results[corpus_name] = measurements

    _print_guidance(results)

    if arguments.json:
        arguments.json.write_text(
            json.dumps(
                {
                    "python": platform.python_version(),
                    "machine": platform.machine(),
                    "cpu": _cpu_description(),
                    "iterations": arguments.iterations,
                    "results": {
                        name: [asdict(m) for m in measurements]
                        for name, measurements in results.items()
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"wrote {arguments.json}")

    return 0


def _available_tiers(policy: object) -> list[tuple[str, PiiRouter]]:
    """Build one router per tier combination that can actually run here."""
    today = date.today()
    tiers: list[tuple[str, PiiRouter]] = []

    tier1 = build_analyzer(policy, today=today)  # type: ignore[arg-type]
    tiers.append(("tier 1 (deterministic)", PiiRouter(tier1, policy)))  # type: ignore[arg-type]

    from pii_service.settings import Settings

    try:
        settings = Settings()
    except Exception:
        # The benchmark measures detection and never fingerprints anything, so
        # the pepper's value is irrelevant here -- but Settings validates it on
        # construction. Substitute a throwaway one rather than refusing to
        # report tier 2/3 numbers on a developer machine with no .env.
        settings = Settings(audit_pepper="benchmark-only-not-a-real-pepper")  # type: ignore[arg-type]

    # Pin the policy directory the same way the analyzer above does. Settings
    # defaults it to a *relative* "config", which resolves only when the cwd
    # happens to be services/pii-service -- so without this, running the
    # benchmark from the repository root (as the README says to) reports
    # "tier 2 not measured" and looks like a missing artifact rather than a
    # wrong working directory.
    settings = settings.model_copy(update={"config_dir": CONFIG_DIR})

    for label, flag, builder in (
        ("tier 1 + 2 (arabic NER)", settings.enable_tier2_arabic_ner, _tier2),
        ("tier 1 + 3 (GLiNER2)", settings.enable_tier3_gliner, _tier3),
    ):
        if not flag:
            print(f"  note: {label} not measured -- disabled in settings")
            continue
        try:
            analyzer = builder(policy, settings, today)
        except Exception as exc:  # artifact or wheel missing
            print(f"  note: {label} not measured -- {type(exc).__name__}: {exc}")
            continue
        tiers.append((label, PiiRouter(analyzer, policy)))  # type: ignore[arg-type]

    print()
    return tiers


def _tier2(policy: object, settings: object, today: date) -> object:
    from pii_service.detect.tier2_arabic_ner import build_tier2_factory

    return build_analyzer(
        policy,  # type: ignore[arg-type]
        today=today,
        tier2_factory=build_tier2_factory(settings),  # type: ignore[arg-type]
    )


def _tier3(policy: object, settings: object, today: date) -> object:
    from pii_service.detect.tier3_gliner import build_tier3_factory

    return build_analyzer(
        policy,  # type: ignore[arg-type]
        today=today,
        tier3_factory=build_tier3_factory(settings),  # type: ignore[arg-type]
    )


def _cpu_description() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                import os

                return f"{line.split(':', 1)[1].strip()} x{os.cpu_count()}"
    except OSError:
        pass
    return platform.processor() or "unknown"


def _print_guidance(results: dict[str, list[Measurement]]) -> None:
    """Turn the numbers into the decision the brief actually wants."""
    chat = results.get("arabic (chat-sized)") or next(iter(results.values()), [])
    gliner = next((m for m in chat if "GLiNER" in m.name), None)

    print("-" * 96)
    if gliner is None:
        print("tier 3 was not measured. Do not enable it, and do not let anyone plan")
        print("around a latency figure, until this benchmark has run on production-like")
        print("hardware with PII_ENABLE_TIER3_GLINER=true.")
    elif gliner.p95_ms > 500:
        print(f"tier 3 p95 is {gliner.p95_ms:.0f}ms. This guardrail runs pre_call, so that")
        print("is added to every request through the gateway. Keep it behind its feature")
        print("flag and default it off, as the brief anticipated.")
    else:
        print(f"tier 3 p95 is {gliner.p95_ms:.0f}ms, which is within the vendor's claimed")
        print("range. Enabling it is defensible -- re-measure under concurrent load first.")
    print("-" * 96)


if __name__ == "__main__":
    raise SystemExit(main())
